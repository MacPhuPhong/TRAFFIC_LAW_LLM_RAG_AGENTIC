# -*- coding: utf-8 -*-
"""
retriever_pgvector.py — Postgres pgvector backend mirror of retriever.py.

Same public interface as `TrafficHybridRetriever`:
  - constructor signature (most kwargs identical; differs in `pg_url` instead of qdrant_*)
  - `get_relevant_chunks(query, top_k, filters)` returning `RetrievedChunk[]`
  - `retrieve` alias
  - `get_chunks_by_location`

Switch in api/main.py via env var `VECTOR_DB`:
    VECTOR_DB=pgvector  → PgVectorHybridRetriever (uses VECTOR_DB_URL)
    VECTOR_DB=qdrant   (default) → TrafficHybridRetriever (uses QDRANT_*)
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

try:
    from langsmith import traceable as _ls_traceable
except ImportError:
    def _ls_traceable(*dargs, **dkwargs):
        if dargs and callable(dargs[0]) and not dkwargs:
            return dargs[0]
        def _wrap(fn):
            return fn
        return _wrap

from .retriever import (
    RetrievedChunk,
    VI_STOPWORDS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_JSONL,
    _e5_query_prefix,
)

logger = logging.getLogger(__name__)

DEFAULT_TABLE = "traffic_chunks"


class PgVectorHybridRetriever:
    """
    Hybrid retriever combining Supabase pgvector dense search
    with the same in-process BM25 lexical index used in TrafficHybridRetriever.
    Scores fused via Reciprocal Rank Fusion (RRF).

    BM25 row index aligns with the `id` column in `traffic_chunks` (insertion
    order from all_chunks.jsonl). Same convention as the Qdrant pipeline.
    """

    def __init__(
        self,
        pg_url: str,
        table: str = DEFAULT_TABLE,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        jsonl_path: str | Path = DEFAULT_JSONL,
        rrf_k: int = 60,
        candidates_per_retriever: int = 30,
        enable_reranker: bool = False,
        reranker_model: str | None = None,
        rerank_top_n: int = 30,
    ):
        self.pg_url = pg_url
        self.table = table
        self.rrf_k = rrf_k
        self.candidates_per_retriever = candidates_per_retriever
        self.jsonl_path = Path(jsonl_path)
        self.enable_reranker = enable_reranker
        self.rerank_top_n = rerank_top_n

        logger.info("Connecting Postgres (pgvector): %s", _redact(pg_url))
        # Use single long-lived connection; pgvector queries are short.
        # autocommit=True so each SELECT is independent (avoids stale tx).
        self.conn = psycopg.connect(pg_url, autocommit=True)
        register_vector(self.conn)
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table}")
            n = cur.fetchone()[0]
        logger.info("pgvector backend ready: %d rows in %s", n, self.table)

        logger.info("Loading embedding model: %s", embedding_model)
        self.model = SentenceTransformer(embedding_model)
        self._query_prefix = _e5_query_prefix(embedding_model)

        self._reranker = None
        if enable_reranker:
            from .reranker import Reranker, DEFAULT_RERANKER_MODEL
            self._reranker = Reranker(model_name=reranker_model or DEFAULT_RERANKER_MODEL)

        self._bm25_index: BM25Okapi | None = None
        self._payloads: list[dict] = []
        self._statuses: list[str] = []
        self._topics: list[str] = []
        self._doc_ids: list[str] = []
        self._dieu_to_ids: dict[tuple[str, int], list[int]] = {}
        self._build_bm25_corpus()

    # --- public API ---------------------------------------------------------

    @_ls_traceable(run_type="retriever", name="PgVectorHybridRetriever.get_relevant_chunks")
    def get_relevant_chunks(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        filters = filters or {}
        status_filter = filters.get("status", "active")
        topic_filter = filters.get("topic")
        doc_id_filter = filters.get("doc_id")
        exclude_topics = list(filters.get("exclude_topics") or [])
        exclude_doc_ids = list(filters.get("exclude_doc_ids") or [])

        auto_excl_topics, auto_excl_docs = self._auto_exclude_from_query(query)
        for t in auto_excl_topics:
            if t not in exclude_topics:
                exclude_topics.append(t)
        for d in auto_excl_docs:
            if d not in exclude_doc_ids:
                exclude_doc_ids.append(d)

        query_vector = self.model.encode(
            self._query_prefix + query, normalize_embeddings=True,
        ).tolist()

        # Dense search via pgvector. `<=>` = cosine distance (0 = identical).
        # Convert to similarity in client (1 - distance) for parity with Qdrant.
        where_sql, params = self._build_where(
            status_filter, topic_filter, doc_id_filter,
            exclude_topics, exclude_doc_ids,
        )
        sql = f"""
            SELECT id, 1.0 - (embedding <=> %s::vector) AS score
            FROM {self.table}
            {where_sql}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params_full = [query_vector, *params, query_vector, self.candidates_per_retriever]
        with self.conn.cursor() as cur:
            cur.execute(sql, params_full)
            dense_rows = cur.fetchall()  # [(id, score), ...]

        dense_hits = [_FakeHit(id=r[0], score=float(r[1]),
                               payload=self._payloads[r[0]]) for r in dense_rows]

        q_tokens = self._tokenize_vi(query)
        bm25_scores = self._bm25_index.get_scores(q_tokens)
        allowed_ids = self._filter_ids(
            status_filter, topic_filter, doc_id_filter,
            exclude_topics, exclude_doc_ids,
        )
        if allowed_ids is None:
            scored = list(enumerate(bm25_scores))
        else:
            scored = [(i, s) for i, s in enumerate(bm25_scores) if i in allowed_ids]
        bm25_hits = sorted(scored, key=lambda x: x[1], reverse=True)[
            : self.candidates_per_retriever
        ]

        fuse_k = max(top_k, self.rerank_top_n) if self.enable_reranker else top_k
        fused = self._rrf_fuse(dense_hits, bm25_hits, fuse_k)

        if self.enable_reranker and self._reranker is not None and fused:
            fused = self._reranker.rerank(query, fused, top_k=top_k)

        return self._attach_siblings(fused, max_neighbors=2)

    retrieve = get_relevant_chunks

    # get_chunks_by_location is BM25-corpus-driven (no DB call needed).
    # Identical implementation to TrafficHybridRetriever.
    def get_chunks_by_location(
        self,
        doc_id: str,
        dieu: int,
        khoan: str | int | None = None,
        diem: str | None = None,
    ) -> list[RetrievedChunk]:
        try:
            dieu_int = int(dieu)
        except (TypeError, ValueError):
            return []
        row_ids = self._dieu_to_ids.get((doc_id, dieu_int), [])
        if not row_ids:
            return []
        khoan_str = str(khoan) if khoan not in (None, "") else None
        diem_str = str(diem).lower() if diem not in (None, "") else None
        results: list[RetrievedChunk] = []
        for rid in row_ids:
            pl = self._payloads[rid]
            if khoan_str is not None:
                pk = pl.get("khoan")
                if pk is None or str(pk) != khoan_str:
                    continue
            if diem_str is not None:
                pd = pl.get("diem")
                if pd is None or str(pd).lower() != diem_str:
                    continue
            content = pl.get("content", "")
            metadata = {k: v for k, v in pl.items() if k != "content"}
            results.append(RetrievedChunk(id=rid, score=0.0, content=content, metadata=metadata))
        return results

    @staticmethod
    def _auto_exclude_from_query(query: str) -> tuple[list[str], list[str]]:
        q = query.lower()
        topics, docs = [], []
        if "không kinh doanh vận tải" in q or "không kinh doanh" in q:
            topics.append("Kinh doanh vận tải")
            docs.append("10/2020/NĐ-CP")
        return topics, docs

    # --- BM25 corpus --------------------------------------------------------

    @staticmethod
    def _tokenize_vi(text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        return [t for t in text.split() if t not in VI_STOPWORDS and len(t) > 1]

    def _build_bm25_corpus(self) -> None:
        logger.info("Building BM25 corpus from: %s", self.jsonl_path)
        t0 = time.time()
        tokenized: list[list[str]] = []
        total_tokens = 0
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                c = json.loads(line)
                tokens = self._tokenize_vi(c.get("content", ""))
                tokenized.append(tokens)
                total_tokens += len(tokens)

                meta = c.get("metadata", {})
                self._payloads.append({
                    "chunk_id":       meta.get("chunk_id", ""),
                    "doc_id":         meta.get("doc_id", ""),
                    "ten_van_ban":    meta.get("ten_van_ban", ""),
                    "issuer":         meta.get("issuer", ""),
                    "document_type":  meta.get("document_type", ""),
                    "status":         meta.get("status", "active"),
                    "effective_date": meta.get("effective_date", ""),
                    "topic":          meta.get("topic", ""),
                    "dieu":           meta.get("dieu", 0),
                    "dieu_title":     meta.get("dieu_title", ""),
                    "khoan":          meta.get("khoan"),
                    "diem":           meta.get("diem"),
                    "level":          meta.get("level", 1),
                    "source_file":    meta.get("source_file", ""),
                    "token_estimate": meta.get("token_estimate", 0),
                    "content":        c.get("content", ""),
                })
                self._statuses.append(meta.get("status", "active"))
                self._topics.append(meta.get("topic", ""))
                self._doc_ids.append(meta.get("doc_id", ""))

                did = meta.get("doc_id", "")
                dieu = meta.get("dieu", 0)
                try:
                    dieu_int = int(dieu) if dieu not in (None, "") else 0
                except (TypeError, ValueError):
                    dieu_int = 0
                if did and dieu_int:
                    self._dieu_to_ids.setdefault((did, dieu_int), []).append(
                        len(self._payloads) - 1
                    )

        self._bm25_index = BM25Okapi(tokenized)
        logger.info(
            "BM25 corpus built: %d docs, %d tokens, %.2fs",
            len(tokenized), total_tokens, time.time() - t0,
        )

    # --- WHERE clause builder ----------------------------------------------

    @staticmethod
    def _build_where(
        status: str | None,
        topic: str | None,
        doc_id: str | None,
        exclude_topics: list[str] | None,
        exclude_doc_ids: list[str] | None,
    ) -> tuple[str, list]:
        clauses, params = [], []
        if status and status != "all":
            clauses.append("status = %s")
            params.append(status)
        if topic:
            clauses.append("topic = %s")
            params.append(topic)
        if doc_id:
            clauses.append("doc_id = %s")
            params.append(doc_id)
        if exclude_topics:
            clauses.append("topic <> ALL(%s)")
            params.append(exclude_topics)
        if exclude_doc_ids:
            clauses.append("doc_id <> ALL(%s)")
            params.append(exclude_doc_ids)
        if not clauses:
            return "", []
        return "WHERE " + " AND ".join(clauses), params

    def _filter_ids(
        self,
        status_filter: str | None,
        topic_filter: str | None,
        doc_id_filter: str | None,
        exclude_topics: list[str] | None = None,
        exclude_doc_ids: list[str] | None = None,
    ) -> set[int] | None:
        status_active = bool(status_filter) and status_filter != "all"
        topic_active = bool(topic_filter)
        doc_active = bool(doc_id_filter)
        excl_topics = set(exclude_topics or [])
        excl_docs = set(exclude_doc_ids or [])
        any_active = status_active or topic_active or doc_active or excl_topics or excl_docs
        if not any_active:
            return None
        allowed: set[int] = set()
        for i, (st, tp, did) in enumerate(zip(self._statuses, self._topics, self._doc_ids)):
            if status_active and st != status_filter:
                continue
            if topic_active and tp != topic_filter:
                continue
            if doc_active and did != doc_id_filter:
                continue
            if tp in excl_topics:
                continue
            if did in excl_docs:
                continue
            allowed.add(i)
        return allowed

    # --- RRF fusion ---------------------------------------------------------

    @_ls_traceable(run_type="tool", name="rrf_fuse_pgvector")
    def _rrf_fuse(self, dense_hits, bm25_hits, top_k: int) -> list[RetrievedChunk]:
        k = self.rrf_k
        scores: dict[int, float] = {}
        payloads: dict[int, dict] = {}

        for rank, h in enumerate(dense_hits, start=1):
            scores[h.id] = scores.get(h.id, 0.0) + 1.0 / (k + rank)
            payloads[h.id] = h.payload

        for rank, (pid, _bm25_score) in enumerate(bm25_hits, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
            payloads.setdefault(pid, self._payloads[pid])

        fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        results: list[RetrievedChunk] = []
        for pid, score in fused:
            pl = payloads[pid]
            content = pl.get("content", "")
            metadata = {k: v for k, v in pl.items() if k != "content"}
            results.append(RetrievedChunk(id=pid, score=score, content=content, metadata=metadata))
        return results

    # --- sibling enrichment (copy from TrafficHybridRetriever) ------------

    _SIBLING_PRIORITY_PATTERNS = (
        "còn bị trừ điểm giấy phép",
        "bị trừ điểm giấy phép lái xe",
        "hình thức xử phạt bổ sung",
        "tước quyền sử dụng giấy phép",
        "tịch thu phương tiện",
    )

    def _sibling_priority(self, row_idx: int) -> int:
        content = (self._payloads[row_idx].get("content") or "").lower()
        for pat in self._SIBLING_PRIORITY_PATTERNS:
            if pat in content:
                return 0
        return 1

    def _attach_siblings(
        self, primaries: list[RetrievedChunk], max_neighbors: int = 2
    ) -> list[RetrievedChunk]:
        seen_ids: set[int] = {c.id for c in primaries}
        enriched: list[RetrievedChunk] = list(primaries)

        def _emit(row_idx: int) -> None:
            if row_idx in seen_ids:
                return
            pl = self._payloads[row_idx]
            content = pl.get("content", "")
            metadata = {k: v for k, v in pl.items() if k != "content"}
            metadata["is_sibling"] = True
            enriched.append(RetrievedChunk(
                id=row_idx, score=0.0, content=content, metadata=metadata,
            ))
            seen_ids.add(row_idx)

        unique_dieu: set[tuple[str, int]] = set()
        for primary in primaries:
            did = primary.metadata.get("doc_id", "")
            dieu = primary.metadata.get("dieu", 0)
            try:
                dieu_int = int(dieu) if dieu not in (None, "") else 0
            except (TypeError, ValueError):
                dieu_int = 0
            if did and dieu_int:
                unique_dieu.add((did, dieu_int))

        for did, dieu_int in unique_dieu:
            for rid in self._dieu_to_ids.get((did, dieu_int), []):
                if self._sibling_priority(rid) == 0:
                    _emit(rid)

        for primary in primaries:
            did = primary.metadata.get("doc_id", "")
            dieu = primary.metadata.get("dieu", 0)
            khoan = primary.metadata.get("khoan")
            try:
                dieu_int = int(dieu) if dieu not in (None, "") else 0
                khoan_int = int(khoan) if khoan not in (None, "") else 0
            except (TypeError, ValueError):
                continue
            if not did or not dieu_int or not khoan_int:
                continue
            for rid in self._dieu_to_ids.get((did, dieu_int), []):
                k = self._payloads[rid].get("khoan")
                try:
                    k_int = int(k) if k not in (None, "") else 0
                except (TypeError, ValueError):
                    continue
                if k_int and 0 < abs(k_int - khoan_int) <= max_neighbors:
                    _emit(rid)

        return enriched


class _FakeHit:
    """Mimics qdrant_client.http.models.ScoredPoint just enough for RRF fusion."""
    __slots__ = ("id", "score", "payload")

    def __init__(self, id: int, score: float, payload: dict):
        self.id = id
        self.score = score
        self.payload = payload


def _redact(url: str) -> str:
    # postgresql://user:PASS@host:port/db → postgresql://user:***@host:port/db
    return re.sub(r"://([^:]+):[^@]+@", r"://\1:***@", url)
