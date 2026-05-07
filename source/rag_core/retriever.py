# -*- coding: utf-8 -*-
"""
retriever.py — Phase 3: Hybrid Retriever (Dense + BM25, RRF fusion)
===================================================================
OOP wrapper around the Phase 2 search logic. Canonical retrieval entry
point for downstream generation and evaluation.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

try:
    from langsmith import traceable as _ls_traceable
except ImportError:  # langsmith not installed → no-op decorator
    def _ls_traceable(*dargs, **dkwargs):
        if dargs and callable(dargs[0]) and not dkwargs:
            return dargs[0]
        def _wrap(fn):
            return fn
        return _wrap

logger = logging.getLogger(__name__)


VI_STOPWORDS = {
    "là", "và", "của", "cho", "các", "có", "được", "trong", "theo",
    "với", "một", "khi", "hoặc", "từ", "này", "đó", "tại", "do",
    "để", "sẽ", "đã", "nếu", "bị", "bởi",
}

DEFAULT_COLLECTION = "Traffic_Law_Hybrid"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_JSONL = (
    Path(__file__).resolve().parent.parent.parent / "Data" / "all_chunks.jsonl"
)


def _e5_query_prefix(model_name: str) -> str:
    # e5 family requires the "query: " prefix at encode time; other models don't.
    return "query: " if "e5" in model_name.lower() else ""


@dataclass
class RetrievedChunk:
    id: int
    score: float
    content: str
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "score": self.score,
            "content": self.content,
            "metadata": self.metadata,
        }


class TrafficHybridRetriever:
    """
    Hybrid retriever combining Qdrant dense vectors (multilingual-e5-small, 384d)
    with an in-process BM25 lexical index. Scores fused via Reciprocal Rank Fusion.

    BM25 row index is aligned to Qdrant point id (JSONL insertion order), which
    is how the indexer upserted points — so no id-mapping layer is needed.
    """

    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        jsonl_path: str | Path = DEFAULT_JSONL,
        rrf_k: int = 60,
        candidates_per_retriever: int = 30,
        enable_reranker: bool = False,
        reranker_model: str | None = None,
        rerank_top_n: int = 30,
    ):
        self.collection_name = collection_name
        self.rrf_k = rrf_k
        self.candidates_per_retriever = candidates_per_retriever
        self.jsonl_path = Path(jsonl_path)
        self.enable_reranker = enable_reranker
        self.rerank_top_n = rerank_top_n

        logger.info(f"Connecting Qdrant: {qdrant_host}:{qdrant_port}")
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.client.get_collections()  # fail fast if unreachable

        logger.info(f"Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        self._query_prefix = _e5_query_prefix(embedding_model)

        self._reranker = None
        if enable_reranker:
            from .reranker import Reranker, DEFAULT_RERANKER_MODEL
            self._reranker = Reranker(
                model_name=reranker_model or DEFAULT_RERANKER_MODEL,
            )

        self._bm25_index: BM25Okapi | None = None
        self._payloads: list[dict] = []
        self._statuses: list[str] = []
        self._topics: list[str] = []
        self._doc_ids: list[str] = []
        self._dieu_to_ids: dict[tuple[str, int], list[int]] = {}
        self._build_bm25_corpus()

    # --- public API ---------------------------------------------------------

    @_ls_traceable(run_type="retriever", name="TrafficHybridRetriever.get_relevant_chunks")
    def get_relevant_chunks(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.

        filters: optional dict accepting any of:
          - "status":         str (e.g. "active", "all" to disable)
          - "topic":          str
          - "doc_id":         str
          - "exclude_topics": list[str]
          - "exclude_doc_ids": list[str]

        Auto-rewrite: phrases like "không kinh doanh vận tải" in the query
        automatically add `Kinh doanh vận tải` to `exclude_topics` so that
        Nghị định 10/2020/NĐ-CP (business-transport-only scope) does not
        dominate retrieval when the user explicitly asked about non-business
        vehicles.
        """
        filters = filters or {}
        status_filter = filters.get("status", "active")
        topic_filter = filters.get("topic")
        doc_id_filter = filters.get("doc_id")
        exclude_topics = list(filters.get("exclude_topics") or [])
        exclude_doc_ids = list(filters.get("exclude_doc_ids") or [])

        auto_excluded_topics, auto_excluded_docs = self._auto_exclude_from_query(query)
        for t in auto_excluded_topics:
            if t not in exclude_topics:
                exclude_topics.append(t)
        for d in auto_excluded_docs:
            if d not in exclude_doc_ids:
                exclude_doc_ids.append(d)

        qdrant_filter = self._build_qdrant_filter(
            status_filter, topic_filter, doc_id_filter,
            exclude_topics, exclude_doc_ids,
        )

        query_vector = self.model.encode(
            self._query_prefix + query, normalize_embeddings=True,
        ).tolist()
        dense_hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=self.candidates_per_retriever,
            with_payload=True,
        )

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

    def get_chunks_by_location(
        self,
        doc_id: str,
        dieu: int,
        khoan: str | int | None = None,
        diem: str | None = None,
    ) -> list[RetrievedChunk]:
        """Direct metadata lookup — no similarity search.

        Returns every chunk whose (doc_id, dieu [, khoan [, diem]]) matches.
        Used when the query contains an explicit cross-reference like
        "điểm h khoản 14 Điều 32" — similarity search often dilutes these
        pin-point references, so we resolve them structurally instead.
        """
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
            results.append(
                RetrievedChunk(id=rid, score=0.0, content=content, metadata=metadata)
            )
        return results

    @staticmethod
    def _auto_exclude_from_query(query: str) -> tuple[list[str], list[str]]:
        """
        Lightweight query-side rewriter: detect exclusion intent and return
        (topics_to_exclude, doc_ids_to_exclude). Extend as new patterns emerge.
        """
        q = query.lower()
        topics: list[str] = []
        docs: list[str] = []
        if "không kinh doanh vận tải" in q or "không kinh doanh" in q:
            # NĐ 10/2020 chỉ áp dụng cho xe kinh doanh vận tải.
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
        logger.info(f"Building BM25 corpus from: {self.jsonl_path}")
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
            f"BM25 corpus built: {len(tokenized)} docs, {total_tokens} tokens, "
            f"{time.time() - t0:.2f}s"
        )

    # --- filtering ----------------------------------------------------------

    @staticmethod
    def _build_qdrant_filter(
        status_filter: str | None,
        topic_filter: str | None,
        doc_id_filter: str | None,
        exclude_topics: list[str] | None = None,
        exclude_doc_ids: list[str] | None = None,
    ) -> Filter | None:
        must = []
        must_not = []
        if status_filter and status_filter != "all":
            must.append(
                FieldCondition(key="status", match=MatchValue(value=status_filter))
            )
        if topic_filter:
            must.append(
                FieldCondition(key="topic", match=MatchValue(value=topic_filter))
            )
        if doc_id_filter:
            must.append(
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id_filter))
            )
        for t in exclude_topics or []:
            must_not.append(
                FieldCondition(key="topic", match=MatchValue(value=t))
            )
        for d in exclude_doc_ids or []:
            must_not.append(
                FieldCondition(key="doc_id", match=MatchValue(value=d))
            )
        if not must and not must_not:
            return None
        return Filter(must=must or None, must_not=must_not or None)

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
        for i, (st, tp, did) in enumerate(
            zip(self._statuses, self._topics, self._doc_ids)
        ):
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

    @_ls_traceable(run_type="tool", name="rrf_fuse")
    def _rrf_fuse(
        self, dense_hits, bm25_hits, top_k: int
    ) -> list[RetrievedChunk]:
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
            results.append(
                RetrievedChunk(id=pid, score=score, content=content, metadata=metadata)
            )
        return results

    # --- sibling enrichment ------------------------------------------------

    # Specific phrases that only appear in "completion-clause" chunks —
    # trừ-điểm-GPLX table or xử-phạt-bổ-sung clause. Must avoid patterns
    # that also occur in the Điều title (every chunk inherits the title as
    # a prefix, so e.g. "trừ điểm giấy phép" would match every Điều 6/7
    # chunk and defeat the priority ranking).
    _SIBLING_PRIORITY_PATTERNS = (
        "còn bị trừ điểm giấy phép",          # opening of the trừ-điểm table
        "bị trừ điểm giấy phép lái xe",       # in-table row: "bị trừ ... N điểm"
        "hình thức xử phạt bổ sung",          # bổ sung clause header
        "tước quyền sử dụng giấy phép",       # bổ sung detail
        "tịch thu phương tiện",               # bổ sung detail
    )

    def _sibling_priority(self, row_idx: int) -> int:
        """Lower number = higher priority sibling. 0 for completion clauses."""
        content = (self._payloads[row_idx].get("content") or "").lower()
        for pat in self._SIBLING_PRIORITY_PATTERNS:
            if pat in content:
                return 0
        return 1

    def _attach_siblings(
        self, primaries: list[RetrievedChunk], max_neighbors: int = 2
    ) -> list[RetrievedChunk]:
        """
        Two-tier sibling enrichment:
          (a) Always attach ALL priority-0 "completion clause" chunks for each
              unique (doc_id, dieu) present in primaries. These are the
              trừ-điểm-GPLX table and xử-phạt-bổ-sung clauses that complete
              a penalty answer. Typically 3 chunks per Điều in NĐ 168.
          (b) For each primary whose khoan is an integer, attach the
              immediately adjacent Khoản (K-1 and K+1) when available. This
              handles range-boundary cases — e.g. query ">0,4 mg" falls into
              K9 while retriever pulled K8 (which contains "0,25-0,4 mg");
              K9 as neighbor fills the gap.

        Siblings keep score=0.0 and are flagged metadata["is_sibling"]=True.
        """
        seen_ids: set[int] = {c.id for c in primaries}
        enriched: list[RetrievedChunk] = list(primaries)

        def _emit(row_idx: int) -> None:
            if row_idx in seen_ids:
                return
            pl = self._payloads[row_idx]
            content = pl.get("content", "")
            metadata = {k: v for k, v in pl.items() if k != "content"}
            metadata["is_sibling"] = True
            enriched.append(
                RetrievedChunk(
                    id=row_idx, score=0.0, content=content, metadata=metadata
                )
            )
            seen_ids.add(row_idx)

        # Tier (a): attach all completion-clause chunks for each unique Điều.
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

        # Tier (b): for each primary Khoản, attach adjacent Khoản (±N).
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
