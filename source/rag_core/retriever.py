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

logger = logging.getLogger(__name__)


VI_STOPWORDS = {
    "là", "và", "của", "cho", "các", "có", "được", "trong", "theo",
    "với", "một", "khi", "hoặc", "từ", "này", "đó", "tại", "do",
    "để", "sẽ", "đã", "nếu", "bị", "bởi",
}

DEFAULT_COLLECTION = "traffic_law_v2"
DEFAULT_EMBEDDING_MODEL = "KeepItReal/vietnamese-sbert"
DEFAULT_JSONL = (
    Path(__file__).resolve().parent.parent.parent / "Data" / "all_chunks.jsonl"
)


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
    Hybrid retriever combining Qdrant dense vectors (vietnamese-sbert) with
    an in-process BM25 lexical index. Scores fused via Reciprocal Rank Fusion.

    BM25 row index is aligned to Qdrant point id (JSONL insertion order), which
    is how the indexer upserted points — so no id-mapping layer is needed.
    """

    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6334,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        jsonl_path: str | Path = DEFAULT_JSONL,
        rrf_k: int = 60,
        candidates_per_retriever: int = 30,
    ):
        self.collection_name = collection_name
        self.rrf_k = rrf_k
        self.candidates_per_retriever = candidates_per_retriever
        self.jsonl_path = Path(jsonl_path)

        logger.info(f"Connecting Qdrant: {qdrant_host}:{qdrant_port}")
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.client.get_collections()  # fail fast if unreachable

        logger.info(f"Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)

        self._bm25_index: BM25Okapi | None = None
        self._payloads: list[dict] = []
        self._statuses: list[str] = []
        self._topics: list[str] = []
        self._doc_ids: list[str] = []
        self._build_bm25_corpus()

    # --- public API ---------------------------------------------------------

    def get_relevant_chunks(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.

        filters: optional dict accepting any of:
          - "status": str (e.g. "active", "all" to disable)
          - "topic":  str
          - "doc_id": str
        """
        filters = filters or {}
        status_filter = filters.get("status", "active")
        topic_filter = filters.get("topic")
        doc_id_filter = filters.get("doc_id")

        qdrant_filter = self._build_qdrant_filter(
            status_filter, topic_filter, doc_id_filter
        )

        query_vector = self.model.encode(query, normalize_embeddings=True).tolist()
        dense_hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=self.candidates_per_retriever,
            with_payload=True,
        )

        q_tokens = self._tokenize_vi(query)
        bm25_scores = self._bm25_index.get_scores(q_tokens)
        allowed_ids = self._filter_ids(status_filter, topic_filter, doc_id_filter)

        if allowed_ids is None:
            scored = list(enumerate(bm25_scores))
        else:
            scored = [(i, s) for i, s in enumerate(bm25_scores) if i in allowed_ids]

        bm25_hits = sorted(scored, key=lambda x: x[1], reverse=True)[
            : self.candidates_per_retriever
        ]

        return self._rrf_fuse(dense_hits, bm25_hits, top_k)

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
    ) -> Filter | None:
        conditions = []
        if status_filter and status_filter != "all":
            conditions.append(
                FieldCondition(key="status", match=MatchValue(value=status_filter))
            )
        if topic_filter:
            conditions.append(
                FieldCondition(key="topic", match=MatchValue(value=topic_filter))
            )
        if doc_id_filter:
            conditions.append(
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id_filter))
            )
        return Filter(must=conditions) if conditions else None

    def _filter_ids(
        self,
        status_filter: str | None,
        topic_filter: str | None,
        doc_id_filter: str | None,
    ) -> set[int] | None:
        status_active = bool(status_filter) and status_filter != "all"
        topic_active = bool(topic_filter)
        doc_active = bool(doc_id_filter)
        if not (status_active or topic_active or doc_active):
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
            allowed.add(i)
        return allowed

    # --- RRF fusion ---------------------------------------------------------

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
