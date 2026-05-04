# -*- coding: utf-8 -*-
"""
reranker.py — Cross-encoder reranker (Phase 3.5)
================================================
Second-stage rescoring of RRF-fused candidates with a cross-encoder.

Default model: BAAI/bge-reranker-v2-m3 (multilingual, ~568M params).
On CPU it scores ~50–80 ms per (query, chunk) pair, so the recommended
candidate budget is ~20–30 candidates per query (≈1.5 s rerank latency).

Public surface:
    Reranker(model_name=..., device=..., max_length=...)
    .rerank(query, chunks, top_k) -> list[RetrievedChunk]

The reranker preserves chunk identity and metadata; only `.score` is
overwritten with the cross-encoder relevance score.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sentence_transformers import CrossEncoder

if TYPE_CHECKING:
    from .retriever import RetrievedChunk

logger = logging.getLogger(__name__)


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class Reranker:
    """Cross-encoder reranker on top of hybrid retrieval.

    Loaded lazily on first .rerank() call so that import-time cost
    stays at zero when the reranker is disabled.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str | None = None,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._model: CrossEncoder | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        logger.info(
            "Loading cross-encoder reranker: %s (device=%s)",
            self.model_name, self.device or "auto",
        )
        self._model = CrossEncoder(
            self.model_name,
            max_length=self.max_length,
            device=self.device,
        )

    def rerank(
        self,
        query: str,
        chunks: list["RetrievedChunk"],
        top_k: int,
    ) -> list["RetrievedChunk"]:
        if not chunks:
            return []
        if top_k >= len(chunks):
            top_k = len(chunks)

        self._ensure_loaded()
        pairs = [(query, c.content) for c in chunks]
        scores = self._model.predict(pairs, show_progress_bar=False)

        rescored = list(zip(chunks, scores))
        rescored.sort(key=lambda kv: float(kv[1]), reverse=True)

        out: list["RetrievedChunk"] = []
        for chunk, score in rescored[:top_k]:
            chunk.score = float(score)
            out.append(chunk)
        return out
