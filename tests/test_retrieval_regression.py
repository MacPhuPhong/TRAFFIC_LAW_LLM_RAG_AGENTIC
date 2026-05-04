# -*- coding: utf-8 -*-
"""Retrieval regression — auto-skipped if Qdrant is unreachable.

Why this exists: the hybrid retriever's behaviour depends on chunker rules,
embedding model, BM25 tokenizer, RRF k, and indexer alignment. A change to
any of those can silently regress quality. This test runs a tiny, fast
sample of the gold set and asserts a quality floor.

Threshold rationale:
- RQ3 (analysis.md) reports e5-small Recall@10 = 0.46 on full 25-question
  set, hierarchical chunker. We pick 5 questions where the baseline is
  expected to hit the gold khoản, and require Recall@10 >= 0.40 — well
  below the empirical mean so noise alone shouldn't trip it.

Skip if:
- Qdrant is not reachable (CI without docker-compose)
- The Qdrant collection is missing or empty
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.utils.metrics import recall_at_k

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "research" / "data" / "eval_qa.jsonl"

REGRESSION_RECALL_FLOOR = 0.40
SUBSET_IDS = ["RQ-001", "RQ-002", "RQ-003", "RQ-004", "RQ-005"]


pytestmark = pytest.mark.requires_qdrant


@pytest.fixture(scope="module")
def retriever():
    from source.rag_core import TrafficHybridRetriever
    try:
        return TrafficHybridRetriever(enable_reranker=False)
    except Exception as exc:
        pytest.skip(f"retriever init failed: {exc}")


@pytest.fixture(scope="module")
def gold_subset() -> list[dict]:
    rows = [json.loads(l) for l in open(EVAL, encoding="utf-8") if l.strip()]
    return [r for r in rows if r["id"] in SUBSET_IDS]


def _chunks_to_dicts(chunks) -> list[dict]:
    out = []
    for c in chunks:
        if c.metadata.get("is_sibling"):
            continue
        out.append({
            "doc_id": c.metadata.get("doc_id"),
            "dieu":   c.metadata.get("dieu"),
            "khoan":  c.metadata.get("khoan"),
            "diem":   c.metadata.get("diem"),
        })
    return out


def test_retrieval_recall_floor(retriever, gold_subset):
    """Mean Recall@10 across the SUBSET_IDS must stay above REGRESSION_RECALL_FLOOR."""
    assert gold_subset, "subset must be non-empty"
    recalls = []
    for q in gold_subset:
        chunks = retriever.get_relevant_chunks(q["question"], top_k=10)
        retrieved = _chunks_to_dicts(chunks)
        recalls.append(recall_at_k(retrieved, q["gold_citations"], k=10, level="khoan"))

    mean_recall = sum(recalls) / len(recalls)
    assert mean_recall >= REGRESSION_RECALL_FLOOR, (
        f"Recall@10 floor breached: {mean_recall:.3f} < {REGRESSION_RECALL_FLOOR} "
        f"(per-q: {recalls})"
    )
