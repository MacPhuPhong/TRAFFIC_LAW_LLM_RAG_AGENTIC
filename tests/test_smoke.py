# -*- coding: utf-8 -*-
"""Smoke tests — guaranteed to run in CI (no external services needed).

Coverage:
- Public API surface of `source.rag_core` is importable.
- Reranker module loads without instantiating the heavy cross-encoder.
- Eval set + corpus files exist and are well-formed.
- pytest.ini and CI workflow are present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent  # traffic_rag/


def test_rag_core_imports():
    """Public entry points are importable."""
    from source.rag_core import (  # noqa: F401
        DEFAULT_RERANKER_MODEL, LegalAnswerGenerator,
        Reranker, TrafficHybridRetriever,
    )
    assert DEFAULT_RERANKER_MODEL == "BAAI/bge-reranker-v2-m3"


def test_reranker_lazy_load():
    """Constructing a Reranker must NOT load the heavy CrossEncoder."""
    from source.rag_core import Reranker
    r = Reranker()
    assert r._model is None, "model must not be loaded until first .rerank() call"


def test_eval_set_present():
    eval_path = ROOT / "research" / "data" / "eval_qa.jsonl"
    assert eval_path.exists(), f"missing eval set: {eval_path}"
    rows = [json.loads(l) for l in open(eval_path, encoding="utf-8") if l.strip()]
    assert len(rows) >= 25, "eval_qa.jsonl must have at least 25 gold questions"
    required = {"id", "question", "gold_answer", "gold_citations"}
    for r in rows:
        missing = required - set(r.keys())
        assert not missing, f"row {r.get('id')} missing fields: {missing}"


def test_corpus_present():
    corpus = ROOT / "Data" / "all_chunks.jsonl"
    assert corpus.exists(), "Data/all_chunks.jsonl must be built before tests"
    # Just check we can read the first few lines without error.
    with open(corpus, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            payload = json.loads(line)
            assert "content" in payload and "metadata" in payload


def test_pytest_ini_present():
    assert (ROOT / "pytest.ini").exists(), "pytest.ini must be at repo root"


def test_ci_workflow_present():
    wf = ROOT / ".github" / "workflows" / "ci.yml"
    assert wf.exists(), "GitHub Actions workflow ci.yml must exist"
