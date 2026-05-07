# -*- coding: utf-8 -*-
"""RQ2 — Chunking ablation: hierarchical (Điều/Khoản/Điểm) vs fixed-size 512.

Runs two passes over the 25-question gold set:
  1. Retrieval-only: Recall@{5,10,20}, MRR, nDCG@10 at khoan + doc_id granularity.
  2. End-to-end vanilla RAG: F1, ROUGE-L, Cit-P/R, refusal rate, latency.

Writes CSVs to research/results/metrics/ and figures to research/results/figures/.

Encoder choice: we use `intfloat/multilingual-e5-small` (max_seq=512) for the
fixed-512 collection so chunks are NOT truncated at embed time. The hierarchical
collection uses its production encoder (`KeepItReal/vietnamese-sbert`, max_seq=256)
— which is fine because hierarchical chunks are typically <256 tokens.
This is a known confound (encoder + chunker change together) and is flagged in
analysis.md §3.2.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # .../traffic_rag
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env so GOOGLE_API_KEY / API_KEY / LANGCHAIN_API_KEY are available.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT.parent / ".env")
except ImportError:
    pass
if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["API_KEY"]

import pandas as pd  # noqa: E402

from research.utils.metrics import (  # noqa: E402
    citation_precision_recall,
    mrr,
    ndcg_at_k,
    recall_at_k,
    rouge_l,
    token_f1,
)
from source.rag_core import LegalAnswerGenerator, TrafficHybridRetriever  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("rq2")

GOLD = ROOT / "research" / "data" / "eval_qa.jsonl"
METRICS_DIR = ROOT / "research" / "results" / "metrics"
FIG_DIR = ROOT / "research" / "results" / "figures"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

REFUSAL_PAT = ["không đủ thông tin", "không tìm thấy", "không có trong", "tôi không biết"]


def is_refusal(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in REFUSAL_PAT)


def load_gold() -> list[dict]:
    return [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]


def _e5_query_wrap(retriever: TrafficHybridRetriever):
    """Prefix 'query: ' for e5 models (required by model card)."""
    orig = retriever.model.encode
    def _enc(text, **kw):
        if isinstance(text, str):
            return orig("query: " + text, **kw)
        return orig([f"query: {t}" for t in text], **kw)
    retriever.model.encode = _enc


def chunk_to_citation(chunk) -> dict:
    m = chunk.metadata if hasattr(chunk, "metadata") else chunk.get("metadata", {})
    return {
        "doc_id": m.get("doc_id"),
        "dieu": m.get("dieu"),
        "khoan": m.get("khoan"),
        "diem": m.get("diem"),
    }


def run_retrieval(name: str, retriever: TrafficHybridRetriever, gold: list[dict]) -> list[dict]:
    rows = []
    for item in gold:
        t0 = time.perf_counter()
        chunks = retriever.get_relevant_chunks(item["question"], top_k=20)
        dt = time.perf_counter() - t0
        retrieved_cites = [chunk_to_citation(c) for c in chunks]
        g = item["gold_citations"]
        rows.append({
            "chunker": name,
            "qid": item["id"],
            "category": item["category"],
            "latency_s": dt,
            "recall@5_khoan": recall_at_k(retrieved_cites, g, 5, "khoan"),
            "recall@10_khoan": recall_at_k(retrieved_cites, g, 10, "khoan"),
            "recall@20_khoan": recall_at_k(retrieved_cites, g, 20, "khoan"),
            "mrr_khoan": mrr(retrieved_cites, g, "khoan"),
            "ndcg@10_khoan": ndcg_at_k(retrieved_cites, g, 10, "khoan"),
            "recall@5_doc": recall_at_k(retrieved_cites, g, 5, "doc_id"),
            "recall@10_doc": recall_at_k(retrieved_cites, g, 10, "doc_id"),
            "recall@20_doc": recall_at_k(retrieved_cites, g, 20, "doc_id"),
            "mrr_doc": mrr(retrieved_cites, g, "doc_id"),
            "ndcg@10_doc": ndcg_at_k(retrieved_cites, g, 10, "doc_id"),
        })
    return rows


def run_vanilla(name: str, retriever, generator, gold: list[dict]) -> list[dict]:
    rows = []
    for item in gold:
        t0 = time.perf_counter()
        chunks = retriever.get_relevant_chunks(item["question"], top_k=10)
        chunk_dicts = [c.to_dict() for c in chunks]
        try:
            out = generator.generate(item["question"], chunk_dicts)
        except Exception as e:
            log.error("  [%s] gen error on %s: %s", name, item["id"], e)
            out = {"answer": "", "sources": []}
        dt = time.perf_counter() - t0
        pred_ans = out.get("answer", "")
        pred_cites = out.get("sources") or []
        g = item["gold_citations"]
        p_k, r_k, f_k = citation_precision_recall(pred_cites, g, "khoan")
        p_d, r_d, f_d = citation_precision_recall(pred_cites, g, "doc_id")
        rows.append({
            "chunker": name,
            "qid": item["id"],
            "category": item["category"],
            "latency_s": dt,
            "f1_token": token_f1(pred_ans, item["gold_answer"]),
            "rouge_l": rouge_l(pred_ans, item["gold_answer"]),
            "cit_p_khoan": p_k, "cit_r_khoan": r_k, "cit_f1_khoan": f_k,
            "cit_p_doc": p_d, "cit_r_doc": r_d, "cit_f1_doc": f_d,
            "refusal": is_refusal(pred_ans),
            "pred_answer": pred_ans[:500],
        })
        log.info("  [%s] %s  F1=%.3f  Cit-R(khoan)=%.2f  latency=%.1fs",
                 name, item["id"], rows[-1]["f1_token"], r_k, dt)
    return rows


def main():
    gold = load_gold()
    log.info("Loaded %d gold questions", len(gold))

    log.info("=== HIERARCHICAL (sbert, traffic_law_v2) ===")
    r_hier = TrafficHybridRetriever()  # defaults
    ret_hier = run_retrieval("hierarchical", r_hier, gold)

    log.info("=== FIXED-512 (e5-small, traffic_law_fixed512) ===")
    r_fixed = TrafficHybridRetriever(
        collection_name="traffic_law_fixed512",
        embedding_model="intfloat/multilingual-e5-small",
        jsonl_path=str(ROOT / "research" / "rq2_chunking" / "data" / "all_chunks_fixed512.jsonl"),
    )
    _e5_query_wrap(r_fixed)
    ret_fixed = run_retrieval("fixed_512", r_fixed, gold)

    ret_df = pd.DataFrame(ret_hier + ret_fixed)
    ret_df.to_csv(METRICS_DIR / "rq2_retrieval_per_question.csv", index=False)

    ret_sum = ret_df.groupby("chunker").agg({
        "recall@5_khoan": "mean", "recall@10_khoan": "mean", "recall@20_khoan": "mean",
        "mrr_khoan": "mean", "ndcg@10_khoan": "mean",
        "recall@5_doc": "mean", "recall@10_doc": "mean", "recall@20_doc": "mean",
        "mrr_doc": "mean", "ndcg@10_doc": "mean",
        "latency_s": "mean",
    }).round(4)
    ret_sum.to_csv(METRICS_DIR / "rq2_retrieval_summary.csv")
    log.info("\n%s", ret_sum.to_string())

    log.info("=== VANILLA RAG (hierarchical) ===")
    gen = LegalAnswerGenerator(provider="google", model="gemini-3.1-flash-lite-preview")
    ans_hier = run_vanilla("hierarchical", r_hier, gen, gold)

    log.info("=== VANILLA RAG (fixed-512) ===")
    ans_fixed = run_vanilla("fixed_512", r_fixed, gen, gold)

    ans_df = pd.DataFrame(ans_hier + ans_fixed)
    ans_df.to_csv(METRICS_DIR / "rq2_answer_per_question.csv", index=False)

    ans_sum = ans_df.groupby("chunker").agg({
        "f1_token": "mean", "rouge_l": "mean",
        "cit_p_khoan": "mean", "cit_r_khoan": "mean", "cit_f1_khoan": "mean",
        "cit_p_doc": "mean", "cit_r_doc": "mean", "cit_f1_doc": "mean",
        "refusal": "mean", "latency_s": "mean",
    }).round(4)
    ans_sum.to_csv(METRICS_DIR / "rq2_answer_summary.csv")
    log.info("\n%s", ans_sum.to_string())

    log.info("DONE. CSVs in %s", METRICS_DIR)


if __name__ == "__main__":
    main()
