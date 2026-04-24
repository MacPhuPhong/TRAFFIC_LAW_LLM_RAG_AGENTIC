# -*- coding: utf-8 -*-
"""RQ2 continuation: run vanilla RAG only (retrieval CSV already saved)."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT.parent / ".env")
except ImportError:
    pass
if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["API_KEY"]

import pandas as pd  # noqa: E402

from research.utils.metrics import (  # noqa: E402
    citation_precision_recall, rouge_l, token_f1,
)
from source.rag_core import LegalAnswerGenerator, TrafficHybridRetriever  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("rq2_vanilla")

GOLD = ROOT / "research" / "data" / "eval_qa.jsonl"
METRICS_DIR = ROOT / "research" / "results" / "metrics"

REFUSAL_PAT = ["không đủ thông tin", "không tìm thấy", "không có trong",
               "tôi không biết"]
def is_refusal(t): return any(p in (t or "").lower() for p in REFUSAL_PAT)


def _e5_query_wrap(retriever):
    orig = retriever.model.encode
    def _enc(text, **kw):
        if isinstance(text, str):
            return orig("query: " + text, **kw)
        return orig([f"query: {t}" for t in text], **kw)
    retriever.model.encode = _enc


def run_vanilla(name, retriever, generator, gold, sleep_s=1.2):
    rows = []
    for item in gold:
        t0 = time.perf_counter()
        try:
            chunks = retriever.get_relevant_chunks(item["question"], top_k=10)
            chunk_dicts = [c.to_dict() for c in chunks]
            out = generator.generate(item["question"], chunk_dicts)
        except Exception as e:
            log.error("  [%s] %s gen error: %s", name, item["id"], e)
            out = {"answer": "", "sources": []}
        dt = time.perf_counter() - t0
        pred = out.get("answer", "")
        cites = out.get("sources") or []
        g = item["gold_citations"]
        p_k, r_k, f_k = citation_precision_recall(cites, g, "khoan")
        p_d, r_d, f_d = citation_precision_recall(cites, g, "doc_id")
        rows.append({
            "chunker": name,
            "qid": item["id"],
            "category": item["category"],
            "latency_s": dt,
            "f1_token": token_f1(pred, item["gold_answer"]),
            "rouge_l": rouge_l(pred, item["gold_answer"]),
            "cit_p_khoan": p_k, "cit_r_khoan": r_k, "cit_f1_khoan": f_k,
            "cit_p_doc": p_d, "cit_r_doc": r_d, "cit_f1_doc": f_d,
            "refusal": is_refusal(pred),
            "pred_answer": pred[:500],
        })
        log.info("  [%s] %s F1=%.3f Cit-R(khoan)=%.2f (%.1fs)",
                 name, item["id"], rows[-1]["f1_token"], r_k, dt)
        time.sleep(sleep_s)
    return rows


def main():
    gold = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]
    log.info("Loaded %d gold", len(gold))

    gen = LegalAnswerGenerator(provider="google", model="gemini-3.1-flash-lite-preview")

    log.info("=== HIER ===")
    r_hier = TrafficHybridRetriever()
    ans_hier = run_vanilla("hierarchical", r_hier, gen, gold)

    log.info("=== FIXED-512 ===")
    r_fx = TrafficHybridRetriever(
        collection_name="traffic_law_fixed512",
        embedding_model="intfloat/multilingual-e5-small",
        jsonl_path=str(ROOT / "Data" / "all_chunks_fixed512.jsonl"),
    )
    _e5_query_wrap(r_fx)
    ans_fx = run_vanilla("fixed_512", r_fx, gen, gold)

    df = pd.DataFrame(ans_hier + ans_fx)
    df.to_csv(METRICS_DIR / "rq2_answer_per_question.csv", index=False)
    summary = df.groupby("chunker").agg({
        "f1_token": "mean", "rouge_l": "mean",
        "cit_p_khoan": "mean", "cit_r_khoan": "mean", "cit_f1_khoan": "mean",
        "cit_p_doc": "mean", "cit_r_doc": "mean", "cit_f1_doc": "mean",
        "refusal": "mean", "latency_s": "mean",
    }).round(4)
    summary.to_csv(METRICS_DIR / "rq2_answer_summary.csv")
    log.info("\n%s", summary.to_string())
    log.info("DONE.")


if __name__ == "__main__":
    main()
