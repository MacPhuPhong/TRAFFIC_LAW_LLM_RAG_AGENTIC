# -*- coding: utf-8 -*-
"""RQ6 — Retrieval-only metrics over the 25-question gold set.

Runs the production hybrid retriever (Qdrant dense e5-base + BM25 + RRF) and
scores Recall@{1,3,5,10,20}, MRR, nDCG@10 at the `khoan` granularity. Writes
per-question + summary CSVs.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from research.utils.metrics import mrr, ndcg_at_k, recall_at_k  # noqa: E402
from source.rag_core import TrafficHybridRetriever  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rq6")

GOLD = ROOT / "research" / "data" / "eval_qa.jsonl"
OUT_DIR = ROOT / "research" / "results" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    gold = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]
    log.info("Loaded %d gold rows", len(gold))

    retriever = TrafficHybridRetriever()

    rows = []
    for row in gold:
        gc = row.get("gold_citations") or []
        if not gc:
            continue  # skip out-of-scope (no gold citation)
        t0 = time.perf_counter()
        chunks = retriever.get_relevant_chunks(row["question"], top_k=20)
        dt = time.perf_counter() - t0
        cites = [{
            "doc_id": c.metadata.get("doc_id"),
            "dieu": c.metadata.get("dieu"),
            "khoan": c.metadata.get("khoan"),
            "diem": c.metadata.get("diem"),
        } for c in chunks]
        res = {
            "qid": row["id"],
            "category": row.get("category"),
            "latency_s": dt,
        }
        for k in (1, 3, 5, 10, 20):
            res[f"recall@{k}"] = recall_at_k(cites, gc, k, "khoan")
        res["mrr"] = mrr(cites, gc, "khoan")
        res["ndcg@10"] = ndcg_at_k(cites, gc, 10, "khoan")
        rows.append(res)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "rq6_retrieval_per_question.csv", index=False)

    summary = df.drop(columns=["qid", "category"]).mean().round(4)
    summary.to_csv(OUT_DIR / "rq6_retrieval_summary.csv", header=["mean"])
    log.info("\n%s", summary.to_string())
    log.info("Wrote rq6_retrieval_*.csv to %s", OUT_DIR)


if __name__ == "__main__":
    main()
