# -*- coding: utf-8 -*-
"""RQ10 — Reranker ablation: hybrid (RRF) vs hybrid + bge-reranker-v2-m3.

For each of the 25 gold questions, run the SAME hybrid retrieval pipeline
with `enable_reranker` toggled on/off and score Recall@10, MRR, nDCG@10
at the khoan granularity. The reranker rescores the top-N RRF candidates
(default N=30) with a cross-encoder, then keeps top-K.

Outputs:
  - research/results/metrics/rq10_reranker.csv      — per-question metrics
  - research/results/metrics/rq10_reranker_summary.csv — aggregate by mode

Run:
  python research/scripts/rq10_reranker_ablation.py [--top-k 10] [--top-n 30]
"""

from __future__ import annotations

import argparse
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("rq10")

GOLD = ROOT / "research" / "data" / "eval_qa.jsonl"
METRICS_DIR = ROOT / "research" / "results" / "metrics"


def load_gold() -> list[dict]:
    return [json.loads(line) for line in open(GOLD, encoding="utf-8") if line.strip()]


def chunks_to_citation_dicts(chunks) -> list[dict]:
    """Convert RetrievedChunk list → list[dict] for metrics.recall_at_k.

    Drops siblings — they were attached for context but are not the
    primary retrieval result we want to score.
    """
    out: list[dict] = []
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


def evaluate(retriever: TrafficHybridRetriever, gold: list[dict], top_k: int, mode: str) -> list[dict]:
    rows: list[dict] = []
    for q in gold:
        t0 = time.time()
        chunks = retriever.get_relevant_chunks(q["question"], top_k=top_k)
        latency_ms = (time.time() - t0) * 1000.0
        retrieved = chunks_to_citation_dicts(chunks)

        rows.append({
            "id":          q["id"],
            "category":    q.get("category", ""),
            "mode":        mode,
            "recall@10":   recall_at_k(retrieved, q["gold_citations"], k=10, level="khoan"),
            "mrr":         mrr(retrieved, q["gold_citations"], level="khoan"),
            "ndcg@10":     ndcg_at_k(retrieved, q["gold_citations"], k=10, level="khoan"),
            "latency_ms":  round(latency_ms, 1),
            "n_primaries": len(retrieved),
        })
    return rows


def summarise(per_q: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(per_q)
    return (
        df.groupby("mode")
          .agg(**{
              "recall@10": ("recall@10", "mean"),
              "mrr":       ("mrr", "mean"),
              "ndcg@10":   ("ndcg@10", "mean"),
              "latency_ms_mean":   ("latency_ms", "mean"),
              "latency_ms_p95":    ("latency_ms", lambda s: s.quantile(0.95)),
              "n":          ("id", "count"),
          })
          .round(4)
          .reset_index()
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--top-n", type=int, default=30,
                   help="Number of RRF candidates to feed into reranker")
    args = p.parse_args()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    gold = load_gold()
    log.info("Loaded %d gold questions", len(gold))

    log.info("=== Mode A: hybrid (no rerank) ===")
    r_a = TrafficHybridRetriever(enable_reranker=False)
    rows_a = evaluate(r_a, gold, top_k=args.top_k, mode="hybrid")
    del r_a

    log.info("=== Mode B: hybrid + bge-reranker-v2-m3 (top_n=%d) ===", args.top_n)
    r_b = TrafficHybridRetriever(enable_reranker=True, rerank_top_n=args.top_n)
    rows_b = evaluate(r_b, gold, top_k=args.top_k, mode="hybrid+rerank")
    del r_b

    all_rows = rows_a + rows_b
    per_q = pd.DataFrame(all_rows)
    summary = summarise(all_rows)

    per_q_path = METRICS_DIR / "rq10_reranker.csv"
    sum_path = METRICS_DIR / "rq10_reranker_summary.csv"
    per_q.to_csv(per_q_path, index=False)
    summary.to_csv(sum_path, index=False)

    log.info("Per-question metrics → %s", per_q_path)
    log.info("Summary → %s", sum_path)

    print("\n=== RQ10 reranker ablation summary ===")
    print(summary.to_string(index=False))

    if len(summary) == 2:
        a = summary[summary["mode"] == "hybrid"].iloc[0]
        b = summary[summary["mode"] == "hybrid+rerank"].iloc[0]
        for metric in ["recall@10", "mrr", "ndcg@10"]:
            base = float(a[metric])
            new = float(b[metric])
            delta_abs = new - base
            delta_rel = (delta_abs / base * 100.0) if base else 0.0
            print(f"  {metric:10s}: {base:.4f} → {new:.4f}  (Δ={delta_abs:+.4f}, {delta_rel:+.1f}%)")
        lat_a = float(a["latency_ms_mean"])
        lat_b = float(b["latency_ms_mean"])
        print(f"  latency  : {lat_a:.0f}ms → {lat_b:.0f}ms (×{lat_b/max(lat_a,1):.1f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
