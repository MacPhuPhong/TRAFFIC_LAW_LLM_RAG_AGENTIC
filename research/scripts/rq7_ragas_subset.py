# -*- coding: utf-8 -*-
"""RQ7-subset — Re-run RAGAS judge eval on a small subset with backoff.

Background: the original RQ7 (notebook 07_rq7_answer_metrics_ragas.ipynb) was
abandoned because Gemini judge throttled 93.5% of calls — RAGAS calls the
judge ~60 times per sample, so 25 samples × 4 metrics × ~60 calls = ~6 000
judge calls in a tight loop, well above the free-tier rate limit.

This script runs RAGAS on a much smaller subset (default n=5) with explicit
sleep + retry between each metric, so the judge stays under the rate limit.

Usage:
  python research/scripts/rq7_ragas_subset.py [--n 5] [--sleep 8] [--judge gemini-1.5-flash]

Outputs:
  - research/results/metrics/rq7_ragas_subset.csv         — per-question metrics
  - research/results/metrics/rq7_ragas_subset_summary.csv — aggregate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("rq7-subset")

EVAL_PATH = ROOT / "research" / "data" / "eval_qa.jsonl"
AGENTIC_RESULTS = ROOT / "research" / "results" / "metrics" / "rq1_agentic_rag.jsonl"
METRICS_DIR = ROOT / "research" / "results" / "metrics"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def build_subset(n: int, seed: int = 42) -> list[dict]:
    """Pick the first `n` items that have non-empty contexts and answers.

    Stable order (no shuffle) so reruns produce the same subset.
    """
    agentic = load_jsonl(AGENTIC_RESULTS)
    gold_by_id = {g["id"]: g for g in load_jsonl(EVAL_PATH)}

    rows: list[dict] = []
    for r in agentic:
        if r.get("error"):
            continue
        if not r.get("answer") or not r.get("contexts"):
            continue
        gold = gold_by_id.get(r["id"])
        if not gold:
            continue
        rows.append({
            "id":           r["id"],
            "question":     r["question"],
            "answer":       r["answer"],
            "contexts":     r["contexts"][:5],   # cap to 5 chunks/sample
            "ground_truth": gold["gold_answer"],
        })
        if len(rows) >= n:
            break
    return rows


def run_metric_with_retry(metric, dataset, llm, max_attempts: int = 4, base_sleep: float = 8.0):
    """Run a single RAGAS metric on the dataset, retrying with exponential
    backoff on throttle/quota errors. Returns the metric DataFrame or None.
    """
    from ragas import evaluate

    for attempt in range(1, max_attempts + 1):
        try:
            res = evaluate(dataset, metrics=[metric], llm=llm)
            return res.to_pandas()
        except Exception as exc:
            msg = str(exc).lower()
            throttled = any(s in msg for s in ("429", "quota", "rate", "throttle", "resourceexhausted"))
            wait = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 2)
            log.warning(
                "metric=%s attempt=%d failed (%s%s) — sleep %.1fs",
                metric.name, attempt,
                "throttled" if throttled else "error: ", exc.__class__.__name__,
                wait,
            )
            time.sleep(wait)
    log.error("metric=%s gave up after %d attempts", metric.name, max_attempts)
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5,
                   help="Number of questions in the subset (default 5)")
    p.add_argument("--sleep", type=float, default=8.0,
                   help="Seconds to sleep between metrics (default 8s)")
    p.add_argument("--judge", default="gemini-1.5-flash",
                   help="Judge LLM model name")
    args = p.parse_args()

    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("API_KEY"):
        log.error("GOOGLE_API_KEY missing — cannot run judge")
        return 1

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    subset = build_subset(args.n)
    log.info("Subset size: %d questions (ids=%s)", len(subset),
             [r["id"] for r in subset])

    from datasets import Dataset
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas.metrics import (
        answer_relevancy, context_precision,
        context_recall, faithfulness,
    )

    llm = ChatGoogleGenerativeAI(
        model=args.judge,
        google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("API_KEY"),
        temperature=0.0,
    )

    data_map = {k: [r[k] for r in subset]
                for k in ("question", "answer", "contexts", "ground_truth")}
    dataset = Dataset.from_dict(data_map)

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    per_metric = {}
    for m in metrics:
        log.info("=== metric: %s ===", m.name)
        df = run_metric_with_retry(m, dataset, llm)
        if df is not None:
            per_metric[m.name] = df[m.name].tolist() if m.name in df.columns else [float("nan")] * len(subset)
        else:
            per_metric[m.name] = [float("nan")] * len(subset)
        log.info("Sleep %.1fs before next metric…", args.sleep)
        time.sleep(args.sleep)

    import pandas as pd
    df_per_q = pd.DataFrame({"id": [r["id"] for r in subset], **per_metric})
    df_per_q_path = METRICS_DIR / "rq7_ragas_subset.csv"
    df_per_q.to_csv(df_per_q_path, index=False)

    summary_rows = []
    for name, vals in per_metric.items():
        clean = [v for v in vals if isinstance(v, (int, float)) and v == v]  # drop NaN
        summary_rows.append({
            "metric":  name,
            "n":       len(clean),
            "mean":    round(sum(clean) / len(clean), 4) if clean else None,
            "min":     round(min(clean), 4) if clean else None,
            "max":     round(max(clean), 4) if clean else None,
        })
    df_sum = pd.DataFrame(summary_rows)
    sum_path = METRICS_DIR / "rq7_ragas_subset_summary.csv"
    df_sum.to_csv(sum_path, index=False)

    log.info("Per-question → %s", df_per_q_path)
    log.info("Summary       → %s", sum_path)
    print("\n=== RQ7 RAGAS subset summary ===")
    print(df_sum.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
