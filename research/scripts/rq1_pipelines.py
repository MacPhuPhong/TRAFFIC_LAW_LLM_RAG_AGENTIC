# -*- coding: utf-8 -*-
"""RQ1 — Three-way comparison: Gemini-only vs Vanilla RAG vs Agentic RAG.

Runs each pipeline over the 25-question gold set, writes JSONL per pipeline,
and a summary CSV with F1 (token-level) / ROUGE-L / mean latency.

This is the script equivalent of notebook 01_rq1_gemini_vs_rag_vs_agentic.ipynb.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(ROOT.parent / ".env")
except ImportError:
    pass
if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["API_KEY"]

import pandas as pd  # noqa: E402

from research.utils.eval_runner import load_results, run_over_dataset  # noqa: E402
from research.utils.metrics import rouge_l, token_f1  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rq1")

EVAL_PATH = ROOT / "research" / "data" / "eval_qa.jsonl"
METRICS_DIR = ROOT / "research" / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

PIPELINES = ["gemini_only", "vanilla_rag", "agentic_rag"]


def main():
    rows = []
    for name in PIPELINES:
        out_path = METRICS_DIR / f"rq1_{name}.jsonl"
        log.info("=== %s → %s ===", name, out_path)
        run_over_dataset(name, EVAL_PATH, out_path=out_path, resume=True)

        for r in load_results(out_path):
            pred_cat = r.get("category")
            exp_cat = r.get("expected_category")
            rows.append({
                "pipeline": name,
                "id": r["id"],
                "category": exp_cat,
                "predicted_category": pred_cat,
                "category_correct": int(pred_cat == exp_cat) if pred_cat and exp_cat else None,
                "f1": token_f1(r.get("answer") or "", r.get("gold_answer") or ""),
                "rouge_l": rouge_l(r.get("answer") or "", r.get("gold_answer") or ""),
                "latency_s": r.get("latency_s"),
            })

    df = pd.DataFrame(rows)
    df.to_csv(METRICS_DIR / "rq1_per_question.csv", index=False)

    # Aggregate. category_correct may be None for pipelines that don't route
    # (gemini_only). Use .dropna() so the mean reflects only pipelines that
    # produced a routing decision.
    summary = df.groupby("pipeline").agg(
        f1=("f1", "mean"),
        rouge_l=("rouge_l", "mean"),
        latency_s=("latency_s", "mean"),
        category_acc=("category_correct", "mean"),
    ).round(4).reset_index()
    summary.to_csv(METRICS_DIR / "rq1_summary.csv", index=False)
    log.info("\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
