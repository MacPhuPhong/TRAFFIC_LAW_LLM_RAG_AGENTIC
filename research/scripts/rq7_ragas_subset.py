# -*- coding: utf-8 -*-
"""RQ7 — RAGAS judge eval on the 35-question legal-only subset using OpenAI.

Background: `langchain_google_genai 1.x` is incompatible with `ragas 0.2`
(TypeError: `generate_content() got 'temperature'`). We switched the judge to
OpenAI (default `gpt-4o-mini`, ~$2/run for 35 questions × 4 metrics).

Usage:
  python research/scripts/rq7_ragas_subset.py
  python research/scripts/rq7_ragas_subset.py --judge gpt-4o
  python research/scripts/rq7_ragas_subset.py --pipeline vanilla_rag

Outputs:
  - research/results/metrics/rq7_ragas.csv         — per-question metrics
  - research/results/metrics/rq7_ragas_summary.csv — aggregate (mean/min/max/n)
"""

from __future__ import annotations

import argparse
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

from research.utils.eval_runner import load_eval_set  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("rq7-ragas")

EVAL_PATH = ROOT / "research" / "data" / "eval_qa.jsonl"
METRICS_DIR = ROOT / "research" / "results" / "metrics"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def build_subset(pipeline: str, legal_only: bool = True) -> list[dict]:
    """Build RAGAS-ready subset from an existing rq1 pipeline output.

    Requires the pipeline output JSONL to have `contexts` + `answer` filled.
    Only includes questions whose category is not out_of_scope when
    `legal_only=True`.
    """
    pipeline_results = ROOT / "research" / "results" / "metrics" / f"rq1_{pipeline}.jsonl"
    if not pipeline_results.exists():
        raise FileNotFoundError(
            f"Pipeline output missing: {pipeline_results}. "
            f"Run rq1_pipelines.py first."
        )
    runs = load_jsonl(pipeline_results)
    gold_by_id = {g["id"]: g for g in load_eval_set(EVAL_PATH, legal_only=legal_only)}

    rows: list[dict] = []
    for r in runs:
        gold = gold_by_id.get(r["id"])
        if not gold:  # filtered out (e.g., OOS)
            continue
        if r.get("error"):
            continue
        if not r.get("answer") or not r.get("contexts"):
            log.debug("Skipping %s: missing answer/contexts", r["id"])
            continue
        rows.append({
            "id":           r["id"],
            "question":     r["question"],
            "answer":       r["answer"],
            "contexts":     r["contexts"][:5],
            "ground_truth": gold["gold_answer"],
        })
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", default="agentic_rag",
                   choices=["gemini_only", "vanilla_rag", "agentic_rag"],
                   help="Which pipeline's outputs to evaluate (default agentic_rag)")
    p.add_argument("--judge", default=None,
                   help="OpenAI judge model. Defaults to env RAGAS_JUDGE_MODEL or gpt-4o-mini")
    p.add_argument("--include-oos", action="store_true",
                   help="Include out_of_scope questions (default: legal-only)")
    args = p.parse_args()

    openai_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("API_KEY_GPT")
    )
    if not openai_key:
        log.error("OpenAI key missing. Set OPENAI_API_KEY or API_KEY_GPT in .env.")
        return 1
    os.environ["OPENAI_API_KEY"] = openai_key

    judge_model = (
        args.judge
        or os.getenv("RAGAS_JUDGE_MODEL")
        or os.getenv("GENERATOR_MODE_GPT")
        or "gpt-4o-mini"
    )
    log.info("Judge model: %s", judge_model)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    subset = build_subset(args.pipeline, legal_only=not args.include_oos)
    log.info("Subset size: %d questions from pipeline=%s (legal_only=%s)",
             len(subset), args.pipeline, not args.include_oos)

    from datasets import Dataset
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy, context_precision,
        context_recall, faithfulness,
    )

    llm = ChatOpenAI(
        model=judge_model,
        api_key=openai_key,
        temperature=0.0,
    )

    data_map = {k: [r[k] for r in subset]
                for k in ("question", "answer", "contexts", "ground_truth")}
    dataset = Dataset.from_dict(data_map)

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    log.info("Running RAGAS on %d questions × %d metrics (~%d judge calls)",
             len(subset), len(metrics), len(subset) * len(metrics) * 5)

    t0 = time.perf_counter()
    result = evaluate(dataset, metrics=metrics, llm=llm)
    elapsed = time.perf_counter() - t0
    log.info("RAGAS finished in %.1fs", elapsed)

    df = result.to_pandas()
    df.insert(0, "id", [r["id"] for r in subset])
    df.insert(1, "pipeline", args.pipeline)

    per_q_path = METRICS_DIR / "rq7_ragas.csv"
    df.to_csv(per_q_path, index=False)

    import pandas as pd
    summary_rows = []
    for m in metrics:
        if m.name not in df.columns:
            continue
        vals = df[m.name].dropna().tolist()
        summary_rows.append({
            "metric": m.name,
            "n":      len(vals),
            "mean":   round(sum(vals) / len(vals), 4) if vals else None,
            "min":    round(min(vals), 4) if vals else None,
            "max":    round(max(vals), 4) if vals else None,
        })
    df_sum = pd.DataFrame(summary_rows)
    df_sum["judge_model"] = judge_model
    df_sum["pipeline"] = args.pipeline
    df_sum["elapsed_s"] = round(elapsed, 1)
    sum_path = METRICS_DIR / "rq7_ragas_summary.csv"
    df_sum.to_csv(sum_path, index=False)

    log.info("Per-question → %s", per_q_path)
    log.info("Summary       → %s", sum_path)
    print("\n=== RQ7 RAGAS summary ===")
    print(df_sum.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
