# -*- coding: utf-8 -*-
"""RQ8 — LangSmith trace pull.

Uses the LangSmith Python client to list every run in the
`LANGCHAIN_PROJECT="Traffic-RAG-Evaluation"` project and aggregates:
  * Per-run-name (analyzer / retriever / generator / ...): count, mean,
    p50, p95 total_time, error rate, tokens, cost.
  * Per-pipeline latency histogram.

Fallback: if LangSmith returns 0 runs, read the local RQ1 JSONL files
and produce equivalent (but less granular) per-pipeline stats.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
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

import pandas as pd  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("rq8")

METRICS_DIR = ROOT / "research" / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

PROJECT = os.environ.get("LANGCHAIN_PROJECT", "Traffic-RAG-Evaluation")


def pull_from_langsmith():
    try:
        from langsmith import Client
    except ImportError:
        log.warning("langsmith not installed — fallback to local JSONL")
        return None
    key = os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY")
    if not key:
        log.warning("No LANGCHAIN_API_KEY — fallback")
        return None
    os.environ["LANGSMITH_API_KEY"] = key
    client = Client(api_key=key)
    log.info("Pulling runs from project '%s' ...", PROJECT)
    try:
        # list_runs is a generator; iterate with a soft cap.
        runs = []
        for i, r in enumerate(client.list_runs(project_name=PROJECT)):
            runs.append(r)
            if i + 1 >= 2000:
                break
    except Exception as e:
        log.error("LangSmith error: %s", e)
        return None
    log.info("  got %d runs", len(runs))
    if not runs:
        return None

    def dt_s(r):
        if r.start_time and r.end_time:
            return (r.end_time - r.start_time).total_seconds()
        return None

    # Filter ragas internal wrappers ("row 0".."row 24", "ragas evaluation")
    import re as _re
    def _keep(r):
        if _re.fullmatch(r"row \d+", r.name or ""):
            return False
        if r.name == "ragas evaluation":
            return False
        return True

    runs = [r for r in runs if _keep(r)]
    log.info("  after filter: %d runs", len(runs))

    groups = defaultdict(list)
    for r in runs:
        groups[r.name].append(r)

    rows = []
    for name, items in groups.items():
        lats = [dt_s(r) for r in items]
        lats = [x for x in lats if x is not None]
        errs = sum(1 for r in items if r.error)
        toks = [r.total_tokens for r in items
                if getattr(r, "total_tokens", None) is not None]
        costs = [float(r.total_cost) for r in items
                 if getattr(r, "total_cost", None) is not None]
        if not lats:
            continue
        lats_sorted = sorted(lats)
        p50 = lats_sorted[len(lats_sorted)//2]
        p95 = lats_sorted[max(0, int(len(lats_sorted)*0.95) - 1)]
        rows.append({
            "run_name": name,
            "run_type": items[0].run_type,
            "count": len(items),
            "mean_s": round(sum(lats)/len(lats), 3),
            "p50_s": round(p50, 3),
            "p95_s": round(p95, 3),
            "error_rate": round(errs/len(items), 3),
            "total_tokens_mean": round(sum(toks)/len(toks), 1) if toks else 0,
            "cost_usd_mean": round(sum(costs)/len(costs), 6) if costs else 0,
        })
    rows.sort(key=lambda x: -x["count"])
    return rows


def fallback_from_jsonl():
    log.info("Fallback: aggregating per-pipeline from rq1_*.jsonl")
    rows = []
    for pname in ("gemini_only", "vanilla_rag", "agentic_rag"):
        fp = METRICS_DIR / f"rq1_{pname}.jsonl"
        if not fp.exists():
            continue
        lats = []
        errs = 0
        for line in open(fp, encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("latency_s") is not None:
                lats.append(d["latency_s"])
            if d.get("error"):
                errs += 1
        if not lats:
            continue
        lats_sorted = sorted(lats)
        p50 = lats_sorted[len(lats_sorted)//2]
        p95 = lats_sorted[max(0, int(len(lats_sorted)*0.95) - 1)]
        rows.append({
            "run_name": pname,
            "count": len(lats),
            "mean_s": round(sum(lats)/len(lats), 3),
            "p50_s": round(p50, 3),
            "p95_s": round(p95, 3),
            "error_rate": round(errs/len(lats), 3),
            "total_tokens_mean": 0,
        })
    return rows


def main():
    rows = pull_from_langsmith()
    source = "langsmith"
    if not rows:
        rows = fallback_from_jsonl()
        source = "local_jsonl_fallback"

    df = pd.DataFrame(rows)
    df["source"] = source
    out = METRICS_DIR / "rq8_langsmith_nodes.csv"
    df.to_csv(out, index=False)
    log.info("Wrote %s (source=%s, %d rows)", out, source, len(df))
    log.info("\n%s", df.to_string())


if __name__ == "__main__":
    main()
