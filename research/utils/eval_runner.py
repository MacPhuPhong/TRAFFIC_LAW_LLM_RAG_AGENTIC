# -*- coding: utf-8 -*-
"""Run one pipeline over the eval set and persist per-question JSONL output.

Usage (from a notebook):

    from research.utils.eval_runner import run_over_dataset
    rows = run_over_dataset("gemini_only", "research/data/eval_qa.jsonl",
                            out_path="research/results/metrics/rq1_gemini_only.jsonl")
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .pipelines import PipelineResult, build_pipeline

logger = logging.getLogger(__name__)


def load_eval_set(path: str | Path, legal_only: bool = False) -> list[dict]:
    """Load eval gold from JSONL.

    Args:
        path: path to eval_qa.jsonl
        legal_only: if True, filter out questions whose `category` or
            `expected_category` is `"out_of_scope"`. Useful for retrieval /
            chunking / embedding ablations where OOS questions inflate metrics
            (gold_citations is empty so recall is trivially 0 or 1).
    """
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if legal_only:
        rows = [
            r for r in rows
            if r.get("expected_category") != "out_of_scope"
            and r.get("category") != "out_of_scope"
        ]
    return rows


def _resume_completed_ids(out_path: Path) -> set[str]:
    """Skip questions already answered in a prior run (crash recovery)."""
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("id"):
                    done.add(rec["id"])
            except Exception:
                continue
    return done


def run_over_dataset(
    pipeline_name: str,
    eval_path: str | Path,
    out_path: str | Path,
    resume: bool = True,
    sleep_s: float = 0.5,
    pipeline_kwargs: dict | None = None,
) -> list[dict]:
    """Run `pipeline_name` on every row of `eval_path`, appending JSONL to `out_path`.

    Each output row:
        {id, question, gold_answer, gold_citations, expected_category,
         category, answer, contexts, citations, latency_s, error}

    Rows are flushed one-at-a-time so you can kill the process and restart.
    """
    eval_path = Path(eval_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_eval_set(eval_path)
    done = _resume_completed_ids(out_path) if resume else set()
    logger.info("Eval set: %d rows, resume=%s, already_done=%d", len(rows), resume, len(done))

    pipeline = build_pipeline(pipeline_name, **(pipeline_kwargs or {}))
    results: list[dict] = []

    with out_path.open("a", encoding="utf-8") as out:
        for i, row in enumerate(rows, 1):
            qid = row["id"]
            if qid in done:
                continue
            t0 = time.perf_counter()
            err = None
            try:
                pr: PipelineResult = pipeline.run(row["question"])
                payload = pr.to_dict()
            except Exception as exc:  # noqa: BLE001
                err = repr(exc)
                payload = {
                    "answer": "",
                    "contexts": [],
                    "citations": [],
                    "category": None,
                    "latency_s": time.perf_counter() - t0,
                    "cost_usd": None,
                }
                logger.exception("Pipeline %s failed on %s", pipeline_name, qid)

            record = {
                "id": qid,
                "pipeline": pipeline_name,
                "question": row["question"],
                "gold_answer": row.get("gold_answer"),
                "gold_citations": row.get("gold_citations", []),
                "expected_category": row.get("expected_category"),
                "error": err,
                **payload,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            results.append(record)

            logger.info("[%d/%d] %s · %s · %.2fs%s",
                        i, len(rows), qid, pipeline_name, record["latency_s"],
                        " · ERROR" if err else "")
            if sleep_s:
                time.sleep(sleep_s)

    return results


def load_results(path: str | Path) -> list[dict]:
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
