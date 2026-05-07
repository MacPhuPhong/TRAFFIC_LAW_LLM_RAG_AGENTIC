# -*- coding: utf-8 -*-
"""LangSmith end-to-end demo for the Traffic-RAG project.

What it does (in order):

  Step 1.  verify the LangSmith connection (key + project visible)
  Step 2.  trace 3 sample queries through the vanilla_rag pipeline
           → prints the trace URL of each run so you can open them
  Step 3.  push `research/data/eval_qa.jsonl` to LangSmith as a dataset
           (idempotent — skips if the dataset already exists)
  Step 4.  run `evaluate()` over a subset of that dataset, attaching all
           evaluators from `research.utils.langsmith_evaluators`

Usage:

    # full demo (Step 1-4)
    python -m research.scripts.langsmith_demo

    # only steps 1-2 (quick smoke test, no eval cost)
    python -m research.scripts.langsmith_demo --skip-eval

    # eval against a specific subset size
    python -m research.scripts.langsmith_demo --n 5

Requires GOOGLE_API_KEY + LANGCHAIN_API_KEY in .env.
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
ROOT = HERE.parent.parent  # .../traffic_rag
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.utils.langsmith_setup import enable_tracing, verify_connection  # noqa: E402


def _alias_api_keys() -> None:
    """Bridge legacy `.env` variable names so downstream modules find what they
    expect. Same behaviour as api/main.py — `API_KEY` is the project's old
    convention for the Gemini key.
    """
    if not os.getenv("GOOGLE_API_KEY") and os.getenv("API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["API_KEY"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("ls-demo")

EVAL_PATH = ROOT / "research" / "data" / "eval_qa.jsonl"
DEMO_QUERIES = [
    "Vượt đèn đỏ đi xe máy phạt bao nhiêu tiền?",
    "Nồng độ cồn mức 1 đối với ô tô phạt bao nhiêu?",
    "Không gắn biển số xe ô tô phạt bao nhiêu?",
]


# ---------------------------------------------------------------------------
# Step 1 — verify
# ---------------------------------------------------------------------------

def step1_verify(project: str) -> bool:
    log.info("=" * 60)
    log.info("STEP 1 — verify LangSmith connection")
    log.info("=" * 60)
    enable_tracing(project=project, run_name="langsmith_demo")
    info = verify_connection()
    if info["ok"]:
        log.info("OK — project='%s'", info["project"])
        log.info("Project URL: %s", info["url"])
        return True
    log.error("FAIL — %s", info["error"])
    log.error("Set LANGCHAIN_API_KEY in .env and try again.")
    return False


# ---------------------------------------------------------------------------
# Step 2 — trace 3 sample queries
# ---------------------------------------------------------------------------

def step2_trace_samples(project: str) -> None:
    log.info("=" * 60)
    log.info("STEP 2 — trace 3 sample queries through vanilla_rag")
    log.info("=" * 60)

    from langsmith import Client
    from langsmith.run_helpers import tracing_context
    from research.utils.pipelines import build_pipeline

    pipe = build_pipeline("vanilla_rag")
    client = Client()

    for i, q in enumerate(DEMO_QUERIES, 1):
        run_name = f"demo_q{i}"
        log.info("[%d/%d] Q: %s", i, len(DEMO_QUERIES), q)
        # `tracing_context` lets us attach the run to a known name + project
        # so we can fetch its URL afterwards. Inside this block, every
        # langsmith-instrumented call (LangChain LLM, our @traceable
        # retriever) is grouped under one parent run.
        with tracing_context(project_name=project, name=run_name, tags=["demo"]):
            t0 = time.perf_counter()
            result = pipe.run(q)
            dt = time.perf_counter() - t0
        # Try to surface the URL of the most recent run with this name.
        try:
            latest = next(client.list_runs(
                project_name=project, filter=f'eq(name, "{run_name}")', limit=1,
            ), None)
            url = client.read_run(latest.id).url if latest else None
        except Exception as exc:
            url = None
            log.debug("could not fetch run URL: %s", exc)
        log.info("    answer: %s", (result.answer or "")[:120].replace("\n", " "))
        log.info("    contexts retrieved: %d, latency: %.2fs", len(result.contexts), dt)
        if url:
            log.info("    trace: %s", url)


# ---------------------------------------------------------------------------
# Step 3 — push eval set as a LangSmith dataset
# ---------------------------------------------------------------------------

def step3_push_dataset(dataset_name: str, n: int | None = None) -> str | None:
    log.info("=" * 60)
    log.info("STEP 3 — push eval set to LangSmith dataset '%s'", dataset_name)
    log.info("=" * 60)

    from langsmith import Client

    client = Client()
    if not EVAL_PATH.exists():
        log.error("eval file missing: %s", EVAL_PATH)
        return None

    rows = []
    with EVAL_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if n:
        rows = rows[:n]
    log.info("loaded %d examples from %s", len(rows), EVAL_PATH.name)

    # Idempotency: if the dataset exists, return its ID and skip uploading.
    try:
        existing = client.read_dataset(dataset_name=dataset_name)
        log.info("dataset already exists (id=%s) — skipping upload", existing.id)
        log.info("dataset URL: %s", existing.url)
        return str(existing.id)
    except Exception:
        pass  # not found → create it

    ds = client.create_dataset(
        dataset_name=dataset_name,
        description="Traffic-Law RAG eval — 25 Vietnamese legal QA pairs",
    )
    log.info("created dataset id=%s", ds.id)

    inputs = [{"question": r["question"],
               "expected_category": r.get("expected_category")} for r in rows]
    outputs = [{"gold_answer":     r.get("gold_answer"),
                "gold_citations":  r.get("gold_citations", []),
                "expected_category": r.get("expected_category")} for r in rows]
    client.create_examples(inputs=inputs, outputs=outputs, dataset_id=ds.id)
    log.info("uploaded %d examples", len(rows))
    log.info("dataset URL: %s", ds.url)
    return str(ds.id)


# ---------------------------------------------------------------------------
# Step 4 — run evaluate() against the dataset
# ---------------------------------------------------------------------------

def step4_evaluate(dataset_name: str, n: int, experiment_prefix: str,
                   pipeline_name: str = "vanilla_rag") -> None:
    log.info("=" * 60)
    log.info("STEP 4 — run evaluate() on dataset '%s' (n=%d, pipeline=%s)",
             dataset_name, n, pipeline_name)
    log.info("=" * 60)

    from langsmith.evaluation import evaluate
    from research.utils.langsmith_evaluators import (
        ALL_EVALUATORS, HEURISTIC_EVALUATORS,
    )
    from research.utils.pipelines import build_pipeline

    # Use heuristic-only by default to avoid burning judge calls; flip via env.
    use_judge = os.getenv("LANGSMITH_DEMO_USE_JUDGE", "false").lower() in ("1", "true", "yes")
    evaluators = ALL_EVALUATORS if use_judge else HEURISTIC_EVALUATORS
    log.info("evaluators: %s", [e.__name__ for e in evaluators])

    pipe = build_pipeline(pipeline_name)

    def target(inputs: dict) -> dict:
        result = pipe.run(inputs["question"])
        # Preserve chunk metadata (doc_id, dieu, khoan, ...) so evaluators can
        # match (doc_id, dieu) directly without parsing free text. Falls back
        # to plain content strings only if the pipeline didn't surface raw
        # chunks.
        contexts = []
        raw_chunks = (result.raw or {}).get("chunks") if result.raw else None
        if raw_chunks:
            for c in raw_chunks:
                if isinstance(c, dict):
                    contexts.append({
                        "content":  c.get("content", ""),
                        "metadata": c.get("metadata", {}),
                    })
                elif hasattr(c, "to_dict"):
                    d = c.to_dict()
                    contexts.append({
                        "content":  d.get("content", ""),
                        "metadata": d.get("metadata", {}),
                    })
                elif hasattr(c, "content"):
                    md = getattr(c, "metadata", None) or {}
                    contexts.append({"content": c.content, "metadata": md})
        if not contexts:
            contexts = [{"content": c} for c in result.contexts]
        return {
            "answer":     result.answer,
            "contexts":   contexts,
            "citations":  result.citations,
            "category":   result.category,
        }

    # Cap dataset size by passing a generator that yields n examples.
    from langsmith import Client
    client = Client()
    examples = list(client.list_examples(dataset_name=dataset_name))[:n]
    log.info("running on %d examples (max_concurrency=2)", len(examples))

    results = evaluate(
        target,
        data=examples,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
        max_concurrency=int(os.getenv("LANGSMITH_DEMO_CONCURRENCY", "1")),
    )
    log.info("eval done — view results at: %s",
             getattr(results, "experiment_url", None) or
             "(open the dataset on smith.langchain.com)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=os.getenv("LANGCHAIN_PROJECT",
                                                   "Traffic-RAG-Evaluation"))
    p.add_argument("--dataset", default="traffic-rag-eval-v1")
    p.add_argument("--n", type=int, default=5,
                   help="how many examples to run in step 4")
    p.add_argument("--pipeline", default="vanilla_rag",
                   choices=["vanilla_rag", "agentic_rag", "gemini_only"])
    p.add_argument("--skip-eval", action="store_true",
                   help="run only steps 1-2 (no dataset upload, no eval)")
    p.add_argument("--skip-samples", action="store_true",
                   help="skip step 2 (the 3 demo queries)")
    args = p.parse_args()

    if not step1_verify(args.project):
        return 1
    _alias_api_keys()

    if not args.skip_samples:
        try:
            step2_trace_samples(args.project)
        except Exception as exc:
            log.exception("STEP 2 failed: %s", exc)

    if args.skip_eval:
        log.info("--skip-eval set — done.")
        return 0

    ds_id = step3_push_dataset(args.dataset, n=None)
    if not ds_id:
        return 1

    try:
        step4_evaluate(args.dataset, n=args.n,
                       experiment_prefix=f"{args.pipeline}-demo",
                       pipeline_name=args.pipeline)
    except Exception as exc:
        log.exception("STEP 4 failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
