# -*- coding: utf-8 -*-
"""RQ5 — Prompt ablation: P0 zero-shot vs P1 role-only vs P2 full engineered rules.

All three variants share:
  * Same retriever (TrafficHybridRetriever, top_k=10).
  * Same LLM (gemini-3.1-flash-lite-preview, temp=0.1).
  * Same citation-extraction logic (post-hoc regex on the answer).

What changes: the SYSTEM_PROMPT text the LLM sees. We monkey-patch
`source.rag_core.generator.SYSTEM_PROMPT` before each call so the generator's
own pipeline is reused verbatim — no forked generator.

Metrics per variant: F1 (token), ROUGE-L, Cit-P/R/F1 (khoan level), refusal
rate, latency. 3 × 25 = 75 LLM calls; sleeps 1.5s between calls to stay
under Gemini free-tier rate limits.
"""

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
from source.rag_core import generator as gen_mod  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("rq5")

GOLD = ROOT / "research" / "data" / "eval_qa.jsonl"
METRICS_DIR = ROOT / "research" / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

REFUSAL_PAT = ["không đủ thông tin", "không tìm thấy", "không có trong",
               "tôi không biết"]
def is_refusal(txt: str) -> bool:
    t = (txt or "").lower()
    return any(p in t for p in REFUSAL_PAT)

P0_ZERO_SHOT = "Trả lời câu hỏi dựa vào NGỮ CẢNH bên dưới."

P1_ROLE_ONLY = (
    "Bạn là Trợ lý Pháp lý Giao thông Việt Nam. "
    "Trả lời câu hỏi của người dùng bằng tiếng Việt dựa trên NGỮ CẢNH bên dưới."
)

P2_FULL = gen_mod.SYSTEM_PROMPT  # baseline (already loaded from module)


VARIANTS = [
    ("P0_zero_shot", P0_ZERO_SHOT),
    ("P1_role_only", P1_ROLE_ONLY),
    ("P2_full_rules", P2_FULL),
]


def load_gold() -> list[dict]:
    return [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]


def run_variant(name: str, prompt: str, retriever, generator, gold, sleep_s: float = 5.0):
    original = gen_mod.SYSTEM_PROMPT
    gen_mod.SYSTEM_PROMPT = prompt
    rows = []
    try:
        for item in gold:
            t0 = time.perf_counter()
            try:
                chunks = retriever.get_relevant_chunks(item["question"], top_k=10)
                chunk_dicts = [c.to_dict() for c in chunks]
                out = generator.generate(item["question"], chunk_dicts)
            except Exception as e:
                log.error("  [%s] %s error: %s", name, item["id"], e)
                out = {"answer": "", "sources": []}
            dt = time.perf_counter() - t0
            pred = out.get("answer", "")
            cites = out.get("sources") or []
            g = item["gold_citations"]
            p, r, f = citation_precision_recall(cites, g, "khoan")
            rows.append({
                "variant": name,
                "qid": item["id"],
                "category": item["category"],
                "latency_s": dt,
                "f1_token": token_f1(pred, item["gold_answer"]),
                "rouge_l": rouge_l(pred, item["gold_answer"]),
                "cit_p": p, "cit_r": r, "cit_f1": f,
                "refusal": is_refusal(pred),
                "pred_answer": pred[:500],
            })
            log.info("  [%s] %s F1=%.3f Cit-R=%.2f refusal=%s (%.1fs)",
                     name, item["id"], rows[-1]["f1_token"], r, rows[-1]["refusal"], dt)
            time.sleep(sleep_s)
    finally:
        gen_mod.SYSTEM_PROMPT = original
    return rows


def main():
    gold = load_gold()
    log.info("Loaded %d gold questions", len(gold))

    log.info("Loading retriever + generator (one-shot)")
    retriever = TrafficHybridRetriever()
    generator = LegalAnswerGenerator(provider="google",
                                     model="gemini-3.1-flash-lite-preview")

    all_rows = []
    for name, prompt in VARIANTS:
        log.info("=== %s ===", name)
        all_rows.extend(run_variant(name, prompt, retriever, generator, gold))

    df = pd.DataFrame(all_rows)
    df.to_csv(METRICS_DIR / "rq5_prompts_per_question.csv", index=False)

    summary = df.groupby("variant").agg({
        "f1_token": "mean", "rouge_l": "mean",
        "cit_p": "mean", "cit_r": "mean", "cit_f1": "mean",
        "refusal": "mean", "latency_s": "mean",
    }).round(4)
    summary.to_csv(METRICS_DIR / "rq5_prompts_summary.csv")
    log.info("\n%s", summary.to_string())
    log.info("DONE.")


if __name__ == "__main__":
    main()
