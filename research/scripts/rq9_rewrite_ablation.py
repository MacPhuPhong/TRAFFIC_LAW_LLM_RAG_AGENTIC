# -*- coding: utf-8 -*-
"""RQ9 — Query-rewrite ablation: NO_REWRITE (raw user question) vs
REWRITE (analyzer expands to legal terminology + standalone form).

Both variants share:
  * Same retriever (TrafficHybridRetriever, top_k=10).
  * Same generator (LegalAnswerGenerator with production P2 prompt).
  * Same citation/answer scoring.

What differs: the STRING passed to the retriever (and to the generator as the
"question" it should answer).
  * NO_REWRITE: raw user question for both.
  * REWRITE: analyzer_node(raw) produces (standalone_query, expanded_query);
    retriever sees expanded_query, generator answers standalone_query.

Metrics per question: retrieval R@5/R@10/MRR/nDCG@10 (khoan level), answer
F1/ROUGE-L/Cit-P/R/F1, refusal, latency. 2 variants × 25 = 50 LLM calls
(generator) + 25 analyzer calls for REWRITE side.
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
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402

from research.utils.metrics import (  # noqa: E402
    citation_precision_recall, mrr, ndcg_at_k,
    recall_at_k, rouge_l, token_f1,
)
from source.agent.nodes import make_analyzer_node  # noqa: E402
from source.rag_core import LegalAnswerGenerator, TrafficHybridRetriever  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("rq9")

GOLD = ROOT / "research" / "data" / "eval_qa.jsonl"
METRICS_DIR = ROOT / "research" / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

REFUSAL_PAT = ["không đủ thông tin", "không tìm thấy", "không có trong",
               "tôi không biết"]

GEN_MODEL = "gemini-3.1-flash-lite-preview"
ANALYZER_MODEL = "gemini-3.1-flash-lite-preview"
SLEEP_S = 5.0


def is_refusal(txt: str) -> bool:
    t = (txt or "").lower()
    return any(p in t for p in REFUSAL_PAT)


def load_gold() -> list[dict]:
    # Legal-only: 35 questions (drop 5 out_of_scope).
    rows = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]
    return [r for r in rows
            if r.get("expected_category") != "out_of_scope"
            and r.get("category") != "out_of_scope"]


def _retrieved_cites(chunk_dicts: list[dict]) -> list[dict]:
    out = []
    for c in chunk_dicts:
        m = c.get("metadata") or {}
        out.append({
            "doc_id": m.get("doc_id"), "dieu": m.get("dieu"),
            "khoan": m.get("khoan"), "diem": m.get("diem"),
        })
    return out


def run_no_rewrite(gold, retriever, generator) -> list[dict]:
    rows = []
    for item in gold:
        raw_q = item["question"]
        t0 = time.perf_counter()
        try:
            chunks = retriever.get_relevant_chunks(raw_q, top_k=10)
            chunk_dicts = [c.to_dict() for c in chunks]
            out = generator.generate(raw_q, chunk_dicts)
        except Exception as e:
            log.error("  [NO_REWRITE] %s error: %s", item["id"], e)
            chunk_dicts, out = [], {"answer": "", "sources": []}
        dt = time.perf_counter() - t0

        retrieved_meta = _retrieved_cites(chunk_dicts)
        g = item["gold_citations"]
        pred = out.get("answer", "")
        p, r, f = citation_precision_recall(out.get("sources") or [], g, "khoan")
        rows.append({
            "variant": "NO_REWRITE", "qid": item["id"], "category": item["category"],
            "used_query": raw_q[:200],
            "r_at_5":   recall_at_k(retrieved_meta, g, 5,  "khoan"),
            "r_at_10":  recall_at_k(retrieved_meta, g, 10, "khoan"),
            "mrr":      mrr(retrieved_meta, g, "khoan"),
            "ndcg_10":  ndcg_at_k(retrieved_meta, g, 10, "khoan"),
            "f1_token": token_f1(pred, item["gold_answer"]),
            "rouge_l":  rouge_l(pred, item["gold_answer"]),
            "cit_p": p, "cit_r": r, "cit_f1": f,
            "refusal": is_refusal(pred),
            "latency_s": dt,
            "pred_answer": pred[:500],
        })
        log.info("  [NO_REWRITE] %s R@10=%.2f MRR=%.2f F1=%.3f Cit-R=%.2f (%.1fs)",
                 item["id"], rows[-1]["r_at_10"], rows[-1]["mrr"],
                 rows[-1]["f1_token"], r, dt)
        time.sleep(SLEEP_S)
    return rows


def run_rewrite(gold, retriever, generator, analyzer_node) -> list[dict]:
    rows = []
    for item in gold:
        raw_q = item["question"]
        t0 = time.perf_counter()
        try:
            state = {"query": raw_q, "chat_history": []}
            az = analyzer_node(state)
            standalone_q = az.get("query") or raw_q
            expanded_q = az.get("expanded_query") or raw_q

            chunks = retriever.get_relevant_chunks(expanded_q, top_k=10)
            chunk_dicts = [c.to_dict() for c in chunks]
            out = generator.generate(standalone_q, chunk_dicts)
        except Exception as e:
            log.error("  [REWRITE] %s error: %s", item["id"], e)
            chunk_dicts, out = [], {"answer": "", "sources": []}
            standalone_q, expanded_q = raw_q, raw_q
        dt = time.perf_counter() - t0

        retrieved_meta = _retrieved_cites(chunk_dicts)
        g = item["gold_citations"]
        pred = out.get("answer", "")
        p, r, f = citation_precision_recall(out.get("sources") or [], g, "khoan")
        rows.append({
            "variant": "REWRITE", "qid": item["id"], "category": item["category"],
            "used_query": expanded_q[:200],
            "standalone_query": standalone_q[:200],
            "r_at_5":   recall_at_k(retrieved_meta, g, 5,  "khoan"),
            "r_at_10":  recall_at_k(retrieved_meta, g, 10, "khoan"),
            "mrr":      mrr(retrieved_meta, g, "khoan"),
            "ndcg_10":  ndcg_at_k(retrieved_meta, g, 10, "khoan"),
            "f1_token": token_f1(pred, item["gold_answer"]),
            "rouge_l":  rouge_l(pred, item["gold_answer"]),
            "cit_p": p, "cit_r": r, "cit_f1": f,
            "refusal": is_refusal(pred),
            "latency_s": dt,
            "pred_answer": pred[:500],
        })
        log.info("  [REWRITE]    %s R@10=%.2f MRR=%.2f F1=%.3f Cit-R=%.2f (%.1fs)",
                 item["id"], rows[-1]["r_at_10"], rows[-1]["mrr"],
                 rows[-1]["f1_token"], r, dt)
        time.sleep(SLEEP_S)
    return rows


def main():
    gold = load_gold()
    log.info("Loaded %d gold questions", len(gold))

    log.info("Loading retriever + generator")
    retriever = TrafficHybridRetriever()
    generator = LegalAnswerGenerator(provider="google", model=GEN_MODEL)

    log.info("Loading analyzer LLM (%s)", ANALYZER_MODEL)
    analyzer_llm = ChatGoogleGenerativeAI(model=ANALYZER_MODEL, temperature=0.0)
    analyzer_node = make_analyzer_node(analyzer_llm)

    all_rows = []
    log.info("=== NO_REWRITE ===")
    all_rows.extend(run_no_rewrite(gold, retriever, generator))
    log.info("=== REWRITE ===")
    all_rows.extend(run_rewrite(gold, retriever, generator, analyzer_node))

    df = pd.DataFrame(all_rows)
    per_q = METRICS_DIR / "rq9_rewrite_per_question.csv"
    df.to_csv(per_q, index=False)
    log.info("Wrote %s", per_q)

    summary = df.groupby("variant").agg({
        "r_at_5": "mean", "r_at_10": "mean", "mrr": "mean", "ndcg_10": "mean",
        "f1_token": "mean", "rouge_l": "mean",
        "cit_p": "mean", "cit_r": "mean", "cit_f1": "mean",
        "refusal": "mean", "latency_s": "mean",
    }).round(4)
    summary_path = METRICS_DIR / "rq9_rewrite_summary.csv"
    summary.to_csv(summary_path)
    log.info("Wrote %s\n%s", summary_path, summary.to_string())
    log.info("DONE.")


if __name__ == "__main__":
    main()
