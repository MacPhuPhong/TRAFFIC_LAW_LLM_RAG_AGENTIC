# -*- coding: utf-8 -*-
"""RQ3 — Embedding ablation: sbert vs mpnet vs e5-small.

Pure dense-retrieval benchmark (no BM25, no filters). For each model:
  1. Encode full hierarchical corpus (`Data/all_chunks.jsonl`, 2705 chunks).
  2. Encode 25 eval queries (with model-specific prefix for e5).
  3. Cosine-rank → top-10 → score Recall@5, Recall@10, MRR, nDCG@10 at khoan level.
  4. Record encode_corpus_s and query_time_ms.

Writes rq3_embedding.csv (overwrites) + per-question CSV.
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from research.utils.metrics import mrr, ndcg_at_k, recall_at_k  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("rq3")

GOLD = ROOT / "research" / "data" / "eval_qa.jsonl"
CORPUS = ROOT / "Data" / "all_chunks.jsonl"
METRICS_DIR = ROOT / "research" / "results" / "metrics"

MODELS = [
    {
        "name": "vietnamese-sbert",
        "hf_id": "KeepItReal/vietnamese-sbert",
        "query_prefix": "",
        "passage_prefix": "",
    },
    {
        "name": "multilingual-mpnet",
        "hf_id": "paraphrase-multilingual-mpnet-base-v2",
        "query_prefix": "",
        "passage_prefix": "",
    },
    {
        "name": "multilingual-e5-small",
        "hf_id": "intfloat/multilingual-e5-small",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
]


def load_gold() -> list[dict]:
    return [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]


def load_corpus() -> list[dict]:
    return [json.loads(l) for l in open(CORPUS, encoding="utf-8") if l.strip()]


def bench_model(spec: dict, corpus: list[dict], gold: list[dict]) -> dict:
    name = spec["name"]
    log.info("=== %s ===", name)
    model = SentenceTransformer(spec["hf_id"])
    max_seq = model.max_seq_length
    dim = model.get_sentence_embedding_dimension()
    log.info("  max_seq=%d  dim=%d", max_seq, dim)

    passages = [spec["passage_prefix"] + c["content"] for c in corpus]
    t0 = time.perf_counter()
    corpus_vecs = model.encode(
        passages, batch_size=64, normalize_embeddings=True,
        show_progress_bar=False, convert_to_numpy=True,
    )
    encode_corpus_s = time.perf_counter() - t0
    log.info("  corpus encode: %.1fs", encode_corpus_s)

    queries = [spec["query_prefix"] + q["question"] for q in gold]
    t0 = time.perf_counter()
    q_vecs = model.encode(queries, normalize_embeddings=True,
                          show_progress_bar=False, convert_to_numpy=True)
    query_total_s = time.perf_counter() - t0
    query_ms = 1000 * query_total_s / len(gold)

    sims = q_vecs @ corpus_vecs.T  # cosine (already normalized)
    top20_idx = np.argsort(-sims, axis=1)[:, :20]

    per_q = []
    for qi, item in enumerate(gold):
        top_cites = []
        for ci in top20_idx[qi]:
            m = corpus[int(ci)]["metadata"]
            top_cites.append({
                "doc_id": m.get("doc_id"),
                "dieu": m.get("dieu"),
                "khoan": m.get("khoan"),
                "diem": m.get("diem"),
            })
        g = item["gold_citations"]
        per_q.append({
            "model": name,
            "qid": item["id"],
            "category": item["category"],
            "recall@5": recall_at_k(top_cites, g, 5, "khoan"),
            "recall@10": recall_at_k(top_cites, g, 10, "khoan"),
            "recall@20": recall_at_k(top_cites, g, 20, "khoan"),
            "mrr": mrr(top_cites, g, "khoan"),
            "ndcg@10": ndcg_at_k(top_cites, g, 10, "khoan"),
        })
    df = pd.DataFrame(per_q)
    summary = {
        "model": name,
        "hf_id": spec["hf_id"],
        "max_seq_length": max_seq,
        "dim": dim,
        "encode_corpus_s": round(encode_corpus_s, 2),
        "query_encode_ms": round(query_ms, 2),
        "recall@5": round(df["recall@5"].mean(), 4),
        "recall@10": round(df["recall@10"].mean(), 4),
        "recall@20": round(df["recall@20"].mean(), 4),
        "mrr": round(df["mrr"].mean(), 4),
        "ndcg@10": round(df["ndcg@10"].mean(), 4),
    }
    del model, corpus_vecs
    return summary, per_q


def main():
    gold = load_gold()
    corpus = load_corpus()
    log.info("Corpus: %d chunks | Gold: %d queries", len(corpus), len(gold))

    summaries, per_q_all = [], []
    for spec in MODELS:
        try:
            s, pq = bench_model(spec, corpus, gold)
            summaries.append(s)
            per_q_all.extend(pq)
        except Exception as e:
            log.error("Model %s failed: %s", spec["name"], e)

    df_sum = pd.DataFrame(summaries)
    df_sum.to_csv(METRICS_DIR / "rq3_embedding.csv", index=False)
    pd.DataFrame(per_q_all).to_csv(METRICS_DIR / "rq3_embedding_per_question.csv", index=False)
    log.info("\n%s", df_sum.to_string())


if __name__ == "__main__":
    main()
