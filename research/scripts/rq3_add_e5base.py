# -*- coding: utf-8 -*-
"""Append e5-base results to rq3_embedding.csv without re-running existing models."""

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

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from research.utils.metrics import mrr, ndcg_at_k, recall_at_k

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rq3_e5base")

GOLD   = ROOT / "research" / "data" / "eval_qa.jsonl"
CORPUS = ROOT / "Data" / "all_chunks.jsonl"
METRICS_DIR = ROOT / "research" / "results" / "metrics"

SPEC = {
    "name": "multilingual-e5-base",
    "hf_id": "intfloat/multilingual-e5-base",
    "query_prefix": "query: ",
    "passage_prefix": "passage: ",
}


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    gold   = load_jsonl(GOLD)
    corpus = load_jsonl(CORPUS)
    log.info("Corpus: %d chunks | Gold: %d queries", len(corpus), len(gold))

    spec = SPEC
    log.info("=== %s ===", spec["name"])
    model = SentenceTransformer(spec["hf_id"])
    max_seq = model.max_seq_length
    dim     = model.get_sentence_embedding_dimension()
    log.info("  max_seq=%d  dim=%d", max_seq, dim)

    passages = [spec["passage_prefix"] + c["content"] for c in corpus]
    t0 = time.perf_counter()
    corpus_vecs = model.encode(passages, batch_size=64, normalize_embeddings=True,
                               show_progress_bar=True, convert_to_numpy=True)
    encode_corpus_s = time.perf_counter() - t0
    log.info("  corpus encode: %.1fs", encode_corpus_s)

    queries = [spec["query_prefix"] + q["question"] for q in gold]
    t0 = time.perf_counter()
    q_vecs = model.encode(queries, normalize_embeddings=True,
                          show_progress_bar=False, convert_to_numpy=True)
    query_ms = 1000 * (time.perf_counter() - t0) / len(gold)

    sims = q_vecs @ corpus_vecs.T
    top20_idx = np.argsort(-sims, axis=1)[:, :20]

    per_q = []
    for qi, item in enumerate(gold):
        top_cites = []
        for ci in top20_idx[qi]:
            m = corpus[int(ci)]["metadata"]
            top_cites.append({"doc_id": m.get("doc_id"), "dieu": m.get("dieu"),
                               "khoan": m.get("khoan"), "diem": m.get("diem")})
        g = item["gold_citations"]
        per_q.append({
            "model": spec["name"], "qid": item["id"], "category": item["category"],
            "recall@5":  recall_at_k(top_cites, g, 5,  "khoan"),
            "recall@10": recall_at_k(top_cites, g, 10, "khoan"),
            "recall@20": recall_at_k(top_cites, g, 20, "khoan"),
            "mrr":       mrr(top_cites, g, "khoan"),
            "ndcg@10":   ndcg_at_k(top_cites, g, 10, "khoan"),
        })

    df_pq = pd.DataFrame(per_q)
    summary = {
        "model": spec["name"], "hf_id": spec["hf_id"],
        "max_seq_length": max_seq, "dim": dim,
        "encode_corpus_s": round(encode_corpus_s, 2),
        "query_encode_ms": round(query_ms, 2),
        "recall@5":  round(df_pq["recall@5"].mean(),  4),
        "recall@10": round(df_pq["recall@10"].mean(), 4),
        "recall@20": round(df_pq["recall@20"].mean(), 4),
        "mrr":       round(df_pq["mrr"].mean(),       4),
        "ndcg@10":   round(df_pq["ndcg@10"].mean(),   4),
    }
    log.info("Summary: %s", summary)

    # Append to existing CSVs
    csv_sum = METRICS_DIR / "rq3_embedding.csv"
    csv_pq  = METRICS_DIR / "rq3_embedding_per_question.csv"

    df_existing = pd.read_csv(csv_sum)
    df_existing = df_existing[df_existing["model"] != spec["name"]]  # remove if re-run
    pd.concat([df_existing, pd.DataFrame([summary])], ignore_index=True).to_csv(csv_sum, index=False)

    df_pq_existing = pd.read_csv(csv_pq)
    df_pq_existing = df_pq_existing[df_pq_existing["model"] != spec["name"]]
    pd.concat([df_pq_existing, df_pq], ignore_index=True).to_csv(csv_pq, index=False)

    log.info("Done — results appended to %s", csv_sum)
    print("\n=== FINAL TABLE ===")
    print(pd.read_csv(csv_sum).to_string(index=False))


if __name__ == "__main__":
    main()
