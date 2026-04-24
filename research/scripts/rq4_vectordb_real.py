# -*- coding: utf-8 -*-
"""RQ4 — Vector DB benchmark on REAL corpus (Qdrant vs Chroma vs FAISS).

Protocol (no BM25, pure dense):
  * Pull 2705 real embedding vectors + payloads from the production
    `traffic_law_v2` Qdrant collection (sbert-768d).
  * Build three indices from the same vectors:
      - Qdrant (already built — we just query it).
      - Chroma persistent (new collection in Data/chroma_bench).
      - FAISS IVF-Flat (in-memory).
  * For each of 25 gold queries, encode once (sbert), run 100 lookups
    per backend, measure p50/p95 latency, Recall@10 (khoan level).
  * Track peak RSS via psutil per phase.

Writes rq4_vectordb_real.csv.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from research.utils.metrics import recall_at_k  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("rq4")

GOLD = ROOT / "research" / "data" / "eval_qa.jsonl"
METRICS_DIR = ROOT / "research" / "results" / "metrics"
CHROMA_DIR = ROOT / "Data" / "chroma_bench"

QDRANT_COLLECTION = "traffic_law_v2"
EMBEDDING_MODEL = "KeepItReal/vietnamese-sbert"
N_REPEATS = 100  # per gold query
TOP_K = 10


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


def pull_vectors_from_qdrant(client: QdrantClient, collection: str):
    log.info("Pulling vectors + payloads from Qdrant '%s'", collection)
    info = client.get_collection(collection)
    n = info.points_count
    vectors, payloads, ids = [], [], []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            limit=500,
            with_vectors=True,
            with_payload=True,
            offset=offset,
        )
        for p in batch:
            vectors.append(p.vector)
            payloads.append(p.payload)
            ids.append(p.id)
        if offset is None:
            break
    V = np.array(vectors, dtype=np.float32)
    log.info("  got %d/%d vectors  dim=%d  rss=%.1fMB", len(V), n, V.shape[1], rss_mb())
    return V, payloads, ids


def build_chroma(vectors, payloads, ids, dim):
    import chromadb
    log.info("Building Chroma (persistent) at %s", CHROMA_DIR)
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True)
    t0 = time.perf_counter()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    coll = client.create_collection("rq4_bench", metadata={"hnsw:space": "cosine"})
    b = 500
    for i in range(0, len(vectors), b):
        coll.add(
            ids=[str(x) for x in ids[i:i+b]],
            embeddings=vectors[i:i+b].tolist(),
            metadatas=[{
                "doc_id": str(p.get("doc_id", "")),
                "dieu": int(p.get("dieu") or 0),
                "khoan": str(p.get("khoan") or ""),
                "diem": str(p.get("diem") or ""),
            } for p in payloads[i:i+b]],
        )
    dt = time.perf_counter() - t0
    log.info("  Chroma built in %.1fs  rss=%.1fMB", dt, rss_mb())
    return coll


def build_faiss(vectors, dim):
    import faiss
    log.info("Building FAISS IVF-Flat (nlist=64)")
    t0 = time.perf_counter()
    V = vectors.copy()
    faiss.normalize_L2(V)  # cosine via inner product
    quantizer = faiss.IndexFlatIP(dim)
    nlist = 64
    idx = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    idx.train(V)
    idx.add(V)
    idx.nprobe = 8
    dt = time.perf_counter() - t0
    log.info("  FAISS built in %.1fs  rss=%.1fMB", dt, rss_mb())
    return idx


def bench_qdrant(client, queries_vec):
    times = []
    results = []
    for qv in queries_vec:
        hits_list = None
        t0 = time.perf_counter_ns()
        for _ in range(N_REPEATS):
            hits_list = client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=qv.tolist(),
                limit=TOP_K,
                with_payload=True,
            )
        dt = (time.perf_counter_ns() - t0) / 1e6 / N_REPEATS
        times.append(dt)
        results.append([{
            "doc_id": h.payload.get("doc_id"),
            "dieu": h.payload.get("dieu"),
            "khoan": h.payload.get("khoan"),
            "diem": h.payload.get("diem"),
        } for h in hits_list])
    return times, results


def bench_chroma(coll, queries_vec):
    times, results = [], []
    for qv in queries_vec:
        q = qv.tolist()
        res = None
        t0 = time.perf_counter_ns()
        for _ in range(N_REPEATS):
            res = coll.query(query_embeddings=[q], n_results=TOP_K,
                             include=["metadatas"])
        dt = (time.perf_counter_ns() - t0) / 1e6 / N_REPEATS
        times.append(dt)
        metas = res["metadatas"][0]
        results.append([{
            "doc_id": m.get("doc_id"),
            "dieu": m.get("dieu"),
            "khoan": m.get("khoan") or None,
            "diem": m.get("diem") or None,
        } for m in metas])
    return times, results


def bench_faiss(idx, payloads, queries_vec):
    import faiss
    times, results = [], []
    for qv in queries_vec:
        Q = qv.reshape(1, -1).astype(np.float32).copy()
        faiss.normalize_L2(Q)
        I = None
        t0 = time.perf_counter_ns()
        for _ in range(N_REPEATS):
            _, I = idx.search(Q, TOP_K)
        dt = (time.perf_counter_ns() - t0) / 1e6 / N_REPEATS
        times.append(dt)
        row = []
        for ci in I[0]:
            p = payloads[int(ci)]
            row.append({
                "doc_id": p.get("doc_id"),
                "dieu": p.get("dieu"),
                "khoan": p.get("khoan"),
                "diem": p.get("diem"),
            })
        results.append(row)
    return times, results


def score_recall(results, gold):
    return float(np.mean([
        recall_at_k(r, g["gold_citations"], TOP_K, "khoan")
        for r, g in zip(results, gold)
    ]))


def main():
    gold = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]
    rss0 = rss_mb()
    log.info("baseline rss=%.1fMB", rss0)

    client = QdrantClient(host="localhost", port=6334)
    V, payloads, ids = pull_vectors_from_qdrant(client, QDRANT_COLLECTION)
    dim = V.shape[1]

    log.info("Loading sbert for query encoding")
    model = SentenceTransformer(EMBEDDING_MODEL)
    queries_vec = model.encode(
        [g["question"] for g in gold], normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    ).astype(np.float32)
    del model

    rows = []

    # Qdrant
    log.info("=== Qdrant ===")
    peak_q = rss_mb()
    t_q, r_q = bench_qdrant(client, queries_vec)
    peak_q = max(peak_q, rss_mb())
    rows.append({
        "backend": "Qdrant (server, hnsw)",
        "n_points": len(V), "dim": dim,
        "p50_ms": float(np.percentile(t_q, 50)),
        "p95_ms": float(np.percentile(t_q, 95)),
        "mean_ms": float(np.mean(t_q)),
        "recall@10": round(score_recall(r_q, gold), 4),
        "peak_rss_mb": round(peak_q, 1),
        "notes": "server process, separate",
    })

    # Chroma
    log.info("=== Chroma ===")
    coll = build_chroma(V, payloads, ids, dim)
    peak_c = rss_mb()
    t_c, r_c = bench_chroma(coll, queries_vec)
    peak_c = max(peak_c, rss_mb())
    rows.append({
        "backend": "Chroma (persistent, hnsw)",
        "n_points": len(V), "dim": dim,
        "p50_ms": float(np.percentile(t_c, 50)),
        "p95_ms": float(np.percentile(t_c, 95)),
        "mean_ms": float(np.mean(t_c)),
        "recall@10": round(score_recall(r_c, gold), 4),
        "peak_rss_mb": round(peak_c, 1),
        "notes": "in-process",
    })

    # FAISS
    log.info("=== FAISS ===")
    idx = build_faiss(V, dim)
    peak_f = rss_mb()
    t_f, r_f = bench_faiss(idx, payloads, queries_vec)
    peak_f = max(peak_f, rss_mb())
    rows.append({
        "backend": "FAISS (IVF-Flat nlist=64, nprobe=8)",
        "n_points": len(V), "dim": dim,
        "p50_ms": float(np.percentile(t_f, 50)),
        "p95_ms": float(np.percentile(t_f, 95)),
        "mean_ms": float(np.mean(t_f)),
        "recall@10": round(score_recall(r_f, gold), 4),
        "peak_rss_mb": round(peak_f, 1),
        "notes": "in-memory, approximate",
    })

    df = pd.DataFrame(rows).round({"p50_ms": 2, "p95_ms": 2, "mean_ms": 2})
    df.to_csv(METRICS_DIR / "rq4_vectordb_real.csv", index=False)
    log.info("\n%s", df.to_string())
    log.info("DONE.")


if __name__ == "__main__":
    main()
