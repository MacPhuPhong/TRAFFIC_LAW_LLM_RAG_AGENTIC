"""
Index Data/all_chunks.jsonl into Supabase Postgres (pgvector).
Run once after applying pgvector_schema.sql.

Usage:
  export VECTOR_DB_URL='postgresql://...pooler.supabase.com:5432/postgres'
  python -m deploy.scripts.index_pgvector --recreate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[3]
JSONL_PATH = REPO_ROOT / "traffic_rag" / "Data" / "all_chunks.jsonl"

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
BATCH_EMBED = 64
BATCH_INSERT = 100
VECTOR_DIM = 768
PASSAGE_PREFIX = "passage: "

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("index_pgvector")


def _to_int(x, default=0):
    try:
        return int(x) if x not in (None, "") else default
    except (TypeError, ValueError):
        return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true",
                        help="TRUNCATE table before insert")
    parser.add_argument("--jsonl", default=str(JSONL_PATH))
    args = parser.parse_args()

    url = os.getenv("VECTOR_DB_URL")
    if not url:
        sys.exit("Set VECTOR_DB_URL env var (Supabase Session Pooler, port 5432).")

    log.info("Loading embedding model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    assert model.get_sentence_embedding_dimension() == VECTOR_DIM

    log.info("Reading chunks: %s", args.jsonl)
    chunks = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    log.info("Loaded %d chunks", len(chunks))

    with psycopg.connect(url, autocommit=False) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            if args.recreate:
                log.info("TRUNCATE traffic_chunks")
                cur.execute("TRUNCATE traffic_chunks")
                conn.commit()

            t0 = time.time()
            embedded = 0
            buf_rows = []

            for i in range(0, len(chunks), BATCH_EMBED):
                batch = chunks[i:i + BATCH_EMBED]
                texts = [PASSAGE_PREFIX + (c.get("content") or "") for c in batch]
                vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

                for j, c in enumerate(batch):
                    idx = i + j
                    meta = c.get("metadata") or {}
                    buf_rows.append((
                        idx,
                        vecs[j].tolist(),
                        meta.get("chunk_id", ""),
                        meta.get("doc_id", ""),
                        meta.get("ten_van_ban", ""),
                        meta.get("issuer", ""),
                        meta.get("document_type", ""),
                        meta.get("status", "active") or "active",
                        meta.get("effective_date", "") or "",
                        meta.get("topic", "") or "",
                        _to_int(meta.get("dieu")),
                        meta.get("dieu_title", "") or "",
                        str(meta.get("khoan")) if meta.get("khoan") not in (None, "") else None,
                        str(meta.get("diem")) if meta.get("diem") not in (None, "") else None,
                        _to_int(meta.get("level"), 1),
                        meta.get("source_file", "") or "",
                        _to_int(meta.get("token_estimate")),
                        c.get("content", ""),
                    ))

                while len(buf_rows) >= BATCH_INSERT:
                    flush = buf_rows[:BATCH_INSERT]
                    buf_rows = buf_rows[BATCH_INSERT:]
                    cur.executemany(
                        """INSERT INTO traffic_chunks
                           (id, embedding, chunk_id, doc_id, ten_van_ban, issuer,
                            document_type, status, effective_date, topic, dieu,
                            dieu_title, khoan, diem, level, source_file,
                            token_estimate, content)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                   %s, %s, %s, %s, %s, %s, %s)""",
                        flush,
                    )
                    conn.commit()
                    embedded += len(flush)
                    rate = embedded / (time.time() - t0)
                    log.info("Inserted %d/%d  (%.1f chunks/s)", embedded, len(chunks), rate)

            if buf_rows:
                cur.executemany(
                    """INSERT INTO traffic_chunks
                       (id, embedding, chunk_id, doc_id, ten_van_ban, issuer,
                        document_type, status, effective_date, topic, dieu,
                        dieu_title, khoan, diem, level, source_file,
                        token_estimate, content)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                               %s, %s, %s, %s, %s, %s, %s)""",
                    buf_rows,
                )
                conn.commit()
                embedded += len(buf_rows)

            log.info("Done: %d rows inserted in %.1fs", embedded, time.time() - t0)

            cur.execute("ANALYZE traffic_chunks")
            cur.execute("SELECT COUNT(*) FROM traffic_chunks")
            log.info("Total rows now: %s", cur.fetchone()[0])


if __name__ == "__main__":
    main()
