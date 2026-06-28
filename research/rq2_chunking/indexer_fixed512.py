# -*- coding: utf-8 -*-
"""Indexer for the fixed-size chunk ablation (RQ2).

Loads research/rq2_chunking/data/all_chunks_fixed512.jsonl, embeds with `intfloat/multilingual-e5-small`
(max_seq=512 — no truncation on 512-token chunks), upserts to collection
`traffic_law_fixed512`. Deliberately disables BM25 when loaded via the
retriever; we only benchmark dense retrieval for the ablation.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent.parent

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "traffic_law_fixed512"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
VECTOR_SIZE = 384  # e5-small dim
BATCH_EMBED = 64
BATCH_UPSERT = 100

JSONL_PATH = Path(__file__).resolve().parent / "data" / "all_chunks_fixed512.jsonl"

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / f"indexer_fixed512_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def create_collection(client: QdrantClient, recreate: bool = False):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        if recreate:
            logger.info("Dropping existing collection: %s", COLLECTION_NAME)
            client.delete_collection(COLLECTION_NAME)
        else:
            logger.info("Collection '%s' exists; pass --recreate to drop.", COLLECTION_NAME)
            return
    logger.info("Creating collection '%s' (dim=%d, Cosine)", COLLECTION_NAME, VECTOR_SIZE)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    for field, schema in [
        ("doc_id", PayloadSchemaType.KEYWORD),
        ("status", PayloadSchemaType.KEYWORD),
        ("topic", PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=schema,
        )


def index_chunks(client: QdrantClient, model: SentenceTransformer):
    chunks = []
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    total = len(chunks)
    logger.info("Loaded %d chunks from %s", total, JSONL_PATH)

    # e5 models want "passage: " prefix during encoding per model card.
    texts = ["passage: " + c["content"] for c in chunks]

    t0 = time.time()
    all_vectors = []
    for i in range(0, total, BATCH_EMBED):
        v = model.encode(
            texts[i : i + BATCH_EMBED],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_vectors.extend(v.tolist())
        logger.info("  embedded %d/%d (%.1f chunks/s)",
                    min(i + BATCH_EMBED, total), total,
                    min(i + BATCH_EMBED, total) / max(time.time() - t0, 1e-6))
    embed_time = time.time() - t0
    logger.info("Embedding: %.1fs", embed_time)

    t1 = time.time()
    for i in range(0, total, BATCH_UPSERT):
        batch = []
        for j in range(i, min(i + BATCH_UPSERT, total)):
            meta = chunks[j]["metadata"]
            batch.append(
                PointStruct(
                    id=j,
                    vector=all_vectors[j],
                    payload={
                        "chunk_id": meta.get("chunk_id", ""),
                        "doc_id": meta.get("doc_id", ""),
                        "ten_van_ban": meta.get("ten_van_ban", ""),
                        "issuer": meta.get("issuer", ""),
                        "document_type": meta.get("document_type", ""),
                        "status": meta.get("status", "active"),
                        "effective_date": meta.get("effective_date", ""),
                        "topic": meta.get("topic", ""),
                        "dieu": meta.get("dieu", 0),
                        "khoan": meta.get("khoan"),
                        "diem": meta.get("diem"),
                        "level": meta.get("level", 0),
                        "source_file": meta.get("source_file", ""),
                        "token_estimate": meta.get("token_estimate", 0),
                        "chunk_strategy": meta.get("chunk_strategy", ""),
                        "content": chunks[j]["content"],
                    },
                )
            )
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
    logger.info("Upsert: %.1fs", time.time() - t1)

    info = client.get_collection(COLLECTION_NAME)
    logger.info("Points in Qdrant: %d", info.points_count)


def main():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    try:
        client.get_collections()
    except Exception as e:
        logger.error("Qdrant unreachable: %s. Run: docker compose up -d qdrant", e)
        sys.exit(1)

    logger.info("Loading %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("  max_seq_length=%d, dim=%d",
                model.max_seq_length,
                model.get_sentence_embedding_dimension())

    create_collection(client, recreate="--recreate" in sys.argv)
    if "--validate-only" not in sys.argv:
        index_chunks(client, model)
    logger.info("DONE.")


if __name__ == "__main__":
    main()
