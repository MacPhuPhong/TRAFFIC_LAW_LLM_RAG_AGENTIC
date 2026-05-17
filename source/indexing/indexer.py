# -*- coding: utf-8 -*-
"""
indexer.py — Phase 2: Qdrant Vector Indexing Pipeline
=====================================================
TASK 1: Collection config (768-dim, Cosine, payload indexes)
TASK 2: Batch indexing from all_chunks.jsonl
TASK 3: Hybrid search with metadata filtering
TASK 4: Validation queries (3 test cases)
"""

import json
import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct,
    PayloadSchemaType,
)
from sentence_transformers import SentenceTransformer

# Canonical retriever lives in rag_core; indexer just reuses it for validation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag_core.retriever import TrafficHybridRetriever  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
import os
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_URL = os.getenv("QDRANT_URL") or None
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "Traffic_Law_Hybrid")
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
VECTOR_SIZE = 768
BATCH_EMBED = 64       # Encode 64 texts at a time
BATCH_UPSERT = 100     # Push 100 points at a time

# e5 family requires "passage: " / "query: " prefix at encode time.
PASSAGE_PREFIX = "passage: " if "e5" in EMBEDDING_MODEL.lower() else ""

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
JSONL_PATH = BASE_DIR / "Data" / "all_chunks.jsonl"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / f"indexer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ===================================================================
# TASK 1: CREATE / RESET QDRANT COLLECTION
# ===================================================================
def create_collection(client: QdrantClient, recreate: bool = False):
    """
    Tạo collection với:
      - Vector 384-dim (multilingual-e5-small), Cosine distance
      - Payload indexes: doc_id, status, effective_date, topic
    """
    collections = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in collections:
        if recreate:
            logger.info(f"Xóa collection cũ: {COLLECTION_NAME}")
            client.delete_collection(COLLECTION_NAME)
        else:
            logger.info(f"Collection '{COLLECTION_NAME}' đã tồn tại. Dùng --recreate để tạo lại.")
            return

    logger.info(f"Tạo collection '{COLLECTION_NAME}' (dim={VECTOR_SIZE}, Cosine)")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    # Tạo payload indexes cho Metadata Filtering
    for field, schema in [
        ("doc_id",         PayloadSchemaType.KEYWORD),
        ("status",         PayloadSchemaType.KEYWORD),
        ("effective_date", PayloadSchemaType.KEYWORD),
        ("topic",          PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=schema,
        )
        logger.info(f"  ✓ Payload index: {field} ({schema})")

    logger.info("Collection đã sẵn sàng.\n")


# ===================================================================
# TASK 2: INDEX ALL CHUNKS
# ===================================================================
def index_chunks(client: QdrantClient, model: SentenceTransformer):
    """
    Đọc all_chunks.jsonl → embed → upsert vào Qdrant theo batch.
    Log tiến độ theo từng source_file.
    """
    logger.info(f"Đọc dữ liệu từ: {JSONL_PATH}")

    chunks = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    total = len(chunks)
    logger.info(f"Tổng chunks: {total}")

    # ── Embed theo batch ──
    logger.info(f"Bắt đầu embedding ({EMBEDDING_MODEL}, batch={BATCH_EMBED})...")
    texts = [PASSAGE_PREFIX + c["content"] for c in chunks]
    all_vectors = []

    t0 = time.time()
    for i in range(0, total, BATCH_EMBED):
        batch_texts = texts[i : i + BATCH_EMBED]
        vectors = model.encode(batch_texts, normalize_embeddings=True)
        all_vectors.extend(vectors.tolist())

        done = min(i + BATCH_EMBED, total)
        elapsed = time.time() - t0
        speed = done / elapsed if elapsed > 0 else 0
        logger.info(f"  Embedded {done}/{total}  ({speed:.1f} chunks/s)")

    embed_time = time.time() - t0
    logger.info(f"Embedding hoàn tất: {embed_time:.1f}s\n")

    # ── Upsert vào Qdrant theo batch ──
    logger.info(f"Upsert vào Qdrant (batch={BATCH_UPSERT})...")
    current_source = None
    source_count = 0

    t1 = time.time()
    for i in range(0, total, BATCH_UPSERT):
        batch_points = []
        for j in range(i, min(i + BATCH_UPSERT, total)):
            chunk = chunks[j]
            meta = chunk["metadata"]

            # Log khi chuyển sang source file mới
            src = meta.get("source_file", "")
            if src != current_source:
                if current_source is not None:
                    logger.info(f"    [{current_source}] → {source_count} points")
                current_source = src
                source_count = 0
            source_count += 1

            batch_points.append(
                PointStruct(
                    id=j,
                    vector=all_vectors[j],
                    payload={
                        # Metadata cho filtering
                        "chunk_id":       meta.get("chunk_id", ""),
                        "doc_id":         meta.get("doc_id", ""),
                        "ten_van_ban":    meta.get("ten_van_ban", ""),
                        "issuer":         meta.get("issuer", ""),
                        "document_type":  meta.get("document_type", ""),
                        "status":         meta.get("status", "active"),
                        "effective_date": meta.get("effective_date", ""),
                        "topic":          meta.get("topic", ""),
                        "dieu":           meta.get("dieu", 0),
                        "dieu_title":     meta.get("dieu_title", ""),
                        "khoan":          meta.get("khoan"),
                        "diem":           meta.get("diem"),
                        "level":          meta.get("level", 1),
                        "source_file":    meta.get("source_file", ""),
                        "token_estimate": meta.get("token_estimate", 0),
                        # Content cho display
                        "content":        chunk["content"],
                    },
                )
            )

        client.upsert(collection_name=COLLECTION_NAME, points=batch_points)

    # Log file cuối cùng
    if current_source:
        logger.info(f"    [{current_source}] → {source_count} points")

    upsert_time = time.time() - t1
    logger.info(f"\nUpsert hoàn tất: {upsert_time:.1f}s")

    # Verify count
    info = client.get_collection(COLLECTION_NAME)
    logger.info(f"Points trong Qdrant: {info.points_count}")
    logger.info(f"Tổng thời gian: {embed_time + upsert_time:.1f}s\n")


# ===================================================================
# TASK 3: VALIDATION QUERIES (delegates retrieval to TrafficHybridRetriever)
# ===================================================================
def run_validation(retriever: TrafficHybridRetriever):
    """Chạy 3 truy vấn kiểm định chất lượng tìm kiếm."""

    queries = [
        {
            "query": "Mức trừ điểm GPLX cho hành vi vượt đèn đỏ theo quy định mới 2024?",
            "expected_doc": "168/2024/NĐ-CP",
            "description": "Xử phạt — NĐ 168/2024",
        },
        {
            "query": "Chu kỳ đăng kiểm lần đầu cho xe ô tô con không kinh doanh vận tải sản xuất năm 2025?",
            "expected_doc": "47/2024/TT-BGTVT",
            "description": "Kỹ thuật — TT 47/2024",
        },
        {
            "query": "Hồ sơ đăng ký xe trực tuyến cần những giấy tờ gì?",
            "expected_doc": "79/2024/TT-BCA",
            "description": "Thủ tục — TT 79/2024",
        },
    ]

    logger.info("=" * 60)
    logger.info("VALIDATION QUERIES")
    logger.info("=" * 60)

    total_pass = 0
    for i, q in enumerate(queries, 1):
        results = retriever.get_relevant_chunks(q["query"], top_k=5)

        found_docs = [r.metadata.get("doc_id", "") for r in results]
        passed = q["expected_doc"] in found_docs
        if passed:
            total_pass += 1

        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"\nQuery {i}: {q['description']}")
        logger.info(f"  Q: \"{q['query']}\"")
        logger.info(f"  Expected: {q['expected_doc']}")
        logger.info(f"  Result: {status}")
        logger.info(f"  Top 5 docs: {found_docs}")

        if results:
            top = results[0]
            preview = top.content[:200]
            logger.info(f"  Top 1 (score={top.score:.4f}): {preview}...")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"VALIDATION: {total_pass}/{len(queries)} passed")
    logger.info(f"{'=' * 60}\n")


# ===================================================================
# MAIN
# ===================================================================
def main():
    logger.info("=" * 60)
    logger.info("QDRANT INDEXING PIPELINE v1.0")
    logger.info("=" * 60)

    # 1. Connect to Qdrant (local or cloud)
    if QDRANT_URL:
        logger.info(f"Kết nối Qdrant Cloud: {QDRANT_URL}")
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        logger.info(f"Kết nối Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)

    try:
        client.get_collections()
        logger.info("Kết nối Qdrant thành công!\n")
    except Exception as e:
        logger.error(f"Không thể kết nối Qdrant: {e}")
        logger.error("Local: docker compose up -d qdrant   |   Cloud: kiểm tra QDRANT_URL/QDRANT_API_KEY")
        sys.exit(1)

    # 2. Load embedding model
    logger.info(f"Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info(f"Model loaded. Dimension: {model.get_sentence_embedding_dimension()}\n")

    # 3. Create collection
    recreate = "--recreate" in sys.argv
    create_collection(client, recreate=recreate)

    # 4. Index chunks
    if "--validate-only" not in sys.argv:
        index_chunks(client, model)

    # 5. Validation queries via canonical hybrid retriever (rag_core)
    retriever = TrafficHybridRetriever(
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        jsonl_path=JSONL_PATH,
    )
    run_validation(retriever)

    logger.info("PIPELINE HOÀN TẤT.")


if __name__ == "__main__":
    main()
