-- Schema for Traffic RAG vector store on Supabase Postgres (pgvector).
-- Replaces Qdrant Cloud as the dense vector backend.
-- Run once before indexing.

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS traffic_chunks CASCADE;

CREATE TABLE traffic_chunks (
  id              INTEGER PRIMARY KEY,
  embedding       vector(768),

  -- Mirrors the Qdrant payload schema (built by indexer.py)
  chunk_id        TEXT,
  doc_id          TEXT,
  ten_van_ban     TEXT,
  issuer          TEXT,
  document_type   TEXT,
  status          TEXT,
  effective_date  TEXT,
  topic           TEXT,
  dieu            INTEGER,
  dieu_title      TEXT,
  khoan           TEXT,
  diem            TEXT,
  level           INTEGER,
  source_file     TEXT,
  token_estimate  INTEGER,
  content         TEXT
);

CREATE INDEX traffic_chunks_embedding_hnsw_idx
  ON traffic_chunks
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

CREATE INDEX traffic_chunks_status_idx ON traffic_chunks (status);
CREATE INDEX traffic_chunks_doc_id_idx ON traffic_chunks (doc_id);
CREATE INDEX traffic_chunks_topic_idx  ON traffic_chunks (topic);

ANALYZE traffic_chunks;
