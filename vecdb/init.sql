-- init.sql
-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS lantern;  -- Optional: 2x faster HNSW

-- Create main table
CREATE TABLE IF NOT EXISTS items (
    id BIGSERIAL PRIMARY KEY,
    embedding VECTOR(1024),  -- Grok-beta dim; change to 1536 for Grok-4
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
-- 1. Lantern HNSW (GPU-friendly, ultra-fast)
CREATE INDEX IF NOT EXISTS items_embedding_hnsw_idx 
ON items USING hnsw (embedding vector_cosine_ops) 
WITH (m = 32, ef_construction = 200);

-- 2. IVF fallback (for large-scale, optional)
CREATE INDEX IF NOT EXISTS items_embedding_ivf_idx 
ON items USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 1000);

-- 3. Metadata GIN index (for fast JSONB filtering)
CREATE INDEX IF NOT EXISTS items_metadata_gin_idx 
ON items USING GIN (metadata);

-- Optional: Vacuum and analyze
VACUUM ANALYZE items;