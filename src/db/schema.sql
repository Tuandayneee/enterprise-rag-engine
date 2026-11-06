-- Kích hoạt extension pgvector và pg_trgm cho PostgreSQL
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Bảng lưu trữ chunks tài liệu với cột vector và cột tsvector cho Full-text search
CREATE TABLE IF NOT EXISTS document_chunks (
    id VARCHAR(128) PRIMARY KEY,
    document_id VARCHAR(128) NOT NULL,
    strategy VARCHAR(32) DEFAULT 'semantic',
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding vector(1024),
    tsv_content tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index HNSW (Hierarchical Navigable Small World) tối ưu hóa truy vấn Dense Cosine Similarity
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw 
ON document_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Index GIN cho Full-text Search (Sparse BM25 Keyword Matching)
CREATE INDEX IF NOT EXISTS idx_chunks_tsv 
ON document_chunks 
USING gin (tsv_content);

-- Index Document ID phục vụ truy vấn theo tài liệu
CREATE INDEX IF NOT EXISTS idx_chunks_document_id 
ON document_chunks (document_id);
