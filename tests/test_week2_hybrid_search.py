from src.models import Chunk, SearchQuery
from src.db.connection import InMemoryDualStore
from src.pipeline.hybrid_search import HybridSearchEngine
from src.pipeline.reranker import CrossEncoderReranker


def test_dual_storage_dense_and_sparse():
    storage = InMemoryDualStore()

    chunk1 = Chunk(
        id="c1",
        document_id="d1",
        text="Cơ sở dữ liệu PostgreSQL sử dụng extension pgvector với index HNSW.",
        embedding=[0.1] * 1024,
    )
    chunk2 = Chunk(
        id="c2",
        document_id="d2",
        text="Full-Text Search trong Postgres dùng cột tsvector và index GIN để tìm kiếm từ khóa BM25.",
        embedding=[0.9] * 1024,
    )

    storage.upsert_chunks([chunk1, chunk2])
    assert storage.count() == 2

    # Test Dense search
    dense_hits = storage.dense_search([0.9] * 1024, top_k=1)
    assert len(dense_hits) == 1
    assert dense_hits[0][0].id == "c2"

    # Test Sparse search
    sparse_hits = storage.sparse_search("HNSW extension", top_k=1)
    assert len(sparse_hits) == 1
    assert sparse_hits[0][0].id == "c1"


def test_hybrid_search_with_rrf_and_reranker():
    storage = InMemoryDualStore()
    engine = HybridSearchEngine(storage=storage)

    chunk_a = Chunk(
        id="chunk_a",
        document_id="doc_a",
        text="Thuật toán Reciprocal Rank Fusion RRF kết hợp thứ hạng từ Dense và BM25.",
        embedding=[0.5] * 1024,
    )
    chunk_b = Chunk(
        id="chunk_b",
        document_id="doc_b",
        text="Cross-Encoder Reranker dùng mô hình bge-reranker-large chấm điểm sâu.",
        embedding=[0.2] * 1024,
    )

    storage.upsert_chunks([chunk_a, chunk_b])

    req = SearchQuery(query="Reciprocal Rank Fusion RRF", top_k=2, use_reranker=True)
    resp = engine.search(req)

    assert resp.total_candidates > 0
    assert len(resp.results) > 0
    # Top chunk should be chunk_a because query matches perfectly
    assert resp.results[0].id == "chunk_a"
    assert resp.results[0].rrf_score is not None
    assert resp.results[0].final_score > 0
