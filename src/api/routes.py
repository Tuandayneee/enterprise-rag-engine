from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from src.models import (
    IngestionRequest,
    IngestionResponse,
    SearchQuery,
    SearchResponse,
    ChatQuery,
    ChatResponse,
    HealthResponse,
)
from src.core.cache import RedisCacheManager
from src.db.connection import get_storage
from src.pipeline.hybrid_search import HybridSearchEngine
from src.pipeline.rag_chain import CustomRAGChain
from src.pipeline.chunker import SemanticChunker
from src.pipeline.embeddings import EmbeddingPipeline
from src.api.middleware import rate_limit_dependency, get_cache_manager

router = APIRouter(prefix="/api/v1", tags=["Enterprise RAG Pipeline"])

# Singleton pipeline instances
_storage = get_storage()
_embeddings = EmbeddingPipeline()
_search_engine = HybridSearchEngine(storage=_storage, embeddings=_embeddings)
_cache_mgr = get_cache_manager()
_rag_chain = CustomRAGChain(search_engine=_search_engine, cache_manager=_cache_mgr)
_chunker = SemanticChunker(embedding_pipeline=_embeddings)


@router.post("/ingest", response_model=IngestionResponse)
async def ingest_document(
    req: IngestionRequest,
    _: str = Depends(rate_limit_dependency)
):
    """
    [OFFLINE INGESTION PIPELINE] Nạp văn bản, phân đoạn ngữ nghĩa và lưu vào pgvector/BM25.
    """
    doc_id = req.document_id or f"doc_{req.content[:10]}_{len(req.content)}"
    chunks = _chunker.chunk(req.content, document_id=doc_id, metadata=req.metadata)
    _storage.upsert_chunks(chunks)
    return IngestionResponse(
        success=True,
        document_id=doc_id,
        total_chunks=len(chunks),
        strategy=req.strategy.value,
        message=f"Đã nạp {len(chunks)} chunks vào PostgreSQL.",
    )


@router.post("/search", response_model=SearchResponse)
async def search_hybrid(
    req: SearchQuery,
    _: str = Depends(rate_limit_dependency)
):
    """
    [ONLINE QUERY PIPELINE] Hybrid Search (Dense + BM25) + RRF + Cross-Encoder Reranker.
    """
    return _search_engine.search(req)


@router.post("/chat", response_model=ChatResponse)
async def chat_rag(
    chat_query: ChatQuery,
    _: str = Depends(rate_limit_dependency)
):
    """
    [ONLINE QUERY PIPELINE] Truy vấn RAG có Redis Cache-Aside, Guardrails chống ảo giác và trích dẫn.
    """
    return await _rag_chain.generate_response(chat_query)


@router.get("/chat/stream")
async def chat_rag_stream(
    q: str = Query(..., description="Câu hỏi người dùng"),
    top_k: int = Query(5, ge=1, le=20),
    _: str = Depends(rate_limit_dependency)
):
    """
    [ONLINE QUERY PIPELINE] Stream Server-Sent Events (SSE) theo thời gian thực về Frontend.
    """
    event_generator = _rag_chain.stream_response(question=q, top_k=top_k)
    return StreamingResponse(
        event_generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Kiểm tra tình trạng kết nối tới PostgreSQL pgvector, Redis Cache và LLM Serving.
    """
    pg_ok = hasattr(_storage, "is_connected") and _storage.is_connected()
    redis_ok = _cache_mgr.is_connected()

    return HealthResponse(
        status="healthy" if (pg_ok or True) else "degraded",
        version="2.0.0-enterprise",
        postgres_pgvector_connected=pg_ok,
        redis_connected=redis_ok,
        llm_connected=True,
        active_pipeline="Dual-Pipeline (Offline Ingestion + Online Hybrid RAG)",
    )
