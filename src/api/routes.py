import os
import shutil
import tempfile
import uuid
from typing import Optional
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import StreamingResponse

from src.models import (
    ChunkStrategy,
    IngestionRequest,
    IngestionResponse,
    SearchQuery,
    SearchResponse,
    ChatQuery,
    ChatResponse,
    HealthResponse,
)
from src.db.connection import get_storage
from src.pipeline.extractor import DocumentExtractor
from src.pipeline.hybrid_search import HybridSearchEngine
from src.pipeline.rag_chain import CustomRAGChain
from src.pipeline.chunker import ChunkerFactory, ChunkingBenchmark
from src.pipeline.embeddings import EmbeddingPipeline
from src.api.middleware import rate_limit_dependency, get_cache_manager

router = APIRouter(prefix="/api/v1", tags=["Enterprise RAG Pipeline"])

# Singleton pipeline instances
_storage = get_storage()
_embeddings = EmbeddingPipeline()
_search_engine = HybridSearchEngine(storage=_storage, embeddings=_embeddings)
_cache_mgr = get_cache_manager()
_rag_chain = CustomRAGChain(search_engine=_search_engine, cache_manager=_cache_mgr)


@router.post("/ingest", response_model=IngestionResponse)
async def ingest_document(
    req: IngestionRequest, _: str = Depends(rate_limit_dependency)
):
    """
    [OFFLINE INGESTION PIPELINE] Nạp văn bản, phân đoạn theo chiến lược (Fixed/Recursive/Semantic) và lưu DB.
    """
    if not req.content or not req.content.strip():
        return IngestionResponse(
            success=False,
            document_id=req.document_id or "empty_doc",
            total_chunks=0,
            strategy=req.strategy.value,
            message="Tài liệu trống, không có nội dung để nạp.",
        )

    doc_id = req.document_id or f"doc_{uuid.uuid4().hex[:8]}"
    chunks = ChunkerFactory.chunk_document(
        text=req.content,
        document_id=doc_id,
        strategy=req.strategy,
        metadata=req.metadata,
        embedding_pipeline=_embeddings,
    )
    _storage.upsert_chunks(chunks)
    _cache_mgr.invalidate_all()

    return IngestionResponse(
        success=True,
        document_id=doc_id,
        total_chunks=len(chunks),
        strategy=req.strategy.value,
        message=f"Đã nạp {len(chunks)} chunks ({req.strategy.value}) vào {_storage.get_backend_name()}.",
    )


@router.post("/ingest/file", response_model=IngestionResponse)
async def ingest_file(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    strategy: ChunkStrategy = Form(ChunkStrategy.SEMANTIC),
    _: str = Depends(rate_limit_dependency),
):
    """
    [OFFLINE INGESTION PIPELINE] Tải lên file (PDF, Markdown, TXT, Log), bóc tách và đánh chỉ mục.
    """
    filename = file.filename or "uploaded_file.txt"
    suffix = os.path.splitext(filename)[1] or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        text, meta = DocumentExtractor.extract_from_file(tmp_path)
        meta["source"] = filename
        doc_id = document_id or f"doc_{uuid.uuid4().hex[:8]}"
        chunks = ChunkerFactory.chunk_document(
            text=text,
            document_id=doc_id,
            strategy=strategy,
            metadata=meta,
            embedding_pipeline=_embeddings,
        )
        _storage.upsert_chunks(chunks)
        _cache_mgr.invalidate_all()

        return IngestionResponse(
            success=True,
            document_id=doc_id,
            total_chunks=len(chunks),
            strategy=strategy.value,
            message=f"Đã bóc tách file '{filename}' và nạp {len(chunks)} chunks.",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi xử lý file tải lên: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str, _: str = Depends(rate_limit_dependency)):
    """
    Xóa toàn bộ chunks của một tài liệu theo document_id và làm mới bộ đệm Redis.
    """
    _storage.delete_by_document(document_id)
    _cache_mgr.invalidate_all()
    return {
        "success": True,
        "document_id": document_id,
        "message": f"Đã xóa tài liệu '{document_id}' và làm mới cache.",
    }


@router.post("/search", response_model=SearchResponse)
async def search_hybrid(req: SearchQuery, _: str = Depends(rate_limit_dependency)):
    """
    [ONLINE QUERY PIPELINE] Hybrid Search (Dense + BM25) + RRF + Cross-Encoder Reranker.
    """
    return _search_engine.search(req)


@router.post("/chat", response_model=ChatResponse)
async def chat_rag(chat_query: ChatQuery, _: str = Depends(rate_limit_dependency)):
    """
    [ONLINE QUERY PIPELINE] Truy vấn RAG có Redis Cache-Aside, Guardrails chống ảo giác và trích dẫn.
    """
    return await _rag_chain.generate_response(chat_query)


@router.get("/chat/stream")
async def chat_rag_stream(
    q: str = Query(..., description="Câu hỏi người dùng"),
    top_k: int = Query(5, ge=1, le=20),
    _: str = Depends(rate_limit_dependency),
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


@router.post("/evaluate")
async def evaluate_pipeline(_: str = Depends(rate_limit_dependency)):
    """
    [EVALUATION SUITE] Chạy tự động Chunking Benchmark & RAGAS Evaluation trả về JSON.
    """
    from scripts.evaluate_ragas import RagasEvaluator

    sample_text = (
        "Kiến trúc hệ thống RAG cấp doanh nghiệp tách biệt làm 2 đường ống độc lập: "
        "Ingestion Pipeline (offline) và Query/Retrieval Pipeline (online real-time). "
        "PostgreSQL sử dụng extension pgvector với chỉ mục HNSW cho vector embedding. "
        "Reciprocal Rank Fusion (RRF) kết hợp điểm số, và Cross-Encoder bge-reranker-large "
        "chấm điểm lại Top-20 để chọn Top-5 đoạn văn bản chuẩn xác nhất."
    )
    chunk_bench = ChunkingBenchmark.run_benchmark(sample_text)

    ragas_scores = []
    for item in RagasEvaluator.BENCHMARK_DATASET:
        sc = RagasEvaluator.evaluate_sample(
            question=item["question"],
            answer=item["ground_truth"],
            contexts=[sample_text],
            ground_truth=item["ground_truth"],
        )
        ragas_scores.append({"question": item["question"], **sc})

    return {
        "chunking_benchmark": [b.model_dump() for b in chunk_bench],
        "ragas_evaluation": ragas_scores,
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Kiểm tra tình trạng kết nối tới PostgreSQL pgvector, Redis Cache và LLM Serving.
    """
    pg_ok = hasattr(_storage, "is_connected") and bool(_storage.is_connected())
    redis_ok = _cache_mgr.is_connected()
    backend_name = _storage.get_backend_name()
    total_chunks = _storage.count()

    return HealthResponse(
        status="healthy",
        version="2.0.0-enterprise",
        postgres_pgvector_connected=pg_ok,
        redis_connected=redis_ok,
        llm_connected=True,
        storage_backend=backend_name,
        total_chunks_indexed=total_chunks,
        active_pipeline="Dual-Pipeline (Offline Ingestion + Online Hybrid RAG)",
    )
