from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChunkStrategy(str, Enum):
    FIXED_SIZE = "fixed_size"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class Document(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str
    document_id: str
    text: str
    strategy: ChunkStrategy = ChunkStrategy.SEMANTIC
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    final_score: float = 0.0


class ChunkingBenchmarkResult(BaseModel):
    strategy: str
    total_chunks: int
    avg_chunk_length: float
    std_chunk_length: float
    context_completeness_score: float
    information_loss_ratio: float
    execution_time_ms: float


class IngestionRequest(BaseModel):
    document_id: Optional[str] = None
    content: str
    strategy: ChunkStrategy = ChunkStrategy.SEMANTIC
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestionResponse(BaseModel):
    success: bool
    document_id: str
    total_chunks: int
    strategy: str
    message: str


class SearchQuery(BaseModel):
    query: str
    top_k: int = 5
    candidate_pool: int = 20
    use_reranker: bool = True
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    filters: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    query: str
    total_candidates: int
    results: List[Chunk]
    execution_time_ms: float


class Citation(BaseModel):
    source: str
    document_id: str
    chunk_id: str
    page: Optional[int] = None
    snippet: str


class ChatQuery(BaseModel):
    question: str
    top_k: int = 5
    stream: bool = False
    filters: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    retrieved_chunks: int
    cached: bool = False
    execution_time_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    postgres_pgvector_connected: bool
    redis_connected: bool
    llm_connected: bool
    storage_backend: str = "in_memory_dual_store"
    total_chunks_indexed: int = 0
    active_pipeline: str
