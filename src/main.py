import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.core.config import get_settings
from src.api.routes import router

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("Enterprise-RAG")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("==================================================================")
    logger.info("   ENTERPRISE RAG ENGINE - DUAL PIPELINE ARCHITECTURE IS READY    ")
    logger.info("   - Offline Ingestion: Semantic Chunking -> pgvector / BM25      ")
    logger.info("   - Online Query: Redis Cache -> Hybrid Search -> RRF -> Rerank  ")
    logger.info("   - LLM Serving: vLLM / Ollama -> Defensive Guardrails -> SSE    ")
    logger.info(
        f"   Environment: {settings.app_env.upper()} | Port: {settings.app_port}"
    )
    logger.info("==================================================================")
    yield
    logger.info("Enterprise RAG Engine đã dừng hoạt động an toàn.")


def create_application() -> FastAPI:
    app = FastAPI(
        title="Enterprise RAG Engine",
        description=(
            "Hệ thống RAG cấp doanh nghiệp tách biệt 2 đường ống độc lập:\n"
            "- [OFFLINE INGESTION PIPELINE]: Semantic Chunking -> bge-m3 -> pgvector (HNSW) & tsvector (BM25).\n"
            "- [ONLINE QUERY PIPELINE]: Redis Cache -> Hybrid Search (Dense + Sparse) -> RRF -> bge-reranker-large -> Anti-Hallucination Guardrails -> SSE Streaming."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_application()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.main:app", host=settings.app_host, port=settings.app_port, reload=True
    )
