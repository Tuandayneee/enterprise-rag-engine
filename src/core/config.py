from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # PostgreSQL & pgvector
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "rag_enterprise"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    database_url: Optional[str] = None

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    redis_cache_ttl: int = 3600
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60

    # LLM Serving (vLLM / Ollama)
    llm_provider: str = "mock"  # "mock", "ollama", "vllm", "openai"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:7b-instruct"
    openai_api_key: Optional[str] = None
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # Embedding & Cross-Encoder Models
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    reranker_model_name: str = "BAAI/bge-reranker-large"

    # Chunking & Retrieval Parameters
    chunk_similarity_threshold: float = 0.75
    fixed_chunk_size: int = 512
    fixed_chunk_overlap: int = 50
    semantic_min_chunk_size: int = 100
    semantic_max_chunk_size: int = 800
    rrf_k: int = 60
    rerank_top_k: int = 5
    rerank_candidate_pool: int = 20

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def get_postgres_dsn(self) -> str:
        if self.database_url:
            return self.database_url
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
