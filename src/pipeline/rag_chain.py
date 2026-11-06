import json
import time
import asyncio
import logging
from typing import AsyncIterator, List, Optional
from src.models import Chunk, ChatQuery, ChatResponse
from src.core.config import get_settings
from src.core.cache import RedisCacheManager
from src.core.guardrails import AntiHallucinationGuard
from src.pipeline.hybrid_search import HybridSearchEngine

logger = logging.getLogger(__name__)


class CustomRAGChain:
    """
    Chuỗi RAG tùy chỉnh (Custom RAG Chain) kết nối vLLM / Ollama (Qwen2.5 / Llama-3.1)
    hoặc Mock fallback, tích hợp Guardrails và phát luồng Server-Sent Events (SSE).
    """

    def __init__(
        self,
        search_engine: Optional[HybridSearchEngine] = None,
        cache_manager: Optional[RedisCacheManager] = None,
    ):
        self.settings = get_settings()
        self.search_engine = search_engine or HybridSearchEngine()
        self.cache_manager = cache_manager or RedisCacheManager()

    async def generate_response(self, chat_query: ChatQuery) -> ChatResponse:
        t0 = time.perf_counter()
        question = chat_query.question.strip()

        # 1. KIỂM TRA REDIS CACHE (CACHE-ASIDE PATTERN)
        cached = self.cache_manager.get_cached_response(question, config_prefix=str(chat_query.top_k))
        if cached:
            resp = ChatResponse(**cached)
            resp.cached = True
            resp.execution_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return resp

        # 2. HYBRID SEARCH RETRIEVAL (Dense + Sparse + RRF + Reranker)
        from src.models import SearchQuery
        s_res = self.search_engine.search(
            SearchQuery(query=question, top_k=chat_query.top_k, use_reranker=True)
        )
        retrieved_chunks = s_res.results

        # 3. ASSEMBLE DEFENSIVE PROMPT & CITATIONS
        sys_prompt, user_prompt, citations = AntiHallucinationGuard.assemble_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        # 4. GỌI LLM SERVING (vLLM / Ollama / OpenAI / Mock)
        raw_answer = await self._call_llm(sys_prompt, user_prompt, retrieved_chunks)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        response = ChatResponse(
            question=question,
            answer=raw_answer.strip(),
            citations=citations,
            retrieved_chunks=len(retrieved_chunks),
            cached=False,
            execution_time_ms=round(elapsed_ms, 2),
        )

        # 5. GHI VÀO REDIS CACHE
        self.cache_manager.set_cached_response(
            question, response.model_dump(), config_prefix=str(chat_query.top_k)
        )

        return response

    async def stream_response(self, question: str, top_k: int = 5) -> AsyncIterator[str]:
        """
        Phát luồng Server-Sent Events (SSE):
          - event: citation
          - event: token
          - event: done
        """
        from src.models import SearchQuery
        s_res = self.search_engine.search(SearchQuery(query=question, top_k=top_k, use_reranker=True))
        retrieved_chunks = s_res.results

        sys_prompt, user_prompt, citations = AntiHallucinationGuard.assemble_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        # Phát citation trước
        citation_payload = {
            "citations": [c.model_dump() for c in citations],
            "total_chunks": len(retrieved_chunks),
        }
        yield f"event: citation\ndata: {json.dumps(citation_payload, ensure_ascii=False)}\n\n"

        # Stream tokens
        async for token in self._stream_llm(sys_prompt, user_prompt, retrieved_chunks):
            token_payload = {"token": token}
            yield f"event: token\ndata: {json.dumps(token_payload, ensure_ascii=False)}\n\n"

        yield "event: done\ndata: {\"status\": \"completed\"}\n\n"

    async def _call_llm(self, sys_prompt: str, user_prompt: str, context_chunks: List[Chunk]) -> str:
        provider = self.settings.llm_provider.lower()

        # vLLM / Ollama OpenAI-compatible endpoint
        if provider in ["ollama", "vllm", "openai"]:
            try:
                import httpx
                headers = {"Content-Type": "application/json"}
                if self.settings.openai_api_key:
                    headers["Authorization"] = f"Bearer {self.settings.openai_api_key}"

                endpoint = f"{self.settings.llm_base_url.rstrip('/')}/chat/completions"
                payload = {
                    "model": self.settings.llm_model,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": self.settings.llm_temperature,
                    "max_tokens": self.settings.llm_max_tokens,
                }
                async with httpx.AsyncClient(timeout=45.0) as client:
                    resp = await client.post(endpoint, json=payload, headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        return data["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"Lỗi gọi {provider} ({e}). Dùng Mock Generation.")

        # Fallback Mock Generator có kiểm soát ngữ cảnh
        return self._generate_grounded_mock(user_prompt, context_chunks)

    async def _stream_llm(self, sys_prompt: str, user_prompt: str, context_chunks: List[Chunk]) -> AsyncIterator[str]:
        full_text = await self._call_llm(sys_prompt, user_prompt, context_chunks)
        words = full_text.split(" ")
        for w in words:
            yield w + " "
            await asyncio.sleep(0.015)

    def _generate_grounded_mock(self, user_prompt: str, context_chunks: List[Chunk]) -> str:
        if not context_chunks:
            return "Không tìm thấy thông tin trong tài liệu."

        first_chunk = context_chunks[0]
        source_name = first_chunk.metadata.get("source", first_chunk.document_id)
        page_val = first_chunk.metadata.get("page", 1)

        return (
            f"Dựa trên tài liệu [{source_name}, Trang {page_val}], hệ thống đã xác định: "
            f"{first_chunk.text[:180]}... "
            f"Thông tin này hoàn toàn thỏa mãn các tiêu chí được truy xuất từ cơ sở dữ liệu."
        )
