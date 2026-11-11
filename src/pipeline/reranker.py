import re
import logging
from typing import List, Tuple, Optional
from src.models import Chunk
from src.core.config import get_settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Tái xếp hạng chính xác bằng mô hình Transformer Cross-Encoder (bge-reranker-large).
    Đánh giá tương tác chéo sâu giữa Query và Passage để chọn lọc Top-K tốt nhất.
    """

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model_name
        self._model = None
        self._initialized = False

    def _init_model(self):
        if self._initialized:
            return
        self._initialized = True
        settings = get_settings()
        if not settings.use_neural_models:
            self._model = None
            return
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            logger.info(f"Đã nạp Cross-Encoder model: {self.model_name}")
        except Exception as e:
            logger.info(
                f"Cross-Encoder '{self.model_name}' chưa có sẵn offline ({e}). Kích hoạt bộ tái chấm điểm thuật toán."
            )
            self._model = None

    def rerank(
        self, query: str, candidates: List[Chunk], top_k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        if not candidates:
            return []

        self._init_model()

        # 1. Nếu mô hình bge-reranker tải thành công
        if self._model is not None:
            try:
                pairs = [[query, c.text] for c in candidates]
                scores = self._model.predict(pairs)
                ranked = list(zip(candidates, [float(s) for s in scores]))
                ranked.sort(key=lambda x: x[1], reverse=True)
                return ranked[:top_k]
            except Exception as e:
                logger.warning(
                    f"Lỗi suy luận Cross-Encoder ({e}). Dùng bộ chấm điểm dự phòng."
                )

        # 2. Bộ chấm điểm tương tác sâu dự phòng (Heuristic Cross-Interaction Scorer)
        q_tokens = [t.lower() for t in re.findall(r"\w+", query)]
        if not q_tokens:
            return [(c, 0.5) for c in candidates[:top_k]]

        q_phrase = query.lower().strip()
        scored: List[Tuple[Chunk, float]] = []

        for ch in candidates:
            text_lower = ch.text.lower()
            ch_tokens = set(re.findall(r"\w+", text_lower))

            # Độ bao phủ từ khóa
            match_count = sum(1 for qt in q_tokens if qt in ch_tokens)
            coverage_score = match_count / len(q_tokens)

            # Khớp nguyên cụm từ (Exact phrase match bonus)
            phrase_bonus = 0.35 if q_phrase in text_lower else 0.0

            # Mật độ xuất hiện (Term density)
            density = sum(min(text_lower.count(qt) * 0.05, 0.25) for qt in q_tokens)

            final_score = min(0.99, (0.45 * coverage_score) + phrase_bonus + density)
            scored.append((ch, float(final_score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
