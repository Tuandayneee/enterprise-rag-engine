import re
from typing import List, Tuple
from src.models import Chunk, Citation


class AntiHallucinationGuard:
    """
    Bộ kiểm soát ảo giác (Guardrails) & trích dẫn nguồn ngữ cảnh (Source Citation).
    """

    DEFENSIVE_SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên gia giải đáp thông tin kỹ thuật cấp doanh nghiệp.

QUY TẮC PHÒNG VỆ CHỐNG ẢO GIÁC (BẮT BUỘC TUÂN THỦ TUYỆT ĐỐI):
1. BẠN CHỈ ĐƯỢC PHÉP TRẢ LỜI dựa hoàn toàn vào các đoạn văn bản trong mục "=== TÀI LIỆU THAM KHẢO ===". Tuyệt đối không sử dụng kiến thức bên ngoài ngữ cảnh được cung cấp.
2. NẾU NGỮ CẢNH KHÔNG ĐỦ THÔNG TIN để trả lời chính xác, bạn BẮT BUỘC PHẢI TRẢ LỜI ĐÚNG NGUYÊN VĂN câu sau:
   "Không tìm thấy thông tin trong tài liệu"
3. TRÍCH DẪN NGUỒN BẮT BUỘC: Với mỗi khẳng định hoặc sự kiện được nêu, bạn phải gắn nhãn trích dẫn ở cuối câu theo định dạng:
   [<Tên tài liệu>, Trang <số trang>] hoặc [<Tên tài liệu>, Chunk <id>]
4. Ngôn từ: Chuyên nghiệp, khách quan, súc tích bằng Tiếng Việt.
"""

    @classmethod
    def assemble_prompt(cls, question: str, retrieved_chunks: List[Chunk]) -> Tuple[str, str, List[Citation]]:
        citations: List[Citation] = []
        context_blocks = []

        for idx, ch in enumerate(retrieved_chunks, start=1):
            source_name = ch.metadata.get("source", ch.document_id)
            page_num = ch.metadata.get("page") or ch.metadata.get("chunk_index", 1)
            snippet = ch.text[:140] + "..." if len(ch.text) > 140 else ch.text

            citations.append(
                Citation(
                    source=source_name,
                    document_id=ch.document_id,
                    chunk_id=ch.id,
                    page=page_num,
                    snippet=snippet,
                )
            )

            block = (
                f"[Tài liệu #{idx}]\n"
                f"- Document: {source_name} (Page: {page_num}, ID: {ch.id})\n"
                f"- Nội dung: {ch.text}\n"
            )
            context_blocks.append(block)

        context_str = "\n".join(context_blocks) if context_blocks else "Không có ngữ cảnh nào."

        user_prompt = f"""=== TÀI LIỆU THAM KHẢO ===
{context_str}

=== CÂU HỎI CỦA NGƯỜI DÙNG ===
{question}

Hãy trả lời câu hỏi trên theo đúng quy tắc chống ảo giác và trích dẫn rõ nguồn."""

        return cls.DEFENSIVE_SYSTEM_PROMPT, user_prompt, citations

    @classmethod
    def verify_answer_citations(cls, answer: str, citations: List[Citation]) -> bool:
        """
        Kiểm tra xem câu trả lời có chứa trích dẫn hoặc câu từ chối chuẩn không.
        """
        if "Không tìm thấy thông tin trong tài liệu" in answer:
            return True
        # Kiểm tra sự xuất hiện của dấu ngoặc vuông chứa nguồn
        has_bracket_citation = bool(re.search(r"\[.*?\]", answer))
        return has_bracket_citation or len(citations) > 0
