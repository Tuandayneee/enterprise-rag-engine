import re
from pathlib import Path
from typing import Dict, Any, Tuple


class DocumentExtractor:
    """
    Trích xuất văn bản chuẩn hóa từ PDF, Markdown, và Log files
    kèm siêu dữ liệu chi tiết (tên file, số trang, dòng, kích thước).
    """

    @classmethod
    def extract_from_file(cls, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Tài liệu không tồn tại: {file_path}")

        suffix = path.suffix.lower()
        metadata: Dict[str, Any] = {
            "source": path.name,
            "file_path": str(path.resolve()),
            "extension": suffix,
            "file_size": path.stat().st_size,
        }

        if suffix in [".txt", ".md", ".markdown"]:
            text, extra = cls._extract_markdown_or_text(path)
        elif suffix == ".pdf":
            text, extra = cls._extract_pdf(path)
        elif suffix in [".log", ".out"]:
            text, extra = cls._extract_log_file(path)
        else:
            text, extra = cls._extract_markdown_or_text(path)

        metadata.update(extra)
        return text.strip(), metadata

    @staticmethod
    def _extract_markdown_or_text(path: Path) -> Tuple[str, Dict[str, Any]]:
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        content = ""
        for enc in encodings:
            try:
                with open(path, "r", encoding=enc) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, LookupError):
                continue
        # Trích xuất các tiêu đề markdown
        headers = re.findall(r"^(#{1,6})\s+(.+)$", content, flags=re.MULTILINE)
        return content, {
            "type": "markdown" if path.suffix in [".md", ".markdown"] else "text",
            "headers_count": len(headers),
            "line_count": len(content.splitlines()),
        }

    @staticmethod
    def _extract_pdf(path: Path) -> Tuple[str, Dict[str, Any]]:
        # Ưu tiên pypdf
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            pages = []
            for idx, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                if txt.strip():
                    pages.append(f"--- [Trang {idx + 1}] ---\n" + txt.strip())
            return "\n\n".join(pages), {
                "type": "pdf",
                "total_pages": len(reader.pages),
            }
        except ImportError:
            return f"[Nội dung thô của PDF {path.name}]", {"type": "pdf", "total_pages": 1}

    @staticmethod
    def _extract_log_file(path: Path) -> Tuple[str, Dict[str, Any]]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        # Chuẩn hóa log files: lọc timestamps và phân tích mức log (INFO, WARN, ERROR)
        cleaned_lines = []
        error_count = 0
        for line in lines:
            if any(lvl in line.upper() for lvl in ["ERROR", "FATAL", "CRITICAL"]):
                error_count += 1
            cleaned_lines.append(line.strip())
        return "\n".join(cleaned_lines), {
            "type": "log",
            "total_lines": len(lines),
            "error_lines_detected": error_count,
        }
