"""
Backward-compatibility proxy to src.main:app
Cho phép khởi chạy uvicorn app.main:app hoặc uvicorn src.main:app
"""
import sys
from pathlib import Path

# Đảm bảo root directory luôn nằm trong sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.main import app, create_application

__all__ = ["app", "create_application"]

if __name__ == "__main__":
    import uvicorn
    from src.core.config import get_settings

    settings = get_settings()
    uvicorn.run("src.main:app", host=settings.app_host, port=settings.app_port, reload=True)
