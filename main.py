"""
Root entrypoint proxy to src.main:app
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.main import app, create_application

__all__ = ["app", "create_application"]

if __name__ == "__main__":
    import uvicorn
    from src.core.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "src.main:app", host=settings.app_host, port=settings.app_port, reload=True
    )
