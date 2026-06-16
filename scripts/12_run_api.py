import os
import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_pipeline.config import load_config  # noqa: E402
from sentiment_pipeline.api.settings import get_settings  # noqa: E402


def main() -> None:
    """Start the FastAPI inference service."""
    config = load_config()
    settings = get_settings(config)
    host = settings.api_host if os.getenv("API_HOST") else config.get("api", {}).get("host", "127.0.0.1")
    port = settings.api_port
    print(f"Starting API at http://{host}:{port}")
    print(f"Swagger docs available at http://{host}:{port}/docs")
    uvicorn.run("sentiment_pipeline.api.app:app", host=host, port=port, log_level=settings.log_level.lower())


if __name__ == "__main__":
    main()
