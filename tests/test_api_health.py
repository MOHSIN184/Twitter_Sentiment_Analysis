import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_pipeline.api.app import app  # noqa: E402


def test_root_returns_200() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200


def test_health_returns_status_field() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_model_info_returns_expected_fields() -> None:
    client = TestClient(app)
    response = client.get("/model-info")
    assert response.status_code == 200
    payload = response.json()
    assert "model_path" in payload
    assert "threshold_path" in payload
    assert "labels" in payload


def test_ready_returns_valid_json() -> None:
    client = TestClient(app)
    response = client.get("/ready")
    assert response.status_code in {200, 503}
    assert isinstance(response.json(), dict)
