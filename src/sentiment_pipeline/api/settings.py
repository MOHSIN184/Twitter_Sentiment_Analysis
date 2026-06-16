import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class AppSettings:
    """Runtime settings for the inference API."""

    app_env: str
    api_host: str
    api_port: int
    model_path: Path
    threshold_path: Path
    log_level: str
    max_text_length: int
    max_batch_size: int
    enable_request_logging: bool
    enable_prediction_logging: bool
    prediction_log_path: Path


def get_settings(config: dict[str, Any]) -> AppSettings:
    """Build application settings from environment variables with config fallbacks."""
    load_dotenv()
    deployment = config.get("deployment", {})
    inference = config.get("inference", {})
    observability = config.get("observability", {})

    return AppSettings(
        app_env=os.getenv("APP_ENV", deployment.get("app_env", "production")),
        api_host=os.getenv("API_HOST", deployment.get("api_host", config.get("api", {}).get("host", "127.0.0.1"))),
        api_port=int(os.getenv("API_PORT", deployment.get("api_port", config.get("api", {}).get("port", 8000)))),
        model_path=Path(os.getenv("MODEL_PATH", inference.get("model_path", "models/baseline/best_model.joblib"))),
        threshold_path=Path(
            os.getenv("THRESHOLD_PATH", inference.get("threshold_path", "models/baseline/decision_threshold.json"))
        ),
        log_level=os.getenv("LOG_LEVEL", deployment.get("log_level", "INFO")),
        max_text_length=int(os.getenv("MAX_TEXT_LENGTH", inference.get("max_text_length", 1000))),
        max_batch_size=int(os.getenv("MAX_BATCH_SIZE", inference.get("max_batch_size", 1000))),
        enable_request_logging=_to_bool(
            os.getenv("ENABLE_REQUEST_LOGGING", observability.get("enable_request_logging", True))
        ),
        enable_prediction_logging=_to_bool(
            os.getenv("ENABLE_PREDICTION_LOGGING", observability.get("enable_prediction_logging", True))
        ),
        prediction_log_path=Path(
            os.getenv("PREDICTION_LOG_PATH", observability.get("prediction_log_path", "outputs/logs/predictions.log"))
        ),
    )
