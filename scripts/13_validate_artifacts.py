import json
import sys
from pathlib import Path
from typing import Any, Optional

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_pipeline.config import load_config  # noqa: E402
from sentiment_pipeline.paths import build_paths  # noqa: E402
from sentiment_pipeline.predictor import SentimentPredictor  # noqa: E402


def validate_threshold_value(threshold: float) -> bool:
    """Return True when a threshold is inside the valid probability range."""
    return 0 <= threshold <= 1


def load_threshold(path: Path) -> Optional[dict[str, Any]]:
    """Load and validate a threshold JSON file."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    threshold = float(payload["threshold"])
    if not validate_threshold_value(threshold):
        raise ValueError(f"Threshold must be between 0 and 1: {threshold}")
    return payload


def validate_model_path(model_path: Path) -> bool:
    """Validate that a model file exists and can be loaded by joblib."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    joblib.load(model_path)
    return True


def validate_artifacts() -> bool:
    """Validate required production inference artifacts."""
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        print("FAIL config.yaml missing")
        return False
    print("PASS config.yaml exists")

    config = load_config(config_path)
    paths = build_paths(config)
    model_path = paths["project_root"] / config["inference"]["model_path"]
    threshold_path = paths["project_root"] / config["inference"]["threshold_path"]

    try:
        validate_model_path(model_path)
        print(f"PASS model loads: {model_path}")
    except Exception as exc:
        print(f"FAIL model invalid: {exc}")
        return False

    try:
        threshold = load_threshold(threshold_path)
        if threshold is None:
            print(f"PASS threshold optional and not found: {threshold_path}")
        else:
            print(f"PASS threshold valid: {threshold_path}")
    except Exception as exc:
        print(f"FAIL threshold invalid: {exc}")
        return False

    try:
        predictor = SentimentPredictor(model_path=model_path, threshold_path=threshold_path, config=config)
        prediction = predictor.predict_one("I love this product!")
        if prediction["predicted_label"] not in (0, 1):
            raise ValueError("Predictor returned an invalid label")
        print("PASS SentimentPredictor smoke prediction")
    except Exception as exc:
        print(f"FAIL predictor smoke test: {exc}")
        return False

    print("All production artifacts validated successfully.")
    return True


def main() -> None:
    """Validate production artifacts and exit with a non-zero code on failure."""
    if not validate_artifacts():
        sys.exit(1)


if __name__ == "__main__":
    main()
