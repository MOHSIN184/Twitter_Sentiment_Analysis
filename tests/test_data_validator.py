import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_pipeline.data_validator import validate_dataset  # noqa: E402


def _config() -> dict:
    return {
        "data": {
            "text_column": "text",
            "label_column": "sentiment",
            "allowed_labels": [0, 1],
        }
    }


def test_missing_required_column_raises_error() -> None:
    df = pd.DataFrame({"text": ["good"]})
    with pytest.raises(ValueError, match="Missing required"):
        validate_dataset(df, _config())


def test_invalid_labels_raise_error() -> None:
    df = pd.DataFrame({"text": ["good", "bad"], "sentiment": [1, 2]})
    with pytest.raises(ValueError, match="Invalid label"):
        validate_dataset(df, _config())


def test_missing_text_raises_error() -> None:
    df = pd.DataFrame({"text": ["good", None], "sentiment": [1, 0]})
    with pytest.raises(ValueError, match="Missing values"):
        validate_dataset(df, _config())


def test_valid_dataframe_passes() -> None:
    df = pd.DataFrame({"text": ["good", "bad"], "sentiment": [1, 0]})
    summary = validate_dataset(df, _config())
    assert summary["total_rows"] == 2
    assert summary["label_distribution"] == {"0": 1, "1": 1}
