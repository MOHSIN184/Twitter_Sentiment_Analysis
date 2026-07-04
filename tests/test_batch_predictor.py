import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_pipeline.batch_predictor import PREDICTION_COLUMNS, predict_dataframe, predict_xquik_export  # noqa: E402


class FakePredictor:
    def predict_one(self, text: str) -> dict:
        label = 1 if "love" in text.lower() else 0
        return {
            "original_text": text,
            "cleaned_text": text.strip(),
            "predicted_label": label,
            "predicted_sentiment": "positive" if label == 1 else "negative",
            "decision_threshold": 0.47,
            "negative_probability": 0.2 if label == 1 else 0.8,
            "positive_probability": 0.8 if label == 1 else 0.2,
        }


def test_predict_dataframe_works_with_fake_predictor() -> None:
    df = pd.DataFrame({"text": ["I love this", "This is terrible"]})
    predictions = predict_dataframe(df, FakePredictor())
    assert len(predictions) == 2
    assert predictions["predicted_label"].tolist() == [1, 0]


def test_missing_text_column_raises_error() -> None:
    df = pd.DataFrame({"message": ["hello"]})
    with pytest.raises(ValueError, match="missing required text column"):
        predict_dataframe(df, FakePredictor())


def test_output_dataframe_contains_required_prediction_columns() -> None:
    df = pd.DataFrame({"text": ["I love this"]})
    predictions = predict_dataframe(df, FakePredictor())
    assert predictions.columns.tolist() == PREDICTION_COLUMNS


def test_predict_dataframe_can_keep_source_columns() -> None:
    df = pd.DataFrame({"text": ["I love this"], "tweet_id": ["123"]})
    predictions = predict_dataframe(df, FakePredictor(), keep_source_columns=True)
    assert predictions["tweet_id"].tolist() == ["123"]
    assert predictions.columns.tolist() == ["tweet_id", *PREDICTION_COLUMNS]


def test_predict_xquik_export_preserves_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "xquik.json"
    output_path = tmp_path / "predictions.csv"
    input_path.write_text('{"data":[{"text":"I love this","tweet_id":"123","username":"demo"}]}', encoding="utf-8")

    predictions = predict_xquik_export(input_path, output_path, FakePredictor())

    assert output_path.exists()
    assert predictions["tweet_id"].tolist() == ["123"]
    assert predictions["user"].tolist() == ["demo"]
