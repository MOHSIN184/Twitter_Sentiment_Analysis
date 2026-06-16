import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_pipeline.api.schemas import (  # noqa: E402
    BatchPredictionRequest,
    PredictionRequest,
    PredictionResponse,
)


def test_valid_prediction_request() -> None:
    request = PredictionRequest(text="I love this product!")
    assert request.text == "I love this product!"


def test_empty_text_rejected() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(text="   ")


def test_valid_batch_request() -> None:
    request = BatchPredictionRequest(texts=["I love this product!", "This is terrible"])
    assert len(request.texts) == 2


def test_empty_batch_rejected() -> None:
    with pytest.raises(ValidationError):
        BatchPredictionRequest(texts=[])


def test_response_schema_accepts_valid_prediction_output() -> None:
    response = PredictionResponse(
        original_text="I love this product!",
        cleaned_text="I love this product!",
        predicted_label=1,
        predicted_sentiment="positive",
        decision_threshold=0.47,
        negative_probability=0.1,
        positive_probability=0.9,
    )
    assert response.predicted_sentiment == "positive"
