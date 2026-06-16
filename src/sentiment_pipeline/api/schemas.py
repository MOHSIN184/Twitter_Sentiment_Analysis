from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Request body for a single sentiment prediction."""

    text: str = Field(..., max_length=1000)

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        """Reject blank prediction text."""
        if not value or not value.strip():
            raise ValueError("text cannot be empty")
        return value


class PredictionResponse(BaseModel):
    """Response body for one sentiment prediction."""

    original_text: str
    cleaned_text: str
    predicted_label: int
    predicted_sentiment: str
    decision_threshold: Optional[float]
    negative_probability: Optional[float]
    positive_probability: Optional[float]


class BatchPredictionRequest(BaseModel):
    """Request body for batch sentiment prediction."""

    texts: List[str] = Field(..., min_length=1)

    @field_validator("texts")
    @classmethod
    def texts_must_not_contain_empty_values(cls, values: List[str]) -> List[str]:
        """Reject batches with blank text values."""
        if any(not value or not value.strip() for value in values):
            raise ValueError("texts cannot contain empty values")
        return values


class BatchPredictionResponse(BaseModel):
    """Response body for batch sentiment prediction."""

    total_items: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    """API health response."""

    status: str
    model_loaded: bool
    threshold_loaded: bool


class ModelInfoResponse(BaseModel):
    """Model metadata response."""

    model_path: str
    threshold_path: str
    decision_threshold: Optional[float]
    labels: dict
    feature_type: Optional[str]
