import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from sentiment_pipeline.api.middleware import RequestLoggingMiddleware
from sentiment_pipeline.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from sentiment_pipeline.api.settings import AppSettings, get_settings
from sentiment_pipeline.batch_predictor import predict_dataframe, save_batch_predictions
from sentiment_pipeline.config import load_config
from sentiment_pipeline.logger import setup_logger
from sentiment_pipeline.paths import build_paths
from sentiment_pipeline.predictor import SentimentPredictor

CONFIG = load_config()
PATHS = build_paths(CONFIG)
SETTINGS = get_settings(CONFIG)
LOGGER = setup_logger(PATHS["logs_dir"])
LOGGER.setLevel(getattr(logging, SETTINGS.log_level.upper(), logging.INFO))
PREDICTOR: Optional[SentimentPredictor] = None
PREDICTION_LOGGER = logging.getLogger("sentiment_pipeline.predictions")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Validate artifacts when the API process starts."""
    validate_startup_artifacts()
    yield

api_config = CONFIG.get("api", {})
app = FastAPI(
    title=api_config.get("title", "Twitter Sentiment Analysis API"),
    version=str(api_config.get("version", "1.0.0")),
    description=api_config.get("description", "Production API for binary Twitter sentiment classification."),
    lifespan=lifespan,
)
app.add_middleware(RequestLoggingMiddleware, logger=LOGGER, enabled=SETTINGS.enable_request_logging)


def _resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PATHS["project_root"] / path


def _prediction_logger() -> logging.Logger:
    """Return a file logger for prediction metadata."""
    if not SETTINGS.enable_prediction_logging:
        return PREDICTION_LOGGER
    if not PREDICTION_LOGGER.handlers:
        prediction_log_path = _resolve_project_path(SETTINGS.prediction_log_path)
        prediction_log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(prediction_log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        PREDICTION_LOGGER.addHandler(handler)
        PREDICTION_LOGGER.setLevel(getattr(logging, SETTINGS.log_level.upper(), logging.INFO))
        PREDICTION_LOGGER.propagate = False
    return PREDICTION_LOGGER


def _model_path() -> Path:
    return _resolve_project_path(SETTINGS.model_path)


def _threshold_path() -> Path:
    return _resolve_project_path(SETTINGS.threshold_path)


def _validate_threshold_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8") as file:
            threshold = float(json.load(file)["threshold"])
    except Exception as exc:
        raise RuntimeError(f"Invalid threshold file: {path}") from exc
    if not 0 <= threshold <= 1:
        raise RuntimeError(f"Threshold must be between 0 and 1: {threshold}")


def get_predictor() -> SentimentPredictor:
    """Return a cached predictor instance, loading the model on first use."""
    global PREDICTOR
    if PREDICTOR is None:
        model_path = _model_path()
        if not model_path.exists():
            raise HTTPException(status_code=503, detail=f"Model file not found: {model_path}")
        _validate_threshold_file(_threshold_path())
        PREDICTOR = SentimentPredictor(model_path=model_path, threshold_path=_threshold_path(), config=CONFIG)
        LOGGER.info("Loaded API predictor from %s", model_path)
    return PREDICTOR


def _validate_text_length(text: str) -> None:
    if len(text) > SETTINGS.max_text_length:
        raise HTTPException(status_code=422, detail=f"text exceeds maximum length of {SETTINGS.max_text_length}")


def _validate_batch_size(size: int) -> None:
    if size > SETTINGS.max_batch_size:
        raise HTTPException(status_code=413, detail=f"batch size exceeds maximum of {SETTINGS.max_batch_size}")


def _log_prediction_metadata(request_id: str, prediction: PredictionResponse) -> None:
    if not SETTINGS.enable_prediction_logging:
        return
    payload = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "request_id": request_id,
        "predicted_label": prediction.predicted_label,
        "predicted_sentiment": prediction.predicted_sentiment,
        "positive_probability": prediction.positive_probability,
        "negative_probability": prediction.negative_probability,
        "threshold": prediction.decision_threshold,
        "text_length": len(prediction.cleaned_text),
    }
    _prediction_logger().info(json.dumps(payload))


def _feature_type() -> Optional[str]:
    metrics_path = PATHS["project_root"] / CONFIG["experiments"]["best_model_metrics_path"]
    if not metrics_path.exists():
        return None
    try:
        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        variant = metrics.get("best_config", {}).get("tfidf_variant")
        if variant is None:
            return None
        return CONFIG["experiments"]["tfidf_variants"].get(variant, {}).get("feature_type")
    except Exception as exc:
        logging.getLogger("sentiment_pipeline").warning("Could not load feature type metadata: %s", exc)
        return None


@app.get("/")
def root() -> dict[str, str]:
    """Return a basic API welcome message."""
    return {"message": "Twitter Sentiment Analysis API"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service health and artifact availability."""
    return HealthResponse(
        status="ok",
        model_loaded=PREDICTOR is not None or _model_path().exists(),
        threshold_loaded=_threshold_path().exists(),
    )


@app.get("/ready")
def ready() -> Dict[str, Any]:
    """Return readiness only when the model artifact can be loaded."""
    try:
        get_predictor()
    except Exception:
        LOGGER.exception("Readiness check failed")
        raise HTTPException(status_code=503, detail="Service is not ready")
    return {"status": "ready", "model_loaded": True}


def validate_startup_artifacts() -> None:
    """Validate production artifacts when the API process starts."""
    model_path = _model_path()
    if not model_path.exists():
        LOGGER.error("Model file is missing at startup: %s", model_path)
        return
    try:
        _validate_threshold_file(_threshold_path())
        get_predictor()
    except Exception:
        LOGGER.exception("API startup artifact validation failed")


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Return model and threshold metadata."""
    threshold = None
    threshold_path = _threshold_path()
    if threshold_path.exists():
        with threshold_path.open("r", encoding="utf-8") as file:
            threshold = float(json.load(file)["threshold"])

    return ModelInfoResponse(
        model_path=str(_model_path()),
        threshold_path=str(threshold_path),
        decision_threshold=threshold,
        labels={"0": "negative", "1": "positive"},
        feature_type=_feature_type(),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, http_request: Request) -> PredictionResponse:
    """Predict sentiment for one text input."""
    _validate_text_length(request.text)
    try:
        prediction = get_predictor().predict_one(request.text)
        response = PredictionResponse(**prediction)
        _log_prediction_metadata(getattr(http_request.state, "request_id", ""), response)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest, http_request: Request) -> BatchPredictionResponse:
    """Predict sentiment for a JSON batch of texts."""
    _validate_batch_size(len(request.texts))
    for text in request.texts:
        _validate_text_length(text)

    try:
        predictions = [PredictionResponse(**prediction) for prediction in get_predictor().predict_many(request.texts)]
        request_id = getattr(http_request.state, "request_id", "")
        for prediction in predictions:
            _log_prediction_metadata(request_id, prediction)
        return BatchPredictionResponse(total_items=len(predictions), predictions=predictions)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail="Batch prediction failed") from exc


@app.post("/predict-csv", response_model=BatchPredictionResponse)
def predict_csv_upload(http_request: Request, file: UploadFile = File(...)) -> BatchPredictionResponse:
    """Predict sentiment for an uploaded CSV containing a text column."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a CSV")

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded CSV: {exc}") from exc

    if "text" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain a text column")

    _validate_batch_size(len(df))
    for text in df["text"].fillna("").astype(str):
        if not text.strip():
            raise HTTPException(status_code=422, detail="CSV text column cannot contain empty values")
        _validate_text_length(text)

    try:
        predictions_df = predict_dataframe(df, get_predictor(), text_column="text")
        output_name = f"api_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_path = PATHS["predictions_dir"] / output_name
        save_batch_predictions(predictions_df, output_path)
        LOGGER.info("Saved API CSV predictions to %s", output_path)
        predictions = [
            PredictionResponse(
                original_text=row["text"],
                cleaned_text=row["cleaned_text"],
                predicted_label=row["predicted_label"],
                predicted_sentiment=row["predicted_sentiment"],
                decision_threshold=row["decision_threshold"],
                negative_probability=row["negative_probability"],
                positive_probability=row["positive_probability"],
            )
            for row in predictions_df.to_dict(orient="records")
        ]
        request_id = getattr(http_request.state, "request_id", "")
        for prediction in predictions:
            _log_prediction_metadata(request_id, prediction)
        return BatchPredictionResponse(total_items=len(predictions), predictions=predictions)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("CSV prediction failed")
        raise HTTPException(status_code=500, detail="CSV prediction failed") from exc
