import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sentiment_pipeline.feature_extraction import build_tfidf_vectorizer
from sentiment_pipeline.model_evaluator import evaluate_classification_model

LOGGER = logging.getLogger("sentiment_pipeline")


def _validate_training_columns(df: pd.DataFrame, text_column: str, label_column: str, split_name: str) -> None:
    missing_columns = [column for column in (text_column, label_column) if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{split_name} data is missing required column(s): {missing_columns}")
    if df[text_column].isna().any():
        raise ValueError(f"{split_name} data contains missing values in {text_column}")
    if df[label_column].isna().any():
        raise ValueError(f"{split_name} data contains missing values in {label_column}")


def train_baseline_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[Pipeline, dict[str, Any]]:
    """Train a TF-IDF plus Logistic Regression baseline and evaluate validation metrics."""
    text_column = "clean_text"
    label_column = config["data"]["label_column"]
    _validate_training_columns(train_df, text_column, label_column, "train")
    _validate_training_columns(validation_df, text_column, label_column, "validation")

    lr_config = config.get("model", {}).get("logistic_regression", {})
    classifier = LogisticRegression(
        max_iter=int(lr_config.get("max_iter", 1000)),
        class_weight=lr_config.get("class_weight"),
        solver=lr_config.get("solver", "liblinear"),
        random_state=int(config["project"]["random_state"]),
    )

    model = Pipeline(
        steps=[
            ("tfidf", build_tfidf_vectorizer(config)),
            ("classifier", classifier),
        ]
    )

    LOGGER.info("Training baseline model on %s rows", len(train_df))
    model.fit(train_df[text_column].astype(str), train_df[label_column])
    LOGGER.info("Evaluating baseline model on %s validation rows", len(validation_df))
    validation_metrics, _, _ = evaluate_classification_model(model, validation_df, config, "validation")
    return model, validation_metrics


def save_model(model: Pipeline, model_path: Path) -> None:
    """Save a trained model pipeline with joblib."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    LOGGER.info("Saved model to %s", model_path)


def load_model(model_path: Path) -> Pipeline:
    """Load a trained model pipeline with joblib."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise ValueError(f"Could not load model from {model_path}: {exc}") from exc
    LOGGER.info("Loaded model from %s", model_path)
    return model
