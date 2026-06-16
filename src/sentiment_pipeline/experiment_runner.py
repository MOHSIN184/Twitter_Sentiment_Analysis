import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from sentiment_pipeline.model_evaluator import evaluate_classification_model
from sentiment_pipeline.model_registry import build_classifier, build_experiment_pipeline

LOGGER = logging.getLogger("sentiment_pipeline")


def _validate_experiment_columns(df: pd.DataFrame, split_name: str) -> None:
    required_columns = ["clean_text", "sentiment"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{split_name} data is missing required column(s): {missing_columns}")


def run_model_experiments(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Train enabled model and TF-IDF combinations, evaluated on validation only."""
    _validate_experiment_columns(train_df, "train")
    _validate_experiment_columns(validation_df, "validation")

    experiment_config = config["experiments"]
    models_config = experiment_config["models"]
    tfidf_variants = experiment_config["tfidf_variants"]
    results = []

    for model_name, model_config in models_config.items():
        if not model_config.get("enabled", False):
            continue

        for tfidf_variant, tfidf_config in tfidf_variants.items():
            experiment_name = f"{model_name}__{tfidf_variant}"
            LOGGER.info("Running experiment: %s", experiment_name)
            start_time = time.perf_counter()

            classifier = build_classifier(model_name, model_config)
            model = build_experiment_pipeline(tfidf_config, classifier)
            model.fit(train_df["clean_text"].astype(str), train_df["sentiment"])
            training_time = time.perf_counter() - start_time

            metrics, _, _ = evaluate_classification_model(model, validation_df, config, "validation")
            results.append(
                {
                    "experiment_name": experiment_name,
                    "model_name": model_name,
                    "tfidf_variant": tfidf_variant,
                    "accuracy": metrics["accuracy"],
                    "macro_precision": metrics["macro_precision"],
                    "macro_recall": metrics["macro_recall"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                    "negative_f1": metrics["negative_f1"],
                    "positive_f1": metrics["positive_f1"],
                    "training_time_seconds": float(training_time),
                }
            )

    if not results:
        raise ValueError("No enabled experiments found in config")

    return pd.DataFrame(results).sort_values("macro_f1", ascending=False).reset_index(drop=True)


def save_experiment_results(results_df: pd.DataFrame, output_path: Path) -> None:
    """Save experiment comparison results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)


def select_best_model(results_df: pd.DataFrame, metric: str = "macro_f1") -> dict[str, Any]:
    """Select the best experiment row by the requested validation metric."""
    if metric not in results_df.columns:
        raise ValueError(f"Metric column not found in results: {metric}")
    best_row = results_df.sort_values(metric, ascending=False).iloc[0]
    return best_row.to_dict()


def train_and_save_best_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    config: dict[str, Any],
    best_config: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Train the best validation-selected model on train only and save it.

    Validation remains untouched for threshold tuning, so the saved model is not
    fit on validation rows before threshold selection.
    """
    experiment_config = config["experiments"]
    model_name = str(best_config["model_name"])
    tfidf_variant = str(best_config["tfidf_variant"])

    classifier = build_classifier(model_name, experiment_config["models"][model_name])
    model = build_experiment_pipeline(experiment_config["tfidf_variants"][tfidf_variant], classifier)
    LOGGER.info("Training best model %s with TF-IDF variant %s on train only", model_name, tfidf_variant)
    model.fit(train_df["clean_text"].astype(str), train_df["sentiment"])

    best_model_path = Path(experiment_config["best_model_path"])
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, best_model_path)

    best_metrics = {
        "selected_by": "validation_macro_f1",
        "best_model_path": str(best_model_path),
        "training_data": "train_only",
        "best_config": {
            "experiment_name": best_config["experiment_name"],
            "model_name": model_name,
            "tfidf_variant": tfidf_variant,
        },
        "validation_metrics": {
            "accuracy": float(best_config["accuracy"]),
            "macro_precision": float(best_config["macro_precision"]),
            "macro_recall": float(best_config["macro_recall"]),
            "macro_f1": float(best_config["macro_f1"]),
            "weighted_f1": float(best_config["weighted_f1"]),
            "negative_f1": float(best_config["negative_f1"]),
            "positive_f1": float(best_config["positive_f1"]),
        },
    }

    metrics_path = Path(experiment_config["best_model_metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(best_metrics, file, indent=2)

    return model, best_metrics
