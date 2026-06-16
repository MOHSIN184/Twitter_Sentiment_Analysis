import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def _positive_probability_index(model: Any) -> int:
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 not in classes:
        raise ValueError("Model classes do not include positive label 1")
    return classes.index(1)


def evaluate_thresholds(model: Any, validation_df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    """Evaluate classification metrics for candidate positive-class thresholds."""
    required_columns = ["clean_text", "sentiment"]
    missing_columns = [column for column in required_columns if column not in validation_df.columns]
    if missing_columns:
        raise ValueError(f"Validation data is missing required column(s): {missing_columns}")
    if not hasattr(model, "predict_proba"):
        raise ValueError("Threshold tuning requires a model that supports predict_proba")

    y_true = validation_df["sentiment"]
    probabilities = model.predict_proba(validation_df["clean_text"].astype(str))
    positive_probabilities = probabilities[:, _positive_probability_index(model)]
    rows = []

    for threshold in thresholds:
        y_pred = (positive_probabilities >= threshold).astype(int)
        macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        per_label = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        false_positives = int(matrix[0, 1])
        false_negatives = int(matrix[1, 0])

        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_precision": float(macro[0]),
                "macro_recall": float(macro[1]),
                "macro_f1": float(macro[2]),
                "weighted_f1": float(weighted[2]),
                "negative_f1": float(per_label[2][0]),
                "positive_f1": float(per_label[2][1]),
                "false_positives": false_positives,
                "false_negatives": false_negatives,
            }
        )

    return pd.DataFrame(rows)


def select_best_threshold(results_df: pd.DataFrame, metric: str = "macro_f1") -> dict[str, Any]:
    """Select the threshold row with the highest metric value."""
    if metric not in results_df.columns:
        raise ValueError(f"Metric column not found in threshold results: {metric}")
    best_row = results_df.sort_values(metric, ascending=False).iloc[0].to_dict()
    return best_row


def save_threshold_results(results_df: pd.DataFrame, output_path: Path) -> None:
    """Save threshold tuning results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)


def save_threshold_config(best_threshold: float, best_metrics: dict[str, Any], output_path: Path) -> None:
    """Save the selected threshold and key validation metrics to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "threshold": float(best_threshold),
        "selected_metric": "macro_f1",
        "selected_metric_value": float(best_metrics["macro_f1"]),
        "accuracy": float(best_metrics["accuracy"]),
        "macro_f1": float(best_metrics["macro_f1"]),
        "weighted_f1": float(best_metrics["weighted_f1"]),
        "false_positives": int(best_metrics["false_positives"]),
        "false_negatives": int(best_metrics["false_negatives"]),
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def load_threshold_config(path: Path) -> dict[str, Any]:
    """Load a saved threshold configuration from JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Threshold configuration not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
