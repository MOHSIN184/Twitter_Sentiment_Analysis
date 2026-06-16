import json
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_classification_model(
    model: Any,
    df: pd.DataFrame,
    config: dict[str, Any],
    split_name: str,
    threshold: Optional[float] = None,
) -> tuple[dict[str, Any], Any, Any]:
    """Evaluate a classifier and return metrics, predictions, and probabilities."""
    text_column = "clean_text"
    label_column = config["data"]["label_column"]
    missing_columns = [column for column in (text_column, label_column) if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{split_name} data is missing required column(s): {missing_columns}")

    y_true = df[label_column]
    text_values = df[text_column].astype(str)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(text_values)
    else:
        y_proba = None

    if threshold is None:
        y_pred = model.predict(text_values)
    else:
        if y_proba is None:
            raise ValueError("Custom threshold evaluation requires a model that supports predict_proba")
        y_pred = (y_proba[:, 1] >= threshold).astype(int)

    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    per_label = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        "split": split_name,
        "decision_threshold": float(threshold) if threshold is not None else None,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "negative_precision": float(per_label[0][0]),
        "negative_recall": float(per_label[1][0]),
        "negative_f1": float(per_label[2][0]),
        "positive_precision": float(per_label[0][1]),
        "positive_recall": float(per_label[1][1]),
        "positive_f1": float(per_label[2][1]),
        "confusion_matrix": matrix.astype(int).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["negative", "positive"],
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics, y_pred, y_proba


def save_metrics(metrics: dict[str, Any], path: Path) -> None:
    """Save classification metrics to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def save_predictions(df: pd.DataFrame, y_pred: Any, y_proba: Any, output_path: Path) -> None:
    """Save prediction details to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if y_proba is None:
        negative_probability = [None] * len(df)
        positive_probability = [None] * len(df)
    else:
        negative_probability = y_proba[:, 0]
        positive_probability = y_proba[:, 1]

    predictions = pd.DataFrame(
        {
            "text": df["text"].astype(str),
            "clean_text": df["clean_text"].astype(str),
            "actual_sentiment": df["sentiment"].astype(int),
            "predicted_sentiment": y_pred,
            "negative_probability": negative_probability,
            "positive_probability": positive_probability,
        }
    )
    predictions.to_csv(output_path, index=False)


def plot_confusion_matrix(y_true: Any, y_pred: Any, output_path: Path, title: str) -> None:
    """Plot and save a confusion matrix image with matplotlib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["negative", "positive"])
    ax.set_yticks([0, 1], labels=["negative", "positive"])

    threshold = matrix.max() / 2 if matrix.max() else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] > threshold else "black"
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
