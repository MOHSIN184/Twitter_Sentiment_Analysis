from pathlib import Path
from typing import Any

import pandas as pd


def generate_error_analysis(
    model: Any,
    df: pd.DataFrame,
    config: dict[str, Any],
    split_name: str,
) -> pd.DataFrame:
    """Generate a row-level error analysis DataFrame for misclassified examples."""
    required_columns = ["text", "clean_text", config["data"]["label_column"]]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{split_name} data is missing required column(s): {missing_columns}")

    label_column = config["data"]["label_column"]
    y_pred = model.predict(df["clean_text"].astype(str))
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(df["clean_text"].astype(str))
        negative_probability = y_proba[:, 0]
        positive_probability = y_proba[:, 1]
        confidence = y_proba.max(axis=1)
    else:
        negative_probability = [None] * len(df)
        positive_probability = [None] * len(df)
        confidence = [None] * len(df)

    analysis_df = pd.DataFrame(
        {
            "text": df["text"].astype(str),
            "clean_text": df["clean_text"].astype(str),
            "actual_sentiment": df[label_column].astype(int),
            "predicted_sentiment": y_pred.astype(int),
            "confidence": confidence,
            "negative_probability": negative_probability,
            "positive_probability": positive_probability,
        }
    )
    error_df = analysis_df[analysis_df["actual_sentiment"] != analysis_df["predicted_sentiment"]].copy()
    error_df["error_type"] = error_df.apply(
        lambda row: "false_positive" if row["actual_sentiment"] == 0 and row["predicted_sentiment"] == 1 else "false_negative",
        axis=1,
    )
    error_df["text_length"] = error_df["clean_text"].str.len().astype(int)
    error_df["word_count"] = error_df["clean_text"].str.split().str.len().astype(int)

    columns = [
        "text",
        "clean_text",
        "actual_sentiment",
        "predicted_sentiment",
        "error_type",
        "confidence",
        "negative_probability",
        "positive_probability",
        "text_length",
        "word_count",
    ]
    return error_df[columns].reset_index(drop=True)


def save_error_analysis(error_df: pd.DataFrame, output_path: Path) -> None:
    """Save error analysis rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_df.to_csv(output_path, index=False)


def get_top_misclassified_examples(error_df: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    """Return the top misclassified examples sorted by confidence when available."""
    if "confidence" in error_df.columns and error_df["confidence"].notna().any():
        return error_df.sort_values("confidence", ascending=False).head(n).reset_index(drop=True)
    return error_df.head(n).reset_index(drop=True)
