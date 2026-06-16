from typing import Any

import pandas as pd


def validate_required_columns(df: pd.DataFrame, text_column: str, label_column: str) -> None:
    """Validate that required text and label columns exist."""
    missing_columns = [column for column in (text_column, label_column) if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required column(s): {missing_columns}")


def validate_no_missing_values(df: pd.DataFrame, text_column: str, label_column: str) -> None:
    """Validate that text and label columns do not contain missing values."""
    missing_counts = df[[text_column, label_column]].isna().sum()
    missing = {column: int(count) for column, count in missing_counts.items() if count > 0}
    if missing:
        raise ValueError(f"Missing values found in required columns: {missing}")


def validate_allowed_labels(df: pd.DataFrame, label_column: str, allowed_labels: list[int]) -> None:
    """Validate that labels are limited to the configured allowed values."""
    labels = set(df[label_column].dropna().unique().tolist())
    allowed = set(allowed_labels)
    invalid_labels = sorted(labels - allowed)
    if invalid_labels:
        raise ValueError(f"Invalid label(s) found in {label_column}: {invalid_labels}. Allowed: {sorted(allowed)}")


def validate_text_column(df: pd.DataFrame, text_column: str) -> None:
    """Validate that the text column contains non-empty string-like values."""
    empty_mask = df[text_column].astype(str).str.strip().eq("")
    if empty_mask.any():
        raise ValueError(f"Empty text rows found in {text_column}: {int(empty_mask.sum())}")


def validate_dataset(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Validate the dataset and return a JSON-friendly profiling summary."""
    data_config = config["data"]
    text_column = data_config["text_column"]
    label_column = data_config["label_column"]
    allowed_labels = data_config["allowed_labels"]

    validate_required_columns(df, text_column, label_column)
    validate_no_missing_values(df, text_column, label_column)
    validate_allowed_labels(df, label_column, allowed_labels)
    validate_text_column(df, text_column)

    text_lengths = df[text_column].astype(str).str.len()
    return {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "column_names": [str(column) for column in df.columns],
        "missing_values": {column: int(value) for column, value in df.isna().sum().items()},
        "label_distribution": {
            str(label): int(count)
            for label, count in df[label_column].value_counts(dropna=False).sort_index().items()
        },
        "duplicate_exact_rows": int(df.duplicated().sum()),
        "empty_text_rows": int(df[text_column].astype(str).str.strip().eq("").sum()),
        "average_text_length": float(text_lengths.mean()) if len(text_lengths) else 0.0,
        "median_text_length": float(text_lengths.median()) if len(text_lengths) else 0.0,
        "max_text_length": int(text_lengths.max()) if len(text_lengths) else 0,
    }
