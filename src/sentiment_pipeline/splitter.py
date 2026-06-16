from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def _label_distribution(df: pd.DataFrame, label_column: str) -> dict[str, int]:
    return {str(label): int(count) for label, count in df[label_column].value_counts().sort_index().items()}


def _assert_no_clean_text_overlap(train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_texts = set(train_df["clean_text"])
    validation_texts = set(validation_df["clean_text"])
    test_texts = set(test_df["clean_text"])
    if train_texts & validation_texts or train_texts & test_texts or validation_texts & test_texts:
        raise ValueError("clean_text overlap detected between train, validation, and test splits")


def split_dataset(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split a cleaned dataset into train, validation, and test DataFrames."""
    label_column = config["data"]["label_column"]
    split_config = config["split"]
    train_size = float(split_config["train_size"])
    validation_size = float(split_config["validation_size"])
    test_size = float(split_config["test_size"])
    random_state = int(config["project"]["random_state"])

    if abs((train_size + validation_size + test_size) - 1.0) > 1e-8:
        raise ValueError("Split sizes must sum to 1.0")
    if "clean_text" not in df.columns:
        raise ValueError("Expected a clean_text column before splitting")

    stratify = df[label_column] if split_config.get("stratify", True) else None

    try:
        train_df, temp_df = train_test_split(
            df,
            train_size=train_size,
            random_state=random_state,
            stratify=stratify,
        )
        temp_validation_fraction = validation_size / (validation_size + test_size)
        temp_stratify = temp_df[label_column] if split_config.get("stratify", True) else None
        validation_df, test_df = train_test_split(
            temp_df,
            train_size=temp_validation_fraction,
            random_state=random_state,
            stratify=temp_stratify,
        )
    except ValueError as exc:
        raise ValueError(f"Unable to split dataset with the configured settings: {exc}") from exc

    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    _assert_no_clean_text_overlap(train_df, validation_df, test_df)

    report = {
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "train_label_distribution": _label_distribution(train_df, label_column),
        "validation_label_distribution": _label_distribution(validation_df, label_column),
        "test_label_distribution": _label_distribution(test_df, label_column),
    }
    return train_df, validation_df, test_df, report
