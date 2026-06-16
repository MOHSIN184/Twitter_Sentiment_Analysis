from typing import Any

import pandas as pd

from sentiment_pipeline.text_cleaner import make_duplicate_key


def remove_exact_duplicates(df: pd.DataFrame, text_column: str, label_column: str) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicate rows based on text and label."""
    before = len(df)
    cleaned = df.drop_duplicates(subset=[text_column, label_column], keep="first").copy()
    return cleaned, int(before - len(cleaned))


def find_conflicting_duplicate_texts(
    df: pd.DataFrame,
    duplicate_key_column: str,
    label_column: str,
) -> pd.DataFrame:
    """Return rows whose duplicate key appears with more than one label."""
    label_counts = df.groupby(duplicate_key_column)[label_column].nunique(dropna=False)
    conflicting_keys = label_counts[label_counts > 1].index
    return df[df[duplicate_key_column].isin(conflicting_keys)].copy()


def remove_conflicting_duplicate_texts(
    df: pd.DataFrame,
    duplicate_key_column: str,
    label_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove all rows whose duplicate key has conflicting labels."""
    conflicting_rows = find_conflicting_duplicate_texts(df, duplicate_key_column, label_column)
    if conflicting_rows.empty:
        return df.copy(), conflicting_rows
    cleaned = df[~df[duplicate_key_column].isin(conflicting_rows[duplicate_key_column])].copy()
    return cleaned, conflicting_rows


def remove_duplicate_texts_same_label(
    df: pd.DataFrame,
    duplicate_key_column: str,
    label_column: str,
) -> tuple[pd.DataFrame, int]:
    """For duplicate keys with the same label, keep the first row only."""
    before = len(df)
    cleaned = df.drop_duplicates(subset=[duplicate_key_column, label_column], keep="first").copy()
    return cleaned, int(before - len(cleaned))


def handle_duplicates(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove exact, conflicting, and same-label duplicate texts."""
    rows_before = len(df)
    working, exact_removed = remove_exact_duplicates(df, text_column, label_column)
    working = working.copy()
    working["duplicate_key"] = working[text_column].apply(make_duplicate_key)

    without_conflicts, conflicting_rows = remove_conflicting_duplicate_texts(working, "duplicate_key", label_column)
    conflict_group_count = int(conflicting_rows["duplicate_key"].nunique()) if not conflicting_rows.empty else 0

    without_same_label_duplicates, same_label_removed = remove_duplicate_texts_same_label(
        without_conflicts,
        "duplicate_key",
        label_column,
    )

    report = {
        "rows_before": int(rows_before),
        "exact_duplicate_rows_removed": int(exact_removed),
        "conflicting_duplicate_text_groups": conflict_group_count,
        "conflicting_duplicate_rows_removed": int(len(conflicting_rows)),
        "same_label_duplicate_rows_removed": int(same_label_removed),
        "rows_after": int(len(without_same_label_duplicates)),
        "conflicting_duplicate_rows": conflicting_rows,
    }
    return without_same_label_duplicates.copy(), report
