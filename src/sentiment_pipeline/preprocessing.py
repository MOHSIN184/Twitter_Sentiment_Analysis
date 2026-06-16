from typing import Any

import pandas as pd

from sentiment_pipeline.data_validator import validate_dataset
from sentiment_pipeline.duplicate_handler import handle_duplicates
from sentiment_pipeline.text_cleaner import clean_tweet_text, make_duplicate_key


def preprocess_dataset(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate, deduplicate, and clean the Twitter sentiment dataset."""
    validation_summary = validate_dataset(df, config)
    data_config = config["data"]
    text_column = data_config["text_column"]
    label_column = data_config["label_column"]

    working = df.copy()
    working["duplicate_key"] = working[text_column].apply(make_duplicate_key)
    deduplicated, duplicate_report = handle_duplicates(working, text_column, label_column)

    before_empty_filter = len(deduplicated)
    deduplicated = deduplicated.copy()
    deduplicated["clean_text"] = deduplicated[text_column].apply(lambda value: clean_tweet_text(value, config))
    cleaned = deduplicated[deduplicated["clean_text"].astype(str).str.strip().ne("")].copy()
    empty_clean_text_rows_removed = int(before_empty_filter - len(cleaned))
    before_clean_text_dedupe = len(cleaned)

    clean_text_label_counts = cleaned.groupby("clean_text")[label_column].nunique(dropna=False)
    conflicting_clean_texts = clean_text_label_counts[clean_text_label_counts > 1].index
    conflicting_clean_text_rows_removed = int(cleaned["clean_text"].isin(conflicting_clean_texts).sum())
    cleaned = cleaned[~cleaned["clean_text"].isin(conflicting_clean_texts)].copy()

    before_same_label_clean_text_dedupe = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["clean_text", label_column], keep="first").copy()
    same_label_clean_text_duplicates_removed = int(before_same_label_clean_text_dedupe - len(cleaned))

    final_df = cleaned[[label_column, text_column, "clean_text"]].reset_index(drop=True)
    report = {
        "rows_before": int(len(df)),
        "rows_after": int(len(final_df)),
        "empty_clean_text_rows_removed": empty_clean_text_rows_removed,
        "conflicting_clean_text_groups": int(len(conflicting_clean_texts)),
        "conflicting_clean_text_rows_removed": conflicting_clean_text_rows_removed,
        "same_label_clean_text_duplicates_removed": same_label_clean_text_duplicates_removed,
        "total_clean_text_duplicates_removed": int(before_clean_text_dedupe - len(cleaned)),
        "validation_summary": validation_summary,
        "duplicate_report": duplicate_report,
    }
    return final_df, report
