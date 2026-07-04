from pathlib import Path
from typing import Any

import pandas as pd

from sentiment_pipeline.xquik_export import load_xquik_export_dataframe


PREDICTION_COLUMNS = [
    "text",
    "cleaned_text",
    "predicted_label",
    "predicted_sentiment",
    "decision_threshold",
    "negative_probability",
    "positive_probability",
]


def predict_dataframe(
    df: pd.DataFrame,
    predictor: Any,
    text_column: str = "text",
    keep_source_columns: bool = False,
) -> pd.DataFrame:
    """Predict sentiment for all rows in a DataFrame."""
    if text_column not in df.columns:
        raise ValueError(f"Input data is missing required text column: {text_column}")

    predictions = [predictor.predict_one(str(text)) for text in df[text_column].fillna("")]
    output_df = pd.DataFrame(
        {
            "text": [prediction["original_text"] for prediction in predictions],
            "cleaned_text": [prediction["cleaned_text"] for prediction in predictions],
            "predicted_label": [prediction["predicted_label"] for prediction in predictions],
            "predicted_sentiment": [prediction["predicted_sentiment"] for prediction in predictions],
            "decision_threshold": [prediction["decision_threshold"] for prediction in predictions],
            "negative_probability": [prediction["negative_probability"] for prediction in predictions],
            "positive_probability": [prediction["positive_probability"] for prediction in predictions],
        }
    )
    output_df = output_df[PREDICTION_COLUMNS]
    if not keep_source_columns:
        return output_df

    source_columns = [column for column in df.columns if column not in output_df.columns]
    if not source_columns:
        return output_df

    return pd.concat([df[source_columns].reset_index(drop=True), output_df.reset_index(drop=True)], axis=1)


def save_batch_predictions(predictions_df: pd.DataFrame, output_path: Path) -> None:
    """Save batch predictions to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)


def predict_csv(
    input_path: Path,
    output_path: Path,
    predictor: Any,
    text_column: str = "text",
    keep_source_columns: bool = False,
) -> pd.DataFrame:
    """Load a CSV, predict sentiment for its text column, and save predictions."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        raise ValueError(f"Could not read input CSV at {input_path}: {exc}") from exc

    predictions_df = predict_dataframe(df, predictor, text_column=text_column, keep_source_columns=keep_source_columns)
    save_batch_predictions(predictions_df, output_path)
    return predictions_df


def predict_xquik_export(input_path: Path, output_path: Path, predictor: Any) -> pd.DataFrame:
    """Load an exported Xquik result file, predict sentiment, and save predictions."""
    df = load_xquik_export_dataframe(input_path)
    predictions_df = predict_dataframe(df, predictor, text_column="text", keep_source_columns=True)
    save_batch_predictions(predictions_df, output_path)
    return predictions_df
