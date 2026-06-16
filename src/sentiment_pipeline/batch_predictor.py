from pathlib import Path
from typing import Any

import pandas as pd


PREDICTION_COLUMNS = [
    "text",
    "cleaned_text",
    "predicted_label",
    "predicted_sentiment",
    "decision_threshold",
    "negative_probability",
    "positive_probability",
]


def predict_dataframe(df: pd.DataFrame, predictor: Any, text_column: str = "text") -> pd.DataFrame:
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
    return output_df[PREDICTION_COLUMNS]


def save_batch_predictions(predictions_df: pd.DataFrame, output_path: Path) -> None:
    """Save batch predictions to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)


def predict_csv(input_path: Path, output_path: Path, predictor: Any, text_column: str = "text") -> pd.DataFrame:
    """Load a CSV, predict sentiment for its text column, and save predictions."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        raise ValueError(f"Could not read input CSV at {input_path}: {exc}") from exc

    predictions_df = predict_dataframe(df, predictor, text_column=text_column)
    save_batch_predictions(predictions_df, output_path)
    return predictions_df
