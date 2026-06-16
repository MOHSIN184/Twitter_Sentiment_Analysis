import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_pipeline.model_evaluator import evaluate_classification_model  # noqa: E402
from sentiment_pipeline.threshold_optimizer import (  # noqa: E402
    evaluate_thresholds,
    load_threshold_config,
    save_threshold_config,
    select_best_threshold,
)


class ThresholdDummyModel:
    classes_ = np.array([0, 1])

    def predict(self, texts):
        return np.array([1 if index in (1, 2) else 0 for index, _ in enumerate(texts)])

    def predict_proba(self, texts):
        return np.array(
            [
                [0.80, 0.20],
                [0.55, 0.45],
                [0.40, 0.60],
                [0.30, 0.70],
            ]
        )


def _validation_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": ["a", "b", "c", "d"],
            "clean_text": ["a", "b", "c", "d"],
            "sentiment": [0, 0, 1, 1],
        }
    )


def test_threshold_evaluation_returns_expected_columns() -> None:
    results = evaluate_thresholds(ThresholdDummyModel(), _validation_df(), [0.4, 0.5])
    expected_columns = {
        "threshold",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "negative_f1",
        "positive_f1",
        "false_positives",
        "false_negatives",
    }
    assert expected_columns.issubset(results.columns)


def test_best_threshold_selection_returns_highest_macro_f1() -> None:
    results = pd.DataFrame({"threshold": [0.4, 0.5, 0.6], "macro_f1": [0.6, 0.9, 0.8]})
    best = select_best_threshold(results)
    assert best["threshold"] == 0.5


def test_threshold_config_can_be_saved_and_loaded(tmp_path: Path) -> None:
    path = tmp_path / "threshold.json"
    metrics = {
        "accuracy": 0.8,
        "macro_f1": 0.75,
        "weighted_f1": 0.78,
        "false_positives": 2,
        "false_negatives": 3,
    }
    save_threshold_config(0.47, metrics, path)
    loaded = load_threshold_config(path)
    assert loaded["threshold"] == 0.47
    assert loaded["selected_metric_value"] == 0.75


def test_custom_threshold_changes_predictions_correctly() -> None:
    config = {"data": {"label_column": "sentiment"}}
    metrics, predictions, _ = evaluate_classification_model(
        ThresholdDummyModel(),
        _validation_df(),
        config,
        "validation",
        threshold=0.65,
    )
    assert predictions.tolist() == [0, 0, 0, 1]
    assert metrics["decision_threshold"] == 0.65
