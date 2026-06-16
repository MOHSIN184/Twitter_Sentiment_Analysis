import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_pipeline.error_analysis import generate_error_analysis  # noqa: E402


class DummyModel:
    def predict(self, texts):
        return np.array([1, 0, 1])

    def predict_proba(self, texts):
        return np.array([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])


def test_error_analysis_detects_false_positives_and_false_negatives() -> None:
    df = pd.DataFrame(
        {
            "text": ["bad", "good", "great"],
            "clean_text": ["bad", "good", "great"],
            "sentiment": [0, 1, 1],
        }
    )
    config = {"data": {"label_column": "sentiment"}}

    error_df = generate_error_analysis(DummyModel(), df, config, "validation")

    assert len(error_df) == 2
    assert error_df["error_type"].tolist() == ["false_positive", "false_negative"]
    assert error_df["confidence"].tolist() == [0.8, 0.7]
