import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_pipeline.duplicate_handler import (  # noqa: E402
    find_conflicting_duplicate_texts,
    handle_duplicates,
    remove_exact_duplicates,
)
from sentiment_pipeline.text_cleaner import make_duplicate_key  # noqa: E402


def test_exact_duplicate_removal() -> None:
    df = pd.DataFrame({"text": ["Great", "Great", "Bad"], "sentiment": [1, 1, 0]})
    cleaned, removed = remove_exact_duplicates(df, "text", "sentiment")
    assert removed == 1
    assert len(cleaned) == 2


def test_conflicting_duplicate_detection() -> None:
    df = pd.DataFrame({"text": ["Great", " great ", "Bad"], "sentiment": [1, 0, 0]})
    df["duplicate_key"] = df["text"].apply(make_duplicate_key)
    conflicts = find_conflicting_duplicate_texts(df, "duplicate_key", "sentiment")
    assert len(conflicts) == 2
    assert set(conflicts["sentiment"]) == {0, 1}


def test_same_text_with_same_label_keeps_one_row() -> None:
    df = pd.DataFrame({"text": ["Great", " great ", "Bad"], "sentiment": [1, 1, 0]})
    cleaned, report = handle_duplicates(df, "text", "sentiment")
    assert len(cleaned) == 2
    assert report["same_label_duplicate_rows_removed"] == 1


def test_same_text_with_opposite_labels_removes_all_conflicting_rows() -> None:
    df = pd.DataFrame({"text": ["Great", " great ", "Bad"], "sentiment": [1, 0, 0]})
    cleaned, report = handle_duplicates(df, "text", "sentiment")
    assert cleaned["text"].tolist() == ["Bad"]
    assert report["conflicting_duplicate_text_groups"] == 1
    assert report["conflicting_duplicate_rows_removed"] == 2
