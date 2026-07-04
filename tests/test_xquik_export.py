import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_pipeline.xquik_export import load_xquik_export_dataframe  # noqa: E402


def test_load_xquik_nested_json_export(tmp_path: Path) -> None:
    export_path = tmp_path / "xquik.json"
    export_path.write_text(
        json.dumps({"data": [{"full_text": "I love this release", "username": "demo", "like_count": 7}]}),
        encoding="utf-8",
    )

    df = load_xquik_export_dataframe(export_path)

    assert df.to_dict(orient="records") == [{"text": "I love this release", "user": "demo", "favorite_count": 7}]


def test_load_xquik_jsonl_export(tmp_path: Path) -> None:
    export_path = tmp_path / "xquik.jsonl"
    export_path.write_text('{"text":"First"}\n{"tweet":"Second","retweet_count":3}\n', encoding="utf-8")

    df = load_xquik_export_dataframe(export_path)

    assert df["text"].tolist() == ["First", "Second"]
    assert df["retweet_count"].dropna().tolist() == [3.0]


def test_rejects_xquik_export_without_text(tmp_path: Path) -> None:
    export_path = tmp_path / "xquik.json"
    export_path.write_text(json.dumps({"data": [{"id": "1"}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="rows with text"):
        load_xquik_export_dataframe(export_path)
