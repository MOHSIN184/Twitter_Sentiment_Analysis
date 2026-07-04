import json
from pathlib import Path
from typing import Any

import pandas as pd


TEXT_FIELDS = ("text", "full_text", "tweet", "content", "body")
LIST_FIELDS = ("data", "tweets", "results", "items")
METADATA_FIELDS = (
    "id",
    "tweet_id",
    "user",
    "username",
    "author_username",
    "favorite_count",
    "like_count",
    "retweet_count",
    "reply_count",
    "created_at",
    "url",
)


def _json_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if isinstance(payload, dict):
        for field in LIST_FIELDS:
            value = payload.get(field)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

    raise ValueError("Xquik export must contain a list of result objects.")


def _extract_text(item: dict[str, Any]) -> str:
    for field in TEXT_FIELDS:
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_row(item: dict[str, Any]) -> dict[str, Any]:
    text = _extract_text(item)
    if not text:
        return {}

    row: dict[str, Any] = {"text": text}
    for field in METADATA_FIELDS:
        value = item.get(field)
        if value not in (None, "") and field not in row:
            row[field] = value

    if "username" in row and "user" not in row:
        row["user"] = row.pop("username")
    if "author_username" in row and "user" not in row:
        row["user"] = row.pop("author_username")
    if "like_count" in row and "favorite_count" not in row:
        row["favorite_count"] = row.pop("like_count")

    return row


def load_xquik_export_dataframe(path: Path) -> pd.DataFrame:
    """Load an exported Xquik result file into a prediction-ready DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Xquik export not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        items = _json_rows(json.loads(path.read_text(encoding="utf-8")))
    elif suffix == ".jsonl":
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".csv":
        items = pd.read_csv(path).to_dict(orient="records")
    else:
        raise ValueError("Xquik export must be a JSON, JSONL, or CSV file.")

    rows = [row for item in items if isinstance(item, dict) for row in [_normalize_row(item)] if row]
    if not rows:
        raise ValueError("Xquik export does not contain any rows with text.")

    return pd.DataFrame(rows)
