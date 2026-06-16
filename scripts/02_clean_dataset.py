import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_pipeline.config import load_config  # noqa: E402
from sentiment_pipeline.data_loader import load_dataset  # noqa: E402
from sentiment_pipeline.logger import setup_logger  # noqa: E402
from sentiment_pipeline.paths import build_paths  # noqa: E402
from sentiment_pipeline.preprocessing import preprocess_dataset  # noqa: E402


def _json_ready(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return None
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items() if not isinstance(item, pd.DataFrame)}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def main() -> None:
    """Clean the raw dataset and save the processed dataset and report."""
    config = load_config()
    paths = build_paths(config)
    logger = setup_logger(paths["logs_dir"])

    raw_df = load_dataset(paths["raw_data"])
    cleaned_df, report = preprocess_dataset(raw_df, config)

    cleaned_path = paths["processed_dir"] / "twitter_sentiment_clean.csv"
    cleaned_df.to_csv(cleaned_path, index=False)

    duplicate_report = report.get("duplicate_report", {})
    conflicting_rows = duplicate_report.get("conflicting_duplicate_rows")
    if isinstance(conflicting_rows, pd.DataFrame) and not conflicting_rows.empty:
        conflicting_rows.to_csv(paths["reports_dir"] / "conflicting_duplicates.csv", index=False)

    report_path = paths["reports_dir"] / "cleaning_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(_json_ready(report), file, indent=2)

    logger.info("Saved cleaned dataset to %s", cleaned_path)
    print(f"Rows before cleaning: {len(raw_df)}")
    print(f"Rows after cleaning: {len(cleaned_df)}")


if __name__ == "__main__":
    main()
