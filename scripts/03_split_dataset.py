import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_pipeline.config import load_config  # noqa: E402
from sentiment_pipeline.data_loader import load_dataset  # noqa: E402
from sentiment_pipeline.logger import setup_logger  # noqa: E402
from sentiment_pipeline.paths import build_paths  # noqa: E402
from sentiment_pipeline.splitter import split_dataset  # noqa: E402


def _json_ready(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def main() -> None:
    """Split the cleaned dataset and save train, validation, and test CSVs."""
    config = load_config()
    paths = build_paths(config)
    logger = setup_logger(paths["logs_dir"])

    cleaned_path = paths["processed_dir"] / "twitter_sentiment_clean.csv"
    cleaned_df = load_dataset(cleaned_path)
    train_df, validation_df, test_df, report = split_dataset(cleaned_df, config)

    train_df.to_csv(paths["processed_dir"] / "train.csv", index=False)
    validation_df.to_csv(paths["processed_dir"] / "validation.csv", index=False)
    test_df.to_csv(paths["processed_dir"] / "test.csv", index=False)

    with (paths["reports_dir"] / "split_report.json").open("w", encoding="utf-8") as file:
        json.dump(_json_ready(report), file, indent=2)

    logger.info("Saved train, validation, and test splits to %s", paths["processed_dir"])
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Test rows: {len(test_df)}")


if __name__ == "__main__":
    main()
