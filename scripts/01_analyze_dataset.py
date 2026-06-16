import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_pipeline.config import load_config  # noqa: E402
from sentiment_pipeline.data_loader import load_dataset  # noqa: E402
from sentiment_pipeline.data_validator import validate_dataset  # noqa: E402
from sentiment_pipeline.logger import setup_logger  # noqa: E402
from sentiment_pipeline.paths import build_paths  # noqa: E402


def _json_ready(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def main() -> None:
    """Analyze the raw dataset and save summary reports and figures."""
    config = load_config()
    paths = build_paths(config)
    logger = setup_logger(paths["logs_dir"])

    df = load_dataset(paths["raw_data"])
    summary = validate_dataset(df, config)

    label_column = config["data"]["label_column"]
    text_column = config["data"]["text_column"]
    class_distribution = df[label_column].value_counts().sort_index().rename_axis(label_column).reset_index(name="count")

    summary_path = paths["reports_dir"] / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(_json_ready(summary), file, indent=2)

    class_distribution_path = paths["reports_dir"] / "class_distribution.csv"
    class_distribution.to_csv(class_distribution_path, index=False)

    plt.figure(figsize=(7, 5))
    plt.bar(class_distribution[label_column].astype(str), class_distribution["count"], color=["#D95F59", "#3A8F6B"])
    plt.title("Class Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.savefig(paths["figures_dir"] / "class_distribution.png", dpi=150)
    plt.close()

    text_lengths = df[text_column].astype(str).str.len()
    plt.figure(figsize=(8, 5))
    plt.hist(text_lengths, bins=50, color="#4E79A7", edgecolor="white")
    plt.title("Text Length Distribution")
    plt.xlabel("Characters")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.savefig(paths["figures_dir"] / "text_length_distribution.png", dpi=150)
    plt.close()

    logger.info("Saved dataset analysis reports to %s", paths["reports_dir"])
    print(json.dumps(_json_ready(summary), indent=2))


if __name__ == "__main__":
    main()
