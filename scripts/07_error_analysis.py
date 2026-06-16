import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_pipeline.config import load_config  # noqa: E402
from sentiment_pipeline.data_loader import load_dataset  # noqa: E402
from sentiment_pipeline.error_analysis import (  # noqa: E402
    generate_error_analysis,
    get_top_misclassified_examples,
    save_error_analysis,
)
from sentiment_pipeline.logger import setup_logger  # noqa: E402
from sentiment_pipeline.paths import build_paths  # noqa: E402


def main() -> None:
    """Generate validation error analysis for the selected best model."""
    config = load_config()
    paths = build_paths(config)
    logger = setup_logger(paths["logs_dir"])

    best_model_path = paths["project_root"] / config["experiments"]["best_model_path"]
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model not found: {best_model_path}. Run scripts/06_compare_models.py first.")

    model = joblib.load(best_model_path)
    validation_df = load_dataset(paths["processed_dir"] / "validation.csv")
    error_df = generate_error_analysis(model, validation_df, config, "validation")
    top_errors = get_top_misclassified_examples(error_df, n=100)

    error_path = paths["project_root"] / config["experiments"]["error_analysis_path"]
    top_errors_path = paths["project_root"] / config["experiments"]["top_errors_path"]
    save_error_analysis(error_df, error_path)
    save_error_analysis(top_errors, top_errors_path)

    false_positives = int((error_df["error_type"] == "false_positive").sum())
    false_negatives = int((error_df["error_type"] == "false_negative").sum())
    total_errors = int(len(error_df))
    error_rate = total_errors / len(validation_df) if len(validation_df) else 0.0

    logger.info("Saved validation error analysis to %s", error_path)
    print(f"Total validation rows: {len(validation_df)}")
    print(f"Total errors: {total_errors}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")
    print(f"Error rate: {error_rate:.4f}")


if __name__ == "__main__":
    main()
