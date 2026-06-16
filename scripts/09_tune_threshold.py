import sys
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_pipeline.config import load_config  # noqa: E402
from sentiment_pipeline.data_loader import load_dataset  # noqa: E402
from sentiment_pipeline.logger import setup_logger  # noqa: E402
from sentiment_pipeline.paths import build_paths  # noqa: E402
from sentiment_pipeline.threshold_optimizer import (  # noqa: E402
    evaluate_thresholds,
    save_threshold_config,
    save_threshold_results,
    select_best_threshold,
)


def main() -> None:
    """Tune the positive-class decision threshold on validation data only."""
    config = load_config()
    paths = build_paths(config)
    logger = setup_logger(paths["logs_dir"])

    best_model_path = paths["project_root"] / config["experiments"]["best_model_path"]
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model not found: {best_model_path}. Run scripts/06_compare_models.py first.")

    model = joblib.load(best_model_path)
    validation_df = load_dataset(paths["processed_dir"] / "validation.csv")
    thresholds = [round(value, 2) for value in np.arange(0.30, 0.701, 0.01)]
    results_df = evaluate_thresholds(model, validation_df, thresholds)
    best_threshold = select_best_threshold(results_df, metric="macro_f1")

    tuning_path = paths["project_root"] / config["experiments"]["threshold_tuning_path"]
    threshold_path = paths["project_root"] / config["experiments"]["threshold_path"]
    threshold_metrics_path = paths["project_root"] / config["experiments"]["threshold_metrics_path"]
    save_threshold_results(results_df, tuning_path)
    save_threshold_config(best_threshold["threshold"], best_threshold, threshold_path)
    save_threshold_config(best_threshold["threshold"], best_threshold, threshold_metrics_path)

    default_rows = results_df[results_df["threshold"].round(2) == 0.50]
    default_macro_f1 = default_rows.iloc[0]["macro_f1"] if not default_rows.empty else None

    logger.info("Saved threshold tuning results to %s", tuning_path)
    if default_macro_f1 is not None:
        print(f"Default threshold 0.50 macro F1: {default_macro_f1:.4f}")
    print(f"Best threshold: {best_threshold['threshold']:.2f}")
    print(f"Best macro F1: {best_threshold['macro_f1']:.4f}")
    print(f"False positives: {int(best_threshold['false_positives'])}")
    print(f"False negatives: {int(best_threshold['false_negatives'])}")


if __name__ == "__main__":
    main()
