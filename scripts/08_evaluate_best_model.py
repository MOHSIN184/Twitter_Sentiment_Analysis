import json
import sys
from pathlib import Path
from typing import Optional

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_pipeline.config import load_config  # noqa: E402
from sentiment_pipeline.data_loader import load_dataset  # noqa: E402
from sentiment_pipeline.logger import setup_logger  # noqa: E402
from sentiment_pipeline.model_evaluator import (  # noqa: E402
    evaluate_classification_model,
    plot_confusion_matrix,
    save_metrics,
    save_predictions,
)
from sentiment_pipeline.paths import build_paths  # noqa: E402
from sentiment_pipeline.threshold_optimizer import load_threshold_config  # noqa: E402


def _load_baseline_macro_f1(paths: dict, config: dict) -> Optional[float]:
    baseline_metrics_path = paths["reports_dir"] / "test_metrics.json"
    if not baseline_metrics_path.exists():
        return None
    with baseline_metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    return float(metrics.get("macro_f1")) if metrics.get("macro_f1") is not None else None


def main() -> None:
    """Evaluate the validation-selected best model on the held-out test split."""
    config = load_config()
    paths = build_paths(config)
    logger = setup_logger(paths["logs_dir"])

    best_model_path = paths["project_root"] / config["experiments"]["best_model_path"]
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model not found: {best_model_path}. Run scripts/06_compare_models.py first.")

    model = joblib.load(best_model_path)
    test_df = load_dataset(paths["processed_dir"] / "test.csv")
    threshold_path = paths["project_root"] / config["experiments"]["threshold_path"]
    threshold = None
    if threshold_path.exists():
        threshold = float(load_threshold_config(threshold_path)["threshold"])

    metrics, y_pred, y_proba = evaluate_classification_model(model, test_df, config, "test", threshold=threshold)

    output_dir = paths["project_root"] / config["experiments"]["output_dir"]
    metrics_path = output_dir / "final_best_model_test_metrics.json"
    predictions_path = output_dir / "final_best_model_test_predictions.csv"
    confusion_matrix_path = paths["figures_dir"] / "final_best_model_test_confusion_matrix.png"

    save_metrics(metrics, metrics_path)
    save_predictions(test_df, y_pred, y_proba, predictions_path)
    plot_confusion_matrix(
        test_df[config["data"]["label_column"]],
        y_pred,
        confusion_matrix_path,
        "Final Best Model Test Confusion Matrix",
    )

    baseline_macro_f1 = _load_baseline_macro_f1(paths, config)
    if baseline_macro_f1 is not None and metrics["macro_f1"] < baseline_macro_f1:
        print("WARNING: best model test macro F1 is lower than the baseline test macro F1.")

    logger.info("Saved final best model test metrics to %s", metrics_path)
    print("Final best model test metrics")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Threshold used: {threshold if threshold is not None else 'default model.predict()'}")


if __name__ == "__main__":
    main()
