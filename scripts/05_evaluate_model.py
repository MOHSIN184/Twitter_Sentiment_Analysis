import sys
from pathlib import Path

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
from sentiment_pipeline.model_trainer import load_model  # noqa: E402
from sentiment_pipeline.paths import build_paths  # noqa: E402


def _warn_if_low_quality(metrics: dict) -> None:
    if metrics["accuracy"] < 0.75 or metrics["macro_f1"] < 0.75:
        print("WARNING: test score is lower than expected; inspect data, cleaning, and label balance.")


def main() -> None:
    """Evaluate the saved baseline model on the held-out test split."""
    config = load_config()
    paths = build_paths(config)
    logger = setup_logger(paths["logs_dir"])

    baseline_config = config["model"]["baseline"]
    model_path = paths["project_root"] / baseline_config["model_path"]
    predictions_path = paths["project_root"] / baseline_config["test_predictions_path"]
    metrics_path = paths["reports_dir"] / "test_metrics.json"
    confusion_matrix_path = paths["figures_dir"] / "test_confusion_matrix.png"

    model = load_model(model_path)
    test_df = load_dataset(paths["processed_dir"] / "test.csv")

    test_metrics, y_pred, y_proba = evaluate_classification_model(model, test_df, config, "test")
    save_metrics(test_metrics, metrics_path)
    save_predictions(test_df, y_pred, y_proba, predictions_path)
    plot_confusion_matrix(
        test_df[config["data"]["label_column"]],
        y_pred,
        confusion_matrix_path,
        "Test Confusion Matrix",
    )

    logger.info("Saved test metrics to %s", metrics_path)
    print("Test metrics")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {test_metrics['weighted_f1']:.4f}")
    _warn_if_low_quality(test_metrics)


if __name__ == "__main__":
    main()
