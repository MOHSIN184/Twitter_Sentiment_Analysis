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
from sentiment_pipeline.model_trainer import save_model, train_baseline_model  # noqa: E402
from sentiment_pipeline.paths import build_paths  # noqa: E402


def _warn_if_low_quality(metrics: dict) -> None:
    if metrics["accuracy"] < 0.75 or metrics["macro_f1"] < 0.75:
        print("WARNING: validation score is lower than expected; inspect data, cleaning, and label balance.")


def main() -> None:
    """Train and validate the TF-IDF plus Logistic Regression baseline."""
    config = load_config()
    paths = build_paths(config)
    logger = setup_logger(paths["logs_dir"])

    baseline_config = config["model"]["baseline"]
    model_path = paths["project_root"] / baseline_config["model_path"]
    metrics_path = paths["project_root"] / baseline_config["metrics_path"]
    predictions_path = paths["project_root"] / baseline_config["validation_predictions_path"]
    confusion_matrix_path = paths["figures_dir"] / "validation_confusion_matrix.png"

    train_df = load_dataset(paths["processed_dir"] / "train.csv")
    validation_df = load_dataset(paths["processed_dir"] / "validation.csv")

    model, validation_metrics = train_baseline_model(train_df, validation_df, config)
    save_model(model, model_path)

    validation_metrics, y_pred, y_proba = evaluate_classification_model(model, validation_df, config, "validation")
    save_metrics(validation_metrics, metrics_path)
    save_predictions(validation_df, y_pred, y_proba, predictions_path)
    plot_confusion_matrix(
        validation_df[config["data"]["label_column"]],
        y_pred,
        confusion_matrix_path,
        "Validation Confusion Matrix",
    )

    logger.info("Saved validation metrics to %s", metrics_path)
    print("Validation metrics")
    print(f"Accuracy: {validation_metrics['accuracy']:.4f}")
    print(f"Macro F1: {validation_metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {validation_metrics['weighted_f1']:.4f}")
    _warn_if_low_quality(validation_metrics)


if __name__ == "__main__":
    main()
