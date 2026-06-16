import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_pipeline.config import load_config  # noqa: E402
from sentiment_pipeline.data_loader import load_dataset  # noqa: E402
from sentiment_pipeline.experiment_runner import (  # noqa: E402
    run_model_experiments,
    save_experiment_results,
    select_best_model,
    train_and_save_best_model,
)
from sentiment_pipeline.logger import setup_logger  # noqa: E402
from sentiment_pipeline.paths import build_paths  # noqa: E402


def main() -> None:
    """Compare TF-IDF model experiments using validation data only."""
    config = load_config()
    paths = build_paths(config)
    logger = setup_logger(paths["logs_dir"])

    train_df = load_dataset(paths["processed_dir"] / "train.csv")
    validation_df = load_dataset(paths["processed_dir"] / "validation.csv")

    results_df = run_model_experiments(train_df, validation_df, config)
    comparison_path = paths["project_root"] / config["experiments"]["comparison_report_path"]
    save_experiment_results(results_df, comparison_path)

    best_config = select_best_model(results_df, metric="macro_f1")
    _, best_metrics = train_and_save_best_model(train_df, validation_df, config, best_config)

    logger.info("Saved model comparison report to %s", comparison_path)
    print("Model comparison ranked by validation macro F1")
    print(results_df.to_string(index=False))
    print("\nBest model")
    print(f"Experiment: {best_config['experiment_name']}")
    print(f"Model: {best_config['model_name']}")
    print(f"TF-IDF variant: {best_config['tfidf_variant']}")
    print(f"Validation macro F1: {best_config['macro_f1']:.4f}")
    print(f"Best model saved to: {best_metrics['best_model_path']}")

    hybrid_rows = results_df[results_df["tfidf_variant"] == "real_word_char_hybrid"]
    unigram_bigram_rows = results_df[results_df["tfidf_variant"] == "unigram_bigram"]
    if not hybrid_rows.empty and not unigram_bigram_rows.empty:
        best_hybrid = hybrid_rows["macro_f1"].max()
        best_unigram_bigram = unigram_bigram_rows["macro_f1"].max()
        if best_hybrid > best_unigram_bigram:
            print(f"real_word_char_hybrid beats unigram_bigram by {best_hybrid - best_unigram_bigram:.4f} macro F1")
        else:
            print(f"real_word_char_hybrid trails unigram_bigram by {best_unigram_bigram - best_hybrid:.4f} macro F1")


if __name__ == "__main__":
    main()
