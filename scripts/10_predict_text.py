import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_pipeline.config import load_config  # noqa: E402
from sentiment_pipeline.paths import build_paths  # noqa: E402
from sentiment_pipeline.predictor import SentimentPredictor  # noqa: E402


def main() -> None:
    """Predict sentiment for one text string from the command line."""
    parser = argparse.ArgumentParser(description="Predict Twitter sentiment for one text string.")
    parser.add_argument("--text", required=True, help="Text to classify.")
    args = parser.parse_args()

    config = load_config()
    paths = build_paths(config)
    model_path = paths["project_root"] / config["experiments"]["best_model_path"]
    threshold_path = paths["project_root"] / config["experiments"]["threshold_path"]
    predictor = SentimentPredictor(model_path=model_path, threshold_path=threshold_path, config=config)
    prediction = predictor.predict_one(args.text)

    print(f"Original text: {prediction['original_text']}")
    print(f"Cleaned text: {prediction['cleaned_text']}")
    print(f"Predicted label: {prediction['predicted_label']}")
    print(f"Predicted sentiment: {prediction['predicted_sentiment']}")
    print(f"Negative probability: {prediction['negative_probability']}")
    print(f"Positive probability: {prediction['positive_probability']}")
    print(f"Threshold used: {prediction['decision_threshold']}")


if __name__ == "__main__":
    main()
