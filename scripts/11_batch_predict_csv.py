import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_pipeline.batch_predictor import predict_csv  # noqa: E402
from sentiment_pipeline.config import load_config  # noqa: E402
from sentiment_pipeline.paths import build_paths  # noqa: E402
from sentiment_pipeline.predictor import SentimentPredictor  # noqa: E402


def main() -> None:
    """Run batch sentiment prediction for a CSV file."""
    parser = argparse.ArgumentParser(description="Predict sentiment for a CSV file with a text column.")
    parser.add_argument("--input", required=True, help="Input CSV path containing a text column.")
    parser.add_argument("--output", required=True, help="Output predictions CSV path.")
    args = parser.parse_args()

    config = load_config()
    paths = build_paths(config)
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_absolute():
        input_path = paths["project_root"] / input_path
    if not output_path.is_absolute():
        output_path = paths["project_root"] / output_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    predictor = SentimentPredictor(
        model_path=paths["project_root"] / config["inference"]["model_path"],
        threshold_path=paths["project_root"] / config["inference"]["threshold_path"],
        config=config,
    )
    predictions_df = predict_csv(input_path, output_path, predictor, text_column="text")
    print(f"Saved predictions to: {output_path}")
    print(f"Total rows predicted: {len(predictions_df)}")


if __name__ == "__main__":
    main()
