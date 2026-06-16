from pathlib import Path
from typing import Any, Optional, Union

from sentiment_pipeline.config import load_config
from sentiment_pipeline.model_trainer import load_model
from sentiment_pipeline.threshold_optimizer import load_threshold_config
from sentiment_pipeline.text_cleaner import clean_tweet_text


class SentimentPredictor:
    """Reusable predictor for the trained Twitter sentiment baseline model."""

    def __init__(
        self,
        model_path: Union[Path, str],
        threshold_path: Optional[Union[Path, str]] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """Load a trained model and optional tuned decision threshold."""
        self.model_path = Path(model_path)
        self.config = config if config is not None else load_config()
        self.model = load_model(self.model_path)
        self.decision_threshold = None
        if threshold_path is not None and Path(threshold_path).exists():
            threshold_config = load_threshold_config(Path(threshold_path))
            self.decision_threshold = float(threshold_config["threshold"])

    def _positive_probability_index(self) -> int:
        classes = list(getattr(self.model, "classes_", [0, 1]))
        return classes.index(1) if 1 in classes else 1

    def predict_one(self, text: str) -> dict[str, Any]:
        """Predict sentiment for one raw text string."""
        cleaned_text = clean_tweet_text(text, self.config)
        negative_probability = None
        positive_probability = None

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba([cleaned_text])[0]
            positive_index = self._positive_probability_index()
            negative_index = 1 - positive_index
            negative_probability = float(probabilities[negative_index])
            positive_probability = float(probabilities[positive_index])

        if self.decision_threshold is not None and positive_probability is not None:
            predicted_label = 1 if positive_probability >= self.decision_threshold else 0
        else:
            predicted_label = int(self.model.predict([cleaned_text])[0])

        return {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "predicted_label": predicted_label,
            "predicted_sentiment": "positive" if predicted_label == 1 else "negative",
            "decision_threshold": self.decision_threshold,
            "negative_probability": negative_probability,
            "positive_probability": positive_probability,
        }

    def predict_many(self, texts: list[str]) -> list[dict[str, Any]]:
        """Predict sentiment for multiple raw text strings."""
        return [self.predict_one(text) for text in texts]
