import logging
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

LOGGER = logging.getLogger("sentiment_pipeline")


def _ngram_range(config: dict[str, Any]) -> tuple[int, int]:
    ngram_range = config.get("ngram_range", [1, 1])
    if not isinstance(ngram_range, (list, tuple)) or len(ngram_range) != 2:
        raise ValueError("TF-IDF ngram_range must be a two-item list or tuple")
    return tuple(int(value) for value in ngram_range)


def _build_tfidf_vectorizer(tfidf_config: dict[str, Any]) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=tfidf_config.get("analyzer", "word"),
        max_features=tfidf_config.get("max_features"),
        ngram_range=_ngram_range(tfidf_config),
        min_df=tfidf_config.get("min_df", 1),
        max_df=tfidf_config.get("max_df", 1.0),
        lowercase=bool(tfidf_config.get("lowercase", True)),
        strip_accents=tfidf_config.get("strip_accents"),
    )


def build_classifier(model_name: str, model_config: dict[str, Any]) -> Any:
    """Build a supported classifier from its experiment configuration."""
    LOGGER.info("Building classifier: %s", model_name)

    if model_name == "logistic_regression":
        return LogisticRegression(
            class_weight=model_config.get("class_weight"),
            max_iter=int(model_config.get("max_iter", 1000)),
            solver=model_config.get("solver", "liblinear"),
        )

    if model_name == "linear_svc":
        return LinearSVC(
            class_weight=model_config.get("class_weight"),
            max_iter=int(model_config.get("max_iter", 5000)),
        )

    if model_name == "sgd_classifier":
        return SGDClassifier(
            loss=model_config.get("loss", "log_loss"),
            penalty=model_config.get("penalty", "l2"),
            alpha=float(model_config.get("alpha", 0.0001)),
            class_weight=model_config.get("class_weight"),
            max_iter=int(model_config.get("max_iter", 1000)),
            random_state=int(model_config.get("random_state", 42)),
        )

    if model_name == "complement_nb":
        return ComplementNB(alpha=float(model_config.get("alpha", 1.0)))

    raise ValueError(f"Unknown model name: {model_name}")


def build_text_features(tfidf_config: dict[str, Any]) -> Any:
    """Build word TF-IDF or a real word plus character TF-IDF feature union."""
    feature_type = tfidf_config.get("feature_type", "word")

    if feature_type == "word":
        return _build_tfidf_vectorizer(tfidf_config)

    if feature_type == "word_char_hybrid":
        if "word" not in tfidf_config or "char" not in tfidf_config:
            raise ValueError("word_char_hybrid requires word and char configurations")
        word_features = _build_tfidf_vectorizer(tfidf_config["word"])
        char_config = dict(tfidf_config["char"])
        char_config["analyzer"] = char_config.get("analyzer", "char_wb")
        char_features = _build_tfidf_vectorizer(char_config)
        return FeatureUnion(
            transformer_list=[
                ("word_tfidf", word_features),
                ("char_tfidf", char_features),
            ]
        )

    raise ValueError(f"Unknown TF-IDF feature_type: {feature_type}")


def build_experiment_pipeline(tfidf_config: dict[str, Any], classifier: Any) -> Pipeline:
    """Build a text-feature plus classifier experiment pipeline."""
    return Pipeline(steps=[("features", build_text_features(tfidf_config)), ("classifier", classifier)])
