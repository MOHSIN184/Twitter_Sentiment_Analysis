import sys
from pathlib import Path

import pytest
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_pipeline.model_registry import build_classifier, build_experiment_pipeline, build_text_features  # noqa: E402


def test_build_classifier_returns_known_classifiers() -> None:
    assert isinstance(build_classifier("logistic_regression", {}), LogisticRegression)
    assert isinstance(build_classifier("linear_svc", {}), LinearSVC)
    assert isinstance(build_classifier("sgd_classifier", {}), SGDClassifier)
    assert isinstance(build_classifier("complement_nb", {}), ComplementNB)


def test_build_classifier_raises_for_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unknown model name"):
        build_classifier("not_a_model", {})


def test_build_experiment_pipeline_returns_pipeline() -> None:
    tfidf_config = {
        "max_features": 100,
        "ngram_range": [1, 2],
        "min_df": 1,
        "max_df": 1.0,
        "lowercase": True,
        "strip_accents": "unicode",
    }
    pipeline = build_experiment_pipeline(tfidf_config, LogisticRegression())
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["features", "classifier"]
    assert pipeline.named_steps["features"].ngram_range == (1, 2)


def test_build_text_features_returns_feature_union_for_real_hybrid() -> None:
    tfidf_config = {
        "feature_type": "word_char_hybrid",
        "word": {
            "max_features": 100,
            "ngram_range": [1, 2],
            "min_df": 1,
            "max_df": 1.0,
            "lowercase": True,
            "strip_accents": "unicode",
        },
        "char": {
            "analyzer": "char_wb",
            "max_features": 50,
            "ngram_range": [3, 5],
            "min_df": 1,
            "max_df": 1.0,
            "lowercase": True,
        },
    }
    features = build_text_features(tfidf_config)
    assert list(features.transformer_list)[0][0] == "word_tfidf"
    assert list(features.transformer_list)[1][0] == "char_tfidf"
