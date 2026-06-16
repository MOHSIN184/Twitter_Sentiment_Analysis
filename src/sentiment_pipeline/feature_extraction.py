from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(config: dict[str, Any]) -> TfidfVectorizer:
    """Build a TF-IDF vectorizer from pipeline configuration."""
    tfidf_config = config.get("model", {}).get("tfidf", {})
    ngram_range = tfidf_config.get("ngram_range", [1, 1])

    if not isinstance(ngram_range, (list, tuple)) or len(ngram_range) != 2:
        raise ValueError("model.tfidf.ngram_range must be a two-item list or tuple")

    return TfidfVectorizer(
        max_features=tfidf_config.get("max_features"),
        ngram_range=tuple(int(value) for value in ngram_range),
        min_df=tfidf_config.get("min_df", 1),
        max_df=tfidf_config.get("max_df", 1.0),
        lowercase=bool(tfidf_config.get("lowercase", True)),
        strip_accents=tfidf_config.get("strip_accents"),
    )
