import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_pipeline.text_cleaner import (  # noqa: E402
    clean_hashtags,
    clean_tweet_text,
    decode_html_entities,
    normalize_whitespace,
    replace_mentions,
    replace_urls,
)


def _config() -> dict:
    return {
        "cleaning": {
            "decode_html_entities": True,
            "normalize_whitespace": True,
            "replace_urls": True,
            "replace_mentions": True,
            "keep_hashtag_words": True,
        }
    }


def test_decode_html_entities() -> None:
    assert decode_html_entities("Tom &amp; Jerry") == "Tom & Jerry"


def test_replace_urls() -> None:
    assert replace_urls("Read https://example.com and www.example.org") == "Read <URL> and <URL>"


def test_replace_mentions() -> None:
    assert replace_mentions("Thanks @alice and @bob_123") == "Thanks <USER> and <USER>"


def test_clean_hashtags() -> None:
    assert clean_hashtags("Love #Happy #AI") == "Love Happy AI"


def test_normalize_whitespace() -> None:
    assert normalize_whitespace("  hello\t\tworld\nagain  ") == "hello world again"


def test_clean_tweet_text_keeps_punctuation_and_emoji() -> None:
    text = "Hi @user!!! #Happy \U0001f60a https://example.com"
    assert clean_tweet_text(text, _config()) == "Hi <USER>!!! Happy \U0001f60a <URL>"
