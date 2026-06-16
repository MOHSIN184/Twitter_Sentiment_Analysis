import html
import re
from typing import Any

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
MENTION_PATTERN = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
WHITESPACE_PATTERN = re.compile(r"\s+")


def decode_html_entities(text: str) -> str:
    """Decode HTML entities in text."""
    return html.unescape(str(text))


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and strip leading/trailing spaces."""
    return WHITESPACE_PATTERN.sub(" ", str(text)).strip()


def replace_urls(text: str) -> str:
    """Replace URL-like spans with the ``<URL>`` token."""
    return URL_PATTERN.sub("<URL>", str(text))


def replace_mentions(text: str) -> str:
    """Replace Twitter mentions with the ``<USER>`` token."""
    return MENTION_PATTERN.sub("<USER>", str(text))


def clean_hashtags(text: str) -> str:
    """Convert hashtags into their word content while preserving casing."""
    return HASHTAG_PATTERN.sub(r"\1", str(text))


def clean_tweet_text(text: str, config: dict[str, Any]) -> str:
    """Clean tweet text using configured rules while preserving casing and punctuation."""
    cleaning_config = config.get("cleaning", {})
    cleaned = str(text)

    if cleaning_config.get("decode_html_entities", True):
        cleaned = decode_html_entities(cleaned)
    if cleaning_config.get("replace_urls", True):
        cleaned = replace_urls(cleaned)
    if cleaning_config.get("replace_mentions", True):
        cleaned = replace_mentions(cleaned)
    if cleaning_config.get("keep_hashtag_words", True):
        cleaned = clean_hashtags(cleaned)
    if cleaning_config.get("normalize_whitespace", True):
        cleaned = normalize_whitespace(cleaned)
    else:
        cleaned = cleaned.strip()

    return cleaned


def make_duplicate_key(text: str) -> str:
    """Create a normalized key for duplicate text detection."""
    return normalize_whitespace(decode_html_entities(str(text)).lower()).strip()
