"""
indexer.py — Inverted index builder, storage and loader.

Inverted index schema:
    {
        "<word>": {
            "<url>": {
                "frequency": int,       # occurrences of word in this page
                "positions": [int, ...], # 0-based word positions
                "tf": float             # term frequency = freq / total_words
            },
            ...
        },
        ...
    }

Metadata stored alongside the index:
    {
        "meta": {
            "total_pages": int,
            "total_words": int,
            "built_at": str  # ISO-8601 timestamp
        }
    }
"""

import json
import math
import re
import string
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = Path(__file__).parent.parent / "data" / "index.json"

# Words to skip when building the index (common English stop words).
# Keeping the list short — search engines typically do index stop words,
# but excluding the most frequent ones improves index size and query speed.
STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "it", "as", "be",
    "was", "are", "were", "been", "has", "have", "had", "not",
    "that", "this", "which", "who", "what", "i", "you", "he",
    "she", "we", "they", "me", "him", "her", "us", "them",
})


def tokenise(text: str) -> list[str]:
    """Tokenise text into lowercase, punctuation-stripped words.

    Args:
        text: Raw text string.

    Returns:
        List of tokens (lowercase, no punctuation, stop words removed).
    """
    text = text.lower()
    text = re.sub(r"[" + re.escape(string.punctuation) + r"]", " ", text)
    tokens = [t for t in text.split() if t and t not in STOP_WORDS]
    return tokens


def build_index(pages: dict[str, dict]) -> dict[str, Any]:
    """Build an inverted index with TF-IDF from crawled page data.

    Args:
        pages: Mapping of URL → {"url": str, "title": str, "text": str},
               as returned by crawler.crawl().

    Returns:
        Index dict with "meta" key and one key per unique word.
    """
    # Intermediate structure: word → {url → {"frequency": int, "positions": [int]}}
    raw: dict[str, dict[str, dict]] = {}

    for url, page in pages.items():
        tokens = tokenise(page.get("text", ""))
        total = len(tokens)
        if total == 0:
            continue

        for pos, word in enumerate(tokens):
            raw.setdefault(word, {})
            raw[word].setdefault(url, {"frequency": 0, "positions": [], "tf": 0.0})
            raw[word][url]["frequency"] += 1
            raw[word][url]["positions"].append(pos)

        # Calculate TF for each word in this document
        for word in set(tokens):
            if url in raw.get(word, {}):
                raw[word][url]["tf"] = round(raw[word][url]["frequency"] / total, 6)

    # Calculate IDF and attach TF-IDF to each posting
    total_pages = len(pages)
    index: dict[str, Any] = {}

    for word, postings in raw.items():
        df = len(postings)  # document frequency
        idf = math.log(total_pages / df) if df > 0 else 0.0

        index[word] = {}
        for url, stats in postings.items():
            index[word][url] = {
                "frequency": stats["frequency"],
                "positions": stats["positions"],
                "tf": stats["tf"],
                "tfidf": round(stats["tf"] * idf, 6),
            }

    # Attach metadata
    index["__meta__"] = {
        "total_pages": total_pages,
        "total_words": len(index) - 1,  # exclude __meta__
        "built_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "[Index] Built %d unique words across %d pages.",
        len(index) - 1,
        total_pages,
    )
    return index


def save_index(index: dict[str, Any], path: Path = DEFAULT_INDEX_PATH) -> None:
    """Serialise the index to a JSON file.

    Args:
        index: The inverted index dict (as returned by build_index).
        path: Destination file path (created with parent dirs if needed).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=2)
    logger.info("[Index] Saved to %s", path)


def load_index(path: Path = DEFAULT_INDEX_PATH) -> dict[str, Any]:
    """Load an inverted index from a JSON file.

    Args:
        path: Path to the JSON index file.

    Returns:
        The loaded index dict.

    Raises:
        FileNotFoundError: If the index file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Index file not found at '{path}'. "
            "Run the 'build' command first."
        )
    with open(path, "r", encoding="utf-8") as fh:
        index = json.load(fh)
    meta = index.get("__meta__", {})
    logger.info(
        "[Index] Loaded %d unique words across %d pages from %s",
        meta.get("total_words", "?"),
        meta.get("total_pages", "?"),
        path,
    )
    return index
