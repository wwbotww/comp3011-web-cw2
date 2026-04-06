"""
indexer.py — Inverted index builder, storage and loader.

Inverted index schema
---------------------
{
    "<word>": {
        "<url>": {
            "frequency": int,        # occurrences of <word> in this page
            "positions": [int, ...], # 0-based token positions
            "tf":        float,      # term frequency = frequency / total_tokens
            "tfidf":     float       # tf * idf  (idf = log(N / df))
        },
        ...
    },
    ...
    "__meta__": {
        "total_pages":   int,  # pages passed to build_index
        "indexed_pages": int,  # pages that had indexable content
        "total_words":   int,  # unique words in the index
        "built_at":      str   # ISO-8601 UTC timestamp
    }
}

IDF formula:  idf(w) = log( indexed_pages / df(w) )
  where df(w) = number of indexed pages containing word w.
  When every indexed page contains w, idf = 0 and tfidf = 0.
"""

import json
import math
import re
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = Path(__file__).parent.parent / "data" / "index.json"

# Common English stop words excluded from the index.
# Standard search engines do index stop words, but omitting the most
# frequent ones keeps the index compact and improves query relevance.
STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "it", "as", "be",
    "was", "are", "were", "been", "has", "have", "had", "not",
    "that", "this", "which", "who", "what", "i", "you", "he",
    "she", "we", "they", "me", "him", "her", "us", "them",
})


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenise(text: str) -> list[str]:
    """Convert raw text into a list of clean, index-ready tokens.

    Steps applied in order:
      1. Lowercase the entire string (case-insensitive search requirement).
      2. Replace every punctuation character with a space.
      3. Split on whitespace and drop empty fragments.
      4. Remove stop words.

    Args:
        text: Raw text extracted from a web page.

    Returns:
        Ordered list of lowercase tokens with punctuation and stop words
        removed.  May be empty if *text* contains only stop words or
        punctuation.
    """
    text = text.lower()
    # Replace any character that is not alphanumeric, underscore, or whitespace.
    # Using [^\w\s] instead of string.punctuation so that Unicode punctuation
    # (e.g. em dash —, curly quotes "", ellipsis …) is also stripped.
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t and t not in STOP_WORDS]


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------

def build_index(pages: dict[str, dict]) -> dict[str, Any]:
    """Build an inverted index with TF-IDF scores from crawled page data.

    For each word found across all pages the index stores:
      - per-page frequency, token positions, TF, and TF-IDF score.

    IDF is calculated after all pages have been processed so that the
    document-frequency (df) denominator is known.

    Args:
        pages: Mapping of URL → {"url": str, "title": str, "text": str}
               as returned by ``crawler.crawl()``.  Pages whose text
               tokenises to an empty list are silently skipped and do
               not contribute to IDF calculations.

    Returns:
        Index dict containing one key per unique word and a ``"__meta__"``
        entry with build statistics.
    """
    total_pages = len(pages)

    # word → url → {"frequency": int, "positions": [int]}
    raw: dict[str, dict[str, dict]] = {}

    # Count only pages that actually contributed tokens
    indexed_pages = 0

    for url, page in pages.items():
        tokens = tokenise(page.get("text", ""))
        if not tokens:
            continue

        indexed_pages += 1
        total_tokens = len(tokens)

        # Accumulate frequency and positions per (word, url)
        word_stats: dict[str, dict] = {}
        for pos, word in enumerate(tokens):
            if word not in word_stats:
                word_stats[word] = {"frequency": 0, "positions": []}
            word_stats[word]["frequency"] += 1
            word_stats[word]["positions"].append(pos)

        # Store TF and merge into the global raw index
        for word, stats in word_stats.items():
            tf = round(stats["frequency"] / total_tokens, 6)
            raw.setdefault(word, {})[url] = {
                "frequency": stats["frequency"],
                "positions": stats["positions"],
                "tf": tf,
            }

    # --- IDF + TF-IDF pass ---
    # Use indexed_pages for IDF so that the denominator matches the actual
    # corpus size (pages with no tokens are excluded from the corpus).
    index: dict[str, Any] = {}

    for word, postings in raw.items():
        df = len(postings)
        idf = math.log(indexed_pages / df) if indexed_pages > 0 and df > 0 else 0.0

        index[word] = {}
        for url, stats in postings.items():
            index[word][url] = {
                "frequency": stats["frequency"],
                "positions": stats["positions"],
                "tf":        stats["tf"],
                "tfidf":     round(stats["tf"] * idf, 6),
            }

    index["__meta__"] = {
        "total_pages":   total_pages,
        "indexed_pages": indexed_pages,
        "total_words":   len(index),   # computed before __meta__ is added
        "built_at":      datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "[Index] Built %d unique words from %d/%d page(s).",
        len(index) - 1,
        indexed_pages,
        total_pages,
    )
    return index


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_index(index: dict[str, Any], path: Path = DEFAULT_INDEX_PATH) -> None:
    """Serialise *index* to a UTF-8 JSON file at *path*.

    Parent directories are created automatically if they do not exist.

    Args:
        index: Inverted index dict as returned by :func:`build_index`.
        path:  Destination file path.  Defaults to ``data/index.json``
               relative to the project root.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=2)
    logger.info("[Index] Saved to %s", path)


def load_index(path: Path = DEFAULT_INDEX_PATH) -> dict[str, Any]:
    """Load and return an inverted index from a JSON file.

    Args:
        path: Path to the JSON index file.  Defaults to ``data/index.json``
              relative to the project root.

    Returns:
        The loaded inverted index dict.

    Raises:
        FileNotFoundError: If *path* does not exist.
        json.JSONDecodeError: If the file exists but is not valid JSON.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Index file not found at '{path}'. "
            "Run the 'build' command first to generate the index."
        )
    with open(path, "r", encoding="utf-8") as fh:
        index = json.load(fh)

    meta = index.get("__meta__", {})
    logger.info(
        "[Index] Loaded %s unique words from %d/%d page(s)  (built %s)",
        meta.get("total_words", "?"),
        meta.get("indexed_pages", meta.get("total_pages", "?")),
        meta.get("total_pages", "?"),
        meta.get("built_at", "unknown"),
    )
    return index
