"""
search.py — Search logic implementing the print and find commands.

Both functions accept the inverted index dict produced by
``indexer.build_index()`` and return formatted result strings.
They never raise exceptions for bad input — every code path returns a
user-friendly message.
"""

from typing import Any

from src.indexer import tokenise

# Internal key that holds index metadata — must be excluded from searches.
_META_KEY = "__meta__"


# ---------------------------------------------------------------------------
# print command
# ---------------------------------------------------------------------------

def print_word(index: dict[str, Any], word: str) -> str:
    """Return a formatted inverted-index entry for a single *word*.

    The lookup is case-insensitive: *word* is lowered and stripped before
    being matched against the index.

    Args:
        index: The loaded inverted index (from ``indexer.load_index``).
        word:  The word to look up.  Must be a single token.

    Returns:
        Human-readable multi-line string showing the posting list (URL,
        frequency, positions, TF, TF-IDF) for the word — or a "not found" /
        "invalid input" message.
    """
    normalised = word.lower().strip()

    if not normalised:
        return "Please provide a word to look up."

    if normalised == _META_KEY or normalised not in index:
        return f'Word "{normalised}" was not found in the index.'

    postings = index[normalised]
    lines: list[str] = [
        f'Word: "{normalised}" — found in {len(postings)} page(s)\n'
    ]

    for url, stats in sorted(postings.items()):
        pos = stats["positions"]
        pos_display = str(pos[:10]) + (" ..." if len(pos) > 10 else "")

        lines.append(f"  {url}")
        lines.append(f"    frequency : {stats['frequency']}")
        lines.append(f"    positions : {pos_display}")
        lines.append(f"    tf        : {stats['tf']:.6f}")
        lines.append(f"    tfidf     : {stats['tfidf']:.6f}")
        lines.append("")

    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# find command
# ---------------------------------------------------------------------------

def find_pages(index: dict[str, Any], query: str) -> str:
    """Find pages containing ALL query words, ranked by summed TF-IDF.

    Multi-word queries use AND semantics: only pages that contain
    **every** query token are returned.  Duplicate tokens in the query
    are collapsed so that a word is not counted twice.

    Args:
        index: The loaded inverted index.
        query: One or more space-separated search terms.

    Returns:
        Formatted result string listing matching pages with scores,
        or a descriptive message when no results are found.
    """
    if not query or not query.strip():
        return "Please provide at least one search term."

    tokens = tokenise(query)
    if not tokens:
        return (
            "Query contained only stop words or punctuation "
            "— please try different terms."
        )

    # Deduplicate tokens while preserving order for display purposes
    seen: set[str] = set()
    unique_tokens: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique_tokens.append(t)

    # Look up each unique token in the index
    missing: list[str] = []
    posting_sets: list[set[str]] = []

    for token in unique_tokens:
        if token == _META_KEY or token not in index:
            missing.append(token)
        else:
            posting_sets.append(set(index[token].keys()))

    if missing:
        return (
            "No pages found — the following word(s) are not in the index: "
            + ", ".join(missing)
        )

    # Intersect all posting sets (AND semantics)
    common_urls = posting_sets[0]
    for pset in posting_sets[1:]:
        common_urls &= pset

    if not common_urls:
        return f'No pages found containing all of: {", ".join(unique_tokens)}'

    # Score each page by the sum of TF-IDF across all unique query tokens
    scores: dict[str, float] = {}
    for url in common_urls:
        scores[url] = sum(
            index[t][url]["tfidf"] for t in unique_tokens
        )

    ranked = sorted(common_urls, key=lambda u: scores[u], reverse=True)

    lines: list[str] = [
        f'Results for "{query.strip()}" — {len(ranked)} page(s) found:\n'
    ]
    for rank, url in enumerate(ranked, start=1):
        lines.append(f"  {rank}. {url}  [score: {scores[url]:.4f}]")

    return "\n".join(lines)
