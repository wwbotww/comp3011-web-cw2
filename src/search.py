"""
search.py — Search logic implementing the print and find commands.

The index passed to all functions is the dict produced by indexer.build_index()
(and persisted/loaded by indexer.save/load_index).
"""

from typing import Any

from src.indexer import tokenise


def print_word(index: dict[str, Any], word: str) -> str:
    """Return a formatted string showing the inverted index entry for *word*.

    Normalises *word* to lowercase before lookup (case-insensitive).

    Args:
        index: The loaded inverted index.
        word: The word to look up.

    Returns:
        A multi-line human-readable string.  Always returns a string — even
        when the word is not in the index the message is returned rather than
        raising an exception.
    """
    normalised = word.lower().strip()
    if not normalised:
        return "Please provide a word to look up."

    if normalised not in index or normalised == "__meta__":
        return f'Word "{normalised}" was not found in the index.'

    postings = index[normalised]
    lines: list[str] = [f'Word: "{normalised}" — found in {len(postings)} page(s)\n']

    for url, stats in sorted(postings.items()):
        lines.append(f"  {url}")
        lines.append(f"    frequency : {stats['frequency']}")
        lines.append(f"    positions : {stats['positions'][:10]}"
                     + (" ..." if len(stats["positions"]) > 10 else ""))
        lines.append(f"    tf        : {stats['tf']:.6f}")
        lines.append(f"    tfidf     : {stats['tfidf']:.6f}")
        lines.append("")

    return "\n".join(lines).rstrip()


def find_pages(index: dict[str, Any], query: str) -> str:
    """Return pages that contain ALL words in the query, ranked by TF-IDF.

    Multi-word queries are treated as a conjunction (AND): only pages that
    contain every query word are returned.

    Args:
        index: The loaded inverted index.
        query: One or more space-separated search terms.

    Returns:
        A formatted result string.  Returns a message string (not an
        exception) when the query is empty or yields no results.
    """
    if not query.strip():
        return "Please provide at least one search term."

    tokens = tokenise(query)
    if not tokens:
        return "Query contained only stop words or punctuation — please try different terms."

    # Locate the posting list for each token; short-circuit on first miss
    posting_sets: list[set[str]] = []
    missing: list[str] = []

    for token in tokens:
        if token not in index or token == "__meta__":
            missing.append(token)
        else:
            posting_sets.append(set(index[token].keys()))

    if missing:
        return f'No pages found — the following word(s) are not in the index: {", ".join(missing)}'

    # Intersection of all posting sets
    common_urls = posting_sets[0]
    for pset in posting_sets[1:]:
        common_urls &= pset

    if not common_urls:
        return f'No pages found containing all of: {", ".join(tokens)}'

    # Rank by summed TF-IDF across all query words
    def score(url: str) -> float:
        return sum(index[t][url]["tfidf"] for t in tokens if url in index[t])

    ranked = sorted(common_urls, key=score, reverse=True)

    lines: list[str] = [
        f'Results for "{query.strip()}" — {len(ranked)} page(s) found:\n'
    ]
    for rank, url in enumerate(ranked, start=1):
        s = score(url)
        lines.append(f"  {rank}. {url}  [score: {s:.4f}]")

    return "\n".join(lines)
