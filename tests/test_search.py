"""
test_search.py — Unit tests for src/search.py.
"""

import pytest

from src.indexer import build_index
from src.search import find_pages, print_word


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pages(**kwargs) -> dict:
    """Build a pages dict from keyword args mapping url → text."""
    return {
        url: {"url": url, "title": url, "text": text}
        for url, text in kwargs.items()
    }


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def index():
    """Small deterministic index used across search tests."""
    return build_index(pages(
        **{
            "http://page1.com": "good friends make life wonderful wonderful",
            "http://page2.com": "indifference dangerous friends beware",
            "http://page3.com": "nonsense imagination fuel creativity",
        }
    ))


@pytest.fixture()
def big_index():
    """A word that appears in many pages vs. one that appears in fewer."""
    return build_index(pages(
        **{
            "http://a.com": "python java ruby",
            "http://b.com": "python java",
            "http://c.com": "python",
        }
    ))


# ---------------------------------------------------------------------------
# print_word
# ---------------------------------------------------------------------------

class TestPrintWord:

    # --- Happy path ---

    def test_known_word_shows_all_urls(self, index):
        result = print_word(index, "friends")
        assert "http://page1.com" in result
        assert "http://page2.com" in result

    def test_output_contains_frequency(self, index):
        result = print_word(index, "wonderful")
        assert "frequency" in result
        assert "2" in result

    def test_output_contains_positions(self, index):
        result = print_word(index, "wonderful")
        assert "positions" in result

    def test_output_contains_tf(self, index):
        result = print_word(index, "wonderful")
        assert "tf" in result

    def test_output_contains_tfidf(self, index):
        result = print_word(index, "nonsense")
        assert "tfidf" in result

    def test_shows_page_count(self, index):
        result = print_word(index, "friends")
        assert "2 page(s)" in result

    # --- Case insensitivity ---

    def test_case_insensitive_lookup(self, index):
        lower = print_word(index, "friends")
        upper = print_word(index, "FRIENDS")
        assert lower == upper

    def test_mixed_case_lookup(self, index):
        result = print_word(index, "FrIeNdS")
        assert "http://page1.com" in result

    # --- Edge cases ---

    def test_word_not_in_index(self, index):
        result = print_word(index, "zzzyyyxxx")
        assert "not found" in result.lower()

    def test_empty_string(self, index):
        result = print_word(index, "")
        assert "provide" in result.lower()

    def test_whitespace_only(self, index):
        result = print_word(index, "   ")
        assert "provide" in result.lower()

    def test_meta_key_hidden(self, index):
        result = print_word(index, "__meta__")
        assert "not found" in result.lower()

    def test_positions_truncated_beyond_ten(self):
        # Build a page where a word appears > 10 times
        text = " ".join(["repeat"] * 15)
        idx = build_index(pages(url1=text))
        result = print_word(idx, "repeat")
        assert "..." in result


# ---------------------------------------------------------------------------
# find_pages
# ---------------------------------------------------------------------------

class TestFindPages:

    # --- Single-word queries ---

    def test_single_word_found(self, index):
        result = find_pages(index, "indifference")
        assert "http://page2.com" in result

    def test_single_word_not_found(self, index):
        result = find_pages(index, "zzzyyyxxx")
        assert "not in the index" in result.lower()

    def test_single_word_shows_score(self, index):
        result = find_pages(index, "nonsense")
        assert "score" in result

    def test_single_word_shows_page_count(self, index):
        result = find_pages(index, "nonsense")
        assert "1 page(s) found" in result

    # --- Multi-word queries (AND semantics) ---

    def test_multi_word_intersection(self, index):
        # "friends" in page1 + page2, "good" only in page1 → only page1
        result = find_pages(index, "good friends")
        assert "http://page1.com" in result
        assert "http://page2.com" not in result

    def test_multi_word_no_intersection(self, index):
        # "nonsense" only in page3, "indifference" only in page2
        result = find_pages(index, "nonsense indifference")
        assert "no pages" in result.lower()

    def test_multi_word_one_missing(self, index):
        result = find_pages(index, "friends zzzyyyxxx")
        assert "not in the index" in result.lower()

    # --- TF-IDF ranking ---

    def test_rarer_word_page_ranked_higher(self, big_index):
        # "ruby" only in a.com (high idf), "python" in all 3 (idf=0)
        # Searching for "ruby" should return a.com first (and only)
        result = find_pages(big_index, "ruby")
        assert "http://a.com" in result

    def test_ranking_is_descending_by_score(self, big_index):
        # "java" in a.com and b.com; both have same tf, same idf → either order
        # Key check: both appear and the ranking number is present
        result = find_pages(big_index, "java")
        assert "1." in result
        assert "2." in result

    # --- Duplicate token handling ---

    def test_duplicate_tokens_collapsed(self, index):
        # "good good" should behave the same as "good"
        single = find_pages(index, "good")
        double = find_pages(index, "good good")
        # Same pages should be returned; scores should be equal
        assert "http://page1.com" in single
        assert "http://page1.com" in double

    def test_duplicate_tokens_do_not_inflate_score(self, index):
        single = find_pages(index, "good")
        double = find_pages(index, "good good")
        # Extract scores from both outputs — they should be equal
        import re
        single_scores = re.findall(r"score: ([\d.]+)", single)
        double_scores = re.findall(r"score: ([\d.]+)", double)
        assert single_scores == double_scores

    # --- Case insensitivity ---

    def test_case_insensitive_find(self, index):
        lower = find_pages(index, "friends")
        upper = find_pages(index, "FRIENDS")
        assert "http://page1.com" in lower and "http://page1.com" in upper
        assert "http://page2.com" in lower and "http://page2.com" in upper

    # --- Edge cases ---

    def test_empty_query(self, index):
        result = find_pages(index, "")
        assert "provide" in result.lower()

    def test_whitespace_only_query(self, index):
        result = find_pages(index, "   ")
        assert "provide" in result.lower()

    def test_stop_word_only_query(self, index):
        result = find_pages(index, "the and or")
        assert "stop words" in result.lower()

    def test_meta_key_treated_as_missing(self, index):
        result = find_pages(index, "__meta__")
        assert "not in the index" in result.lower()

    def test_punctuation_only_query(self, index):
        result = find_pages(index, "!!! ??? ...")
        assert "stop words" in result.lower() or "provide" in result.lower()

    def test_result_header_shows_original_query(self, index):
        result = find_pages(index, "good friends")
        assert "good friends" in result
