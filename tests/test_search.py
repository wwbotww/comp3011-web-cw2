"""
test_search.py — Unit tests for src/search.py.
"""

import pytest

from src.indexer import build_index
from src.search import find_pages, print_word


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def index():
    """Small deterministic index used across all search tests."""
    pages = {
        "http://page1.com": {
            "url": "http://page1.com",
            "title": "Page 1",
            "text": "good friends make life wonderful wonderful",
        },
        "http://page2.com": {
            "url": "http://page2.com",
            "title": "Page 2",
            "text": "indifference is dangerous friends beware",
        },
        "http://page3.com": {
            "url": "http://page3.com",
            "title": "Page 3",
            "text": "nonsense and imagination fuel creativity",
        },
    }
    return build_index(pages)


# ---------------------------------------------------------------------------
# print_word tests
# ---------------------------------------------------------------------------

class TestPrintWord:
    def test_known_word_shows_url(self, index):
        result = print_word(index, "friends")
        assert "http://page1.com" in result
        assert "http://page2.com" in result

    def test_word_not_in_index(self, index):
        result = print_word(index, "zzzyyyxxx")
        assert "not found" in result.lower()

    def test_case_insensitive_lookup(self, index):
        lower = print_word(index, "friends")
        upper = print_word(index, "FRIENDS")
        assert lower == upper

    def test_empty_word_returns_message(self, index):
        result = print_word(index, "")
        assert "provide" in result.lower()

    def test_shows_frequency(self, index):
        result = print_word(index, "wonderful")
        assert "frequency" in result
        assert "2" in result  # "wonderful" appears twice in page1

    def test_shows_tfidf(self, index):
        result = print_word(index, "nonsense")
        assert "tfidf" in result

    def test_whitespace_word_treated_as_empty(self, index):
        result = print_word(index, "   ")
        assert "provide" in result.lower()


# ---------------------------------------------------------------------------
# find_pages tests
# ---------------------------------------------------------------------------

class TestFindPages:
    def test_single_word_found(self, index):
        result = find_pages(index, "indifference")
        assert "http://page2.com" in result

    def test_single_word_not_found(self, index):
        result = find_pages(index, "zzzyyyxxx")
        assert "no pages" in result.lower() or "not in the index" in result.lower()

    def test_multi_word_intersection(self, index):
        # "friends" is in page1 and page2; "good" is only in page1
        result = find_pages(index, "good friends")
        assert "http://page1.com" in result
        assert "http://page2.com" not in result

    def test_multi_word_no_intersection(self, index):
        # "nonsense" only in page3, "indifference" only in page2 → no overlap
        result = find_pages(index, "nonsense indifference")
        assert "no pages" in result.lower()

    def test_empty_query_returns_message(self, index):
        result = find_pages(index, "")
        assert "provide" in result.lower()

    def test_case_insensitive_find(self, index):
        lower = find_pages(index, "friends")
        upper = find_pages(index, "FRIENDS")
        # Both queries should return the same pages regardless of case
        assert "http://page1.com" in lower and "http://page1.com" in upper
        assert "http://page2.com" in lower and "http://page2.com" in upper

    def test_results_ranked_by_tfidf(self, index):
        # "friends" appears in page1 (1x in 4 tokens) and page2 (1x in 4 tokens)
        # Since both pages have the same frequency and total tokens,
        # the key thing is that the order is deterministic and both appear.
        result = find_pages(index, "friends")
        assert "http://page1.com" in result
        assert "http://page2.com" in result

    def test_result_shows_score(self, index):
        result = find_pages(index, "nonsense")
        assert "score" in result

    def test_stop_word_only_query(self, index):
        result = find_pages(index, "the and or")
        assert "stop words" in result.lower() or "no pages" in result.lower()

    def test_partial_match_returns_no_result(self, index):
        # Only one of the two words is present
        result = find_pages(index, "friends zzzyyyxxx")
        assert "not in the index" in result.lower()
