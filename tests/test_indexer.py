"""
test_indexer.py — Unit tests for src/indexer.py.
"""

import json
import math
from pathlib import Path

import pytest

from src.indexer import (
    STOP_WORDS,
    build_index,
    load_index,
    save_index,
    tokenise,
)


# ---------------------------------------------------------------------------
# tokenise tests
# ---------------------------------------------------------------------------

class TestTokenise:
    def test_lowercases_input(self):
        tokens = tokenise("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_strips_punctuation(self):
        tokens = tokenise("Hello, world! It's great.")
        assert "hello" in tokens
        assert "world" in tokens
        # punctuation-only fragments should not appear
        assert all(t.isalpha() or "_" in t for t in tokens)

    def test_removes_stop_words(self):
        tokens = tokenise("the quick brown fox")
        assert "the" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens

    def test_empty_string(self):
        assert tokenise("") == []

    def test_only_stop_words(self):
        assert tokenise("the and or but") == []

    def test_numbers_are_kept(self):
        tokens = tokenise("3 wise men")
        assert "3" in tokens

    def test_case_insensitive_stop_word_removal(self):
        tokens = tokenise("The quick fox")
        assert "the" not in tokens


# ---------------------------------------------------------------------------
# build_index tests
# ---------------------------------------------------------------------------

class TestBuildIndex:
    def _pages(self, **kwargs) -> dict:
        """Helper: build a pages dict from keyword args (url → text)."""
        return {url: {"url": url, "title": url, "text": text} for url, text in kwargs.items()}

    def test_basic_word_frequency(self):
        pages = self._pages(url1="good good bad")
        index = build_index(pages)
        assert "good" in index
        assert index["good"]["url1"]["frequency"] == 2

    def test_word_positions_recorded(self):
        pages = self._pages(url1="the quick brown fox quick")
        index = build_index(pages)
        # "quick" appears at positions 1 and 4 (0-based, after stop word removal shift)
        assert "quick" in index
        assert len(index["quick"]["url1"]["positions"]) == 2

    def test_tf_is_between_zero_and_one(self):
        pages = self._pages(url1="apple orange apple banana")
        index = build_index(pages)
        for word in ["apple", "orange", "banana"]:
            if word in index:
                tf = index[word]["url1"]["tf"]
                assert 0.0 < tf <= 1.0

    def test_tfidf_present(self):
        pages = self._pages(url1="python code", url2="java code")
        index = build_index(pages)
        assert "tfidf" in index["code"]["url1"]

    def test_meta_block_present(self):
        pages = self._pages(url1="hello world")
        index = build_index(pages)
        assert "__meta__" in index
        meta = index["__meta__"]
        assert meta["total_pages"] == 1
        assert "built_at" in meta

    def test_case_insensitive_indexing(self):
        pages = self._pages(url1="Good good GOOD")
        index = build_index(pages)
        assert "good" in index
        assert index["good"]["url1"]["frequency"] == 3

    def test_empty_pages(self):
        index = build_index({})
        assert index["__meta__"]["total_pages"] == 0

    def test_multi_page_df(self):
        pages = self._pages(url1="python java", url2="python ruby")
        index = build_index(pages)
        # "python" appears in both pages → df = 2
        # idf = log(2/2) = 0
        assert index["python"]["url1"]["tfidf"] == pytest.approx(0.0, abs=1e-6)

    def test_rare_word_higher_idf(self):
        pages = self._pages(url1="python java", url2="python ruby")
        index = build_index(pages)
        # "java" appears in only 1 of 2 pages, "python" in 2 of 2
        java_idf = math.log(2 / 1)
        assert index["java"]["url1"]["tfidf"] > index["python"]["url1"]["tfidf"]


# ---------------------------------------------------------------------------
# save_index / load_index tests
# ---------------------------------------------------------------------------

class TestIndexPersistence:
    def test_round_trip(self, tmp_path):
        pages = {"http://a.com": {"url": "http://a.com", "title": "A", "text": "hello world"}}
        index = build_index(pages)
        dest = tmp_path / "index.json"
        save_index(index, dest)
        loaded = load_index(dest)
        assert loaded == index

    def test_save_creates_parent_dirs(self, tmp_path):
        pages = {"http://a.com": {"url": "http://a.com", "title": "A", "text": "test"}}
        index = build_index(pages)
        dest = tmp_path / "sub" / "dir" / "index.json"
        save_index(index, dest)
        assert dest.exists()

    def test_load_raises_when_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_index(tmp_path / "nonexistent.json")

    def test_save_is_valid_json(self, tmp_path):
        pages = {"http://a.com": {"url": "http://a.com", "title": "A", "text": "foo bar"}}
        index = build_index(pages)
        dest = tmp_path / "index.json"
        save_index(index, dest)
        with open(dest) as fh:
            loaded = json.load(fh)
        assert isinstance(loaded, dict)
