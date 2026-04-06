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
# Helpers
# ---------------------------------------------------------------------------

def pages(**kwargs) -> dict:
    """Build a pages dict from keyword args mapping url → text."""
    return {url: {"url": url, "title": url, "text": text} for url, text in kwargs.items()}


# ---------------------------------------------------------------------------
# tokenise
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
        assert all(t.isalpha() or "_" in t for t in tokens)

    def test_removes_stop_words(self):
        tokens = tokenise("the quick brown fox")
        assert "the" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens

    def test_empty_string_returns_empty_list(self):
        assert tokenise("") == []

    def test_only_stop_words_returns_empty_list(self):
        assert tokenise("the and or but") == []

    def test_numbers_kept_as_tokens(self):
        assert "3" in tokenise("3 wise men")

    def test_case_insensitive_stop_word_removal(self):
        tokens = tokenise("The quick fox")
        assert "the" not in tokens

    def test_multiple_spaces_collapsed(self):
        tokens = tokenise("foo   bar")
        assert tokens == ["foo", "bar"]

    def test_preserves_order(self):
        tokens = tokenise("apple banana cherry")
        assert tokens == ["apple", "banana", "cherry"]

    def test_unicode_punctuation_stripped(self):
        # em dash (—) and curly quotes are not in string.punctuation
        # but must still be stripped so words aren't glued together
        tokens = tokenise("life—and death.")
        assert "life" in tokens
        assert "death" in tokens
        # "and" is a stop word so must not appear
        assert "and" not in tokens


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------

class TestBuildIndex:

    def test_basic_word_frequency(self):
        index = build_index(pages(url1="good good bad"))
        assert index["good"]["url1"]["frequency"] == 2
        assert index["bad"]["url1"]["frequency"] == 1

    def test_word_positions_recorded_in_order(self):
        index = build_index(pages(url1="apple banana apple"))
        positions = index["apple"]["url1"]["positions"]
        assert positions == [0, 2]

    def test_positions_are_token_level_not_char_level(self):
        # After stop-word removal: tokens = ["quick", "brown", "fox"]
        index = build_index(pages(url1="the quick brown fox"))
        assert index["quick"]["url1"]["positions"] == [0]
        assert index["brown"]["url1"]["positions"] == [1]
        assert index["fox"]["url1"]["positions"]   == [2]

    def test_tf_is_frequency_over_total_tokens(self):
        # tokens = ["apple", "apple", "orange"]  total = 3
        index = build_index(pages(url1="apple apple orange"))
        assert index["apple"]["url1"]["tf"]  == pytest.approx(2 / 3, rel=1e-4)
        assert index["orange"]["url1"]["tf"] == pytest.approx(1 / 3, rel=1e-4)

    def test_tf_between_zero_and_one(self):
        index = build_index(pages(url1="apple orange apple banana"))
        for word in ["apple", "orange", "banana"]:
            assert 0.0 < index[word]["url1"]["tf"] <= 1.0

    def test_tfidf_field_present(self):
        index = build_index(pages(url1="python code", url2="java code"))
        assert "tfidf" in index["code"]["url1"]

    def test_case_insensitive_indexing(self):
        index = build_index(pages(url1="Good good GOOD"))
        assert "good" in index
        assert index["good"]["url1"]["frequency"] == 3

    def test_punctuation_stripped_before_indexing(self):
        index = build_index(pages(url1="hello, world!"))
        assert "hello" in index
        assert "world" in index

    # --- IDF / TF-IDF ---

    def test_word_in_all_pages_has_zero_tfidf(self):
        # "python" in both pages → df = N → idf = log(1) = 0
        index = build_index(pages(url1="python java", url2="python ruby"))
        assert index["python"]["url1"]["tfidf"] == pytest.approx(0.0, abs=1e-6)
        assert index["python"]["url2"]["tfidf"] == pytest.approx(0.0, abs=1e-6)

    def test_rare_word_has_higher_tfidf_than_common_word(self):
        # "java" in 1/2 pages, "python" in 2/2 pages
        index = build_index(pages(url1="python java", url2="python ruby"))
        assert index["java"]["url1"]["tfidf"] > index["python"]["url1"]["tfidf"]

    def test_tfidf_uses_indexed_pages_not_total_pages(self):
        # page2 has empty text and is skipped — indexed_pages = 1
        # "hello" df=1, indexed_pages=1 → idf = log(1/1) = 0
        p = {
            "url1": {"url": "url1", "title": "t", "text": "hello world"},
            "url2": {"url": "url2", "title": "t", "text": ""},   # empty, skipped
        }
        index = build_index(p)
        # idf = log(1/1) = 0, so tfidf = 0
        assert index["hello"]["url1"]["tfidf"] == pytest.approx(0.0, abs=1e-6)

    # --- Metadata ---

    def test_meta_block_present(self):
        index = build_index(pages(url1="hello world"))
        assert "__meta__" in index

    def test_meta_total_pages(self):
        index = build_index(pages(url1="hello", url2="world"))
        assert index["__meta__"]["total_pages"] == 2

    def test_meta_indexed_pages_excludes_empty(self):
        p = {
            "url1": {"url": "url1", "title": "t", "text": "hello"},
            "url2": {"url": "url2", "title": "t", "text": ""},
        }
        index = build_index(p)
        assert index["__meta__"]["total_pages"]   == 2
        assert index["__meta__"]["indexed_pages"] == 1

    def test_meta_total_words_matches_index_keys(self):
        index = build_index(pages(url1="apple banana cherry"))
        word_keys = [k for k in index if k != "__meta__"]
        assert index["__meta__"]["total_words"] == len(word_keys)

    def test_meta_built_at_is_iso_string(self):
        index = build_index(pages(url1="hello"))
        built_at = index["__meta__"]["built_at"]
        assert isinstance(built_at, str)
        assert "T" in built_at   # ISO-8601 separator

    # --- Edge cases ---

    def test_empty_pages_dict(self):
        index = build_index({})
        assert index["__meta__"]["total_pages"]   == 0
        assert index["__meta__"]["indexed_pages"] == 0
        assert index["__meta__"]["total_words"]   == 0

    def test_page_with_empty_text_skipped(self):
        p = {"url1": {"url": "url1", "title": "t", "text": ""}}
        index = build_index(p)
        assert index["__meta__"]["indexed_pages"] == 0
        assert index["__meta__"]["total_words"]   == 0

    def test_page_with_only_stop_words_skipped(self):
        p = {"url1": {"url": "url1", "title": "t", "text": "the and or but"}}
        index = build_index(p)
        assert index["__meta__"]["indexed_pages"] == 0

    def test_missing_text_field_treated_as_empty(self):
        p = {"url1": {"url": "url1", "title": "t"}}   # no "text" key
        index = build_index(p)
        assert index["__meta__"]["indexed_pages"] == 0

    def test_multi_page_word_appears_in_both_postings(self):
        index = build_index(pages(url1="python rocks", url2="python rules"))
        assert "url1" in index["python"]
        assert "url2" in index["python"]


# ---------------------------------------------------------------------------
# save_index / load_index
# ---------------------------------------------------------------------------

class TestIndexPersistence:

    def test_round_trip_preserves_data(self, tmp_path):
        index = build_index(pages(url1="hello world"))
        dest = tmp_path / "index.json"
        save_index(index, dest)
        loaded = load_index(dest)
        assert loaded == index

    def test_save_creates_missing_parent_dirs(self, tmp_path):
        index = build_index(pages(url1="test"))
        dest = tmp_path / "nested" / "deep" / "index.json"
        save_index(index, dest)
        assert dest.exists()

    def test_saved_file_is_valid_utf8_json(self, tmp_path):
        index = build_index(pages(url1="café naïve"))
        dest = tmp_path / "index.json"
        save_index(index, dest)
        with open(dest, encoding="utf-8") as fh:
            loaded = json.load(fh)
        assert isinstance(loaded, dict)

    def test_load_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Run the 'build' command"):
            load_index(tmp_path / "nonexistent.json")

    def test_load_raises_on_corrupt_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{ this is not json }", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_index(bad)

    def test_saved_index_contains_meta(self, tmp_path):
        index = build_index(pages(url1="hello"))
        dest = tmp_path / "index.json"
        save_index(index, dest)
        with open(dest, encoding="utf-8") as fh:
            raw = json.load(fh)
        assert "__meta__" in raw

    def test_load_returns_dict(self, tmp_path):
        index = build_index(pages(url1="hello"))
        dest = tmp_path / "index.json"
        save_index(index, dest)
        result = load_index(dest)
        assert isinstance(result, dict)
