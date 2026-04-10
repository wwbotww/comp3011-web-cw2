"""
test_main.py — Unit tests for src/main.py.

All crawl/index operations are mocked so tests run instantly.
"""

import json
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.indexer import build_index
from src.main import cmd_build, cmd_find, cmd_load, cmd_print, run_shell


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_index() -> dict:
    """Build a small deterministic index for testing."""
    pages = {
        "http://a.com": {"url": "http://a.com", "title": "A", "text": "hello world"},
        "http://b.com": {"url": "http://b.com", "title": "B", "text": "hello python"},
    }
    return build_index(pages)


# ---------------------------------------------------------------------------
# cmd_build
# ---------------------------------------------------------------------------

class TestCmdBuild:

    @patch("src.main.save_index")
    @patch("src.main.build_index")
    @patch("src.main.crawl")
    def test_returns_index_on_success(self, mock_crawl, mock_build, mock_save):
        pages = {"http://a.com": {"url": "http://a.com", "title": "A", "text": "hello"}}
        mock_crawl.return_value = pages
        fake_index = {"hello": {}, "__meta__": {"total_words": 1, "indexed_pages": 1, "total_pages": 1, "built_at": "now"}}
        mock_build.return_value = fake_index

        result = cmd_build()

        assert result == fake_index
        mock_crawl.assert_called_once()
        mock_build.assert_called_once_with(pages)
        mock_save.assert_called_once()

    @patch("src.main.crawl")
    def test_returns_none_when_crawl_fails(self, mock_crawl):
        mock_crawl.return_value = {}

        result = cmd_build()

        assert result is None

    @patch("src.main.save_index")
    @patch("src.main.build_index")
    @patch("src.main.crawl")
    def test_prints_summary(self, mock_crawl, mock_build, mock_save, capsys):
        mock_crawl.return_value = {"http://a.com": {"url": "http://a.com", "title": "A", "text": "hi"}}
        mock_build.return_value = {
            "hi": {},
            "__meta__": {"total_words": 1, "indexed_pages": 1, "total_pages": 1, "built_at": "now"},
        }

        cmd_build()

        output = capsys.readouterr().out
        assert "1 unique words" in output
        assert "Saved to" in output


# ---------------------------------------------------------------------------
# cmd_load
# ---------------------------------------------------------------------------

class TestCmdLoad:

    def test_returns_index_on_success(self, tmp_path):
        idx = _small_index()
        dest = tmp_path / "index.json"
        with open(dest, "w") as f:
            json.dump(idx, f)

        result = cmd_load(index_path=dest)

        assert result is not None
        assert "__meta__" in result

    def test_returns_none_when_file_missing(self, tmp_path, capsys):
        result = cmd_load(index_path=tmp_path / "nope.json")

        assert result is None
        assert "error" in capsys.readouterr().out.lower()

    def test_returns_none_on_corrupt_json(self, tmp_path, capsys):
        bad = tmp_path / "bad.json"
        bad.write_text("NOT JSON", encoding="utf-8")

        result = cmd_load(index_path=bad)

        assert result is None
        assert "corrupted" in capsys.readouterr().out.lower()

    def test_prints_loaded_summary(self, tmp_path, capsys):
        idx = _small_index()
        dest = tmp_path / "index.json"
        with open(dest, "w") as f:
            json.dump(idx, f)

        cmd_load(index_path=dest)

        output = capsys.readouterr().out
        assert "[Loaded]" in output


# ---------------------------------------------------------------------------
# cmd_print
# ---------------------------------------------------------------------------

class TestCmdPrint:

    def test_prints_word_entry(self, capsys):
        idx = _small_index()
        cmd_print(idx, ["hello"])
        output = capsys.readouterr().out
        assert "http://a.com" in output
        assert "frequency" in output

    def test_no_index_loaded(self, capsys):
        cmd_print(None, ["hello"])
        output = capsys.readouterr().out
        assert "no index" in output.lower()

    def test_no_args(self, capsys):
        idx = _small_index()
        cmd_print(idx, [])
        output = capsys.readouterr().out
        assert "usage" in output.lower()

    def test_word_not_found(self, capsys):
        idx = _small_index()
        cmd_print(idx, ["zzzzz"])
        output = capsys.readouterr().out
        assert "not found" in output.lower()


# ---------------------------------------------------------------------------
# cmd_find
# ---------------------------------------------------------------------------

class TestCmdFind:

    def test_finds_matching_pages(self, capsys):
        idx = _small_index()
        cmd_find(idx, ["hello"])
        output = capsys.readouterr().out
        assert "http://a.com" in output
        assert "http://b.com" in output

    def test_no_index_loaded(self, capsys):
        cmd_find(None, ["hello"])
        output = capsys.readouterr().out
        assert "no index" in output.lower()

    def test_no_args(self, capsys):
        idx = _small_index()
        cmd_find(idx, [])
        output = capsys.readouterr().out
        assert "usage" in output.lower()

    def test_multi_word_query(self, capsys):
        idx = _small_index()
        cmd_find(idx, ["hello", "python"])
        output = capsys.readouterr().out
        assert "http://b.com" in output
        assert "http://a.com" not in output


# ---------------------------------------------------------------------------
# run_shell (integration via mocked stdin)
# ---------------------------------------------------------------------------

class TestRunShell:

    def _run_with_input(self, commands: list[str], capsys) -> str:
        """Feed a sequence of commands to run_shell and return stdout."""
        text = "\n".join(commands) + "\n"
        with patch("builtins.input", side_effect=commands + [EOFError]):
            run_shell()
        return capsys.readouterr().out

    def test_help_command(self, capsys):
        output = self._run_with_input(["help", "quit"], capsys)
        assert "Available commands" in output

    def test_quit_command(self, capsys):
        output = self._run_with_input(["quit"], capsys)
        assert "Goodbye" in output

    def test_exit_command(self, capsys):
        output = self._run_with_input(["exit"], capsys)
        assert "Goodbye" in output

    def test_unknown_command(self, capsys):
        output = self._run_with_input(["foobar", "quit"], capsys)
        assert "Unknown command" in output

    def test_empty_input_ignored(self, capsys):
        output = self._run_with_input(["", "quit"], capsys)
        assert "Goodbye" in output

    def test_eof_exits_gracefully(self, capsys):
        with patch("builtins.input", side_effect=EOFError):
            run_shell()
        output = capsys.readouterr().out
        assert "Goodbye" in output

    def test_keyboard_interrupt_exits(self, capsys):
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            run_shell()
        output = capsys.readouterr().out
        assert "Goodbye" in output

    def test_print_without_load_shows_error(self, capsys):
        output = self._run_with_input(["print hello", "quit"], capsys)
        assert "no index" in output.lower()

    def test_find_without_load_shows_error(self, capsys):
        output = self._run_with_input(["find hello", "quit"], capsys)
        assert "no index" in output.lower()

    @patch("src.main.cmd_load")
    def test_load_then_print(self, mock_load, capsys):
        mock_load.return_value = _small_index()

        output = self._run_with_input(["load", "print hello", "quit"], capsys)
        assert "frequency" in output

    @patch("src.main.cmd_load")
    def test_load_then_find(self, mock_load, capsys):
        mock_load.return_value = _small_index()

        output = self._run_with_input(["load", "find hello", "quit"], capsys)
        assert "http://a.com" in output

    def test_command_case_insensitive(self, capsys):
        output = self._run_with_input(["HELP", "QUIT"], capsys)
        assert "Available commands" in output
        assert "Goodbye" in output
