"""
main.py — Interactive command-line shell for the search engine tool.

Commands
--------
    build           Crawl the website, build the index and save it.
    load            Load a previously saved index from disk.
    print <word>    Display the inverted index entry for <word>.
    find <query>    Find pages containing all words in <query>.
    help            Show available commands.
    quit / exit     Exit the program.
"""

import json
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure the project root is on sys.path so that `python src/main.py`
# works from the project root without installing the package.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.crawler import crawl
from src.indexer import build_index, load_index, save_index, DEFAULT_INDEX_PATH
from src.search import find_pages, print_word

HELP_TEXT = """\
Available commands:
  build            Crawl quotes.toscrape.com, build the inverted index and save it.
  load             Load the inverted index from disk (must have run 'build' first).
  print <word>     Display the inverted index entry for <word>.
  find <query>     Find all pages containing every word in <query>.
  help             Show this help message.
  quit / exit      Exit the search engine."""


def _progress(url: str, phase: str = "listing", page_num: int = 0) -> None:
    """Default progress callback printed to stdout during a crawl."""
    if phase == "listing":
        print(f"  [page {page_num}] {url}")
    else:
        print(f"  [author]  {url}")


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_build(
    progress_callback=_progress,
    index_path: Path = DEFAULT_INDEX_PATH,
) -> Optional[dict[str, Any]]:
    """Crawl the target website, build the index and save it to disk.

    Returns the index dict on success, or ``None`` if the crawl failed.
    """
    print(
        "Starting crawl — this will take several minutes "
        "due to the 6-second politeness window."
    )
    print("Target: https://quotes.toscrape.com\n")

    pages = crawl(progress_callback=progress_callback)
    if not pages:
        print("[Error] No pages were crawled. Check your internet connection.")
        return None

    print(f"\n[Crawl complete] {len(pages)} pages fetched. Building index...")
    index = build_index(pages)
    save_index(index, index_path)

    meta = index.get("__meta__", {})
    print(
        f"[Done] Index contains {meta.get('total_words', '?')} unique words "
        f"from {meta.get('indexed_pages', '?')}/{meta.get('total_pages', '?')} pages.\n"
        f"       Saved to: {index_path}"
    )
    return index


def cmd_load(
    index_path: Path = DEFAULT_INDEX_PATH,
) -> Optional[dict[str, Any]]:
    """Load a previously built index from disk.

    Returns the index dict on success, or ``None`` on failure.
    """
    try:
        index = load_index(index_path)
    except FileNotFoundError as exc:
        print(f"[Error] {exc}")
        return None
    except json.JSONDecodeError:
        print(f"[Error] Index file at '{index_path}' is corrupted. Please run 'build' again.")
        return None

    meta = index.get("__meta__", {})
    print(
        f"[Loaded] {meta.get('total_words', '?')} unique words "
        f"from {meta.get('indexed_pages', meta.get('total_pages', '?'))}/"
        f"{meta.get('total_pages', '?')} pages  "
        f"(built at {meta.get('built_at', 'unknown')})"
    )
    return index


def cmd_print(index: Optional[dict[str, Any]], args: list[str]) -> None:
    """Handle the ``print <word>`` command."""
    if index is None:
        print("[Error] No index loaded. Run 'build' or 'load' first.")
        return
    if not args:
        print("[Error] Usage: print <word>")
        return
    print(print_word(index, args[0]))


def cmd_find(index: Optional[dict[str, Any]], args: list[str]) -> None:
    """Handle the ``find <query>`` command."""
    if index is None:
        print("[Error] No index loaded. Run 'build' or 'load' first.")
        return
    if not args:
        print("[Error] Usage: find <query>")
        return
    query = " ".join(args)
    print(find_pages(index, query))


# ---------------------------------------------------------------------------
# Interactive shell
# ---------------------------------------------------------------------------

def run_shell() -> None:
    """Start the interactive command-line REPL.

    Reads one command per line from stdin, dispatches it to the
    appropriate handler, and loops until the user types ``quit`` /
    ``exit`` or sends EOF / Ctrl-C.
    """
    print("Search Engine Tool — COMP3011 Coursework 2")
    print('Type "help" for available commands.\n')

    index: Optional[dict[str, Any]] = None

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue

        parts = raw.split()
        command = parts[0].lower()
        args = parts[1:]

        if command == "build":
            index = cmd_build()

        elif command == "load":
            index = cmd_load()

        elif command == "print":
            cmd_print(index, args)

        elif command == "find":
            cmd_find(index, args)

        elif command == "help":
            print(HELP_TEXT)

        elif command in ("quit", "exit"):
            print("Goodbye!")
            break

        else:
            print(
                f'[Error] Unknown command: "{command}". '
                'Type "help" for available commands.'
            )


if __name__ == "__main__":
    run_shell()
