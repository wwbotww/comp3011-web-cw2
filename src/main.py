"""
main.py — Interactive command-line shell for the search engine tool.

Commands:
    build           Crawl the website, build the index, and save it.
    load            Load a previously saved index from disk.
    print <word>    Display the inverted index entry for <word>.
    find <query>    Find pages containing all words in <query>.
    help            Show this help message.
    quit / exit     Exit the program.
"""

import sys
from pathlib import Path
from typing import Any, Optional

# Allow running as `python src/main.py` from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.crawler import crawl
from src.indexer import build_index, load_index, save_index, DEFAULT_INDEX_PATH
from src.search import find_pages, print_word

HELP_TEXT = """
Available commands:
  build            Crawl quotes.toscrape.com, build the inverted index and save it.
  load             Load the inverted index from disk (must have run 'build' first).
  print <word>     Display the inverted index entry for <word>.
  find <query>     Find all pages containing every word in <query>.
  help             Show this help message.
  quit / exit      Exit the search engine.
""".strip()


def _progress(url: str, phase: str = "listing", page_num: int = 0) -> None:
    if phase == "listing":
        print(f"  [page {page_num}] {url}")
    else:
        print(f"  [author]  {url}")


def cmd_build() -> Optional[dict[str, Any]]:
    """Crawl the website, build the index and save it to disk."""
    print("Starting crawl — this will take several minutes due to the 6-second politeness window.")
    print("Target: https://quotes.toscrape.com\n")

    pages = crawl(progress_callback=_progress)
    if not pages:
        print("[Error] No pages were crawled. Check your internet connection.")
        return None

    print(f"\n[Crawl complete] {len(pages)} pages fetched. Building index...")
    index = build_index(pages)
    save_index(index, DEFAULT_INDEX_PATH)

    meta = index.get("__meta__", {})
    print(
        f"[Done] Index contains {meta.get('total_words', '?')} unique words "
        f"across {meta.get('total_pages', '?')} pages.\n"
        f"       Saved to: {DEFAULT_INDEX_PATH}"
    )
    return index


def cmd_load() -> Optional[dict[str, Any]]:
    """Load the index from disk."""
    try:
        index = load_index(DEFAULT_INDEX_PATH)
        meta = index.get("__meta__", {})
        print(
            f"[Loaded] {meta.get('total_words', '?')} unique words across "
            f"{meta.get('total_pages', '?')} pages  "
            f"(built at {meta.get('built_at', 'unknown')})"
        )
        return index
    except FileNotFoundError as exc:
        print(f"[Error] {exc}")
        return None


def run_shell() -> None:
    """Start the interactive command-line shell."""
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
            if not args:
                print("[Error] Usage: print <word>")
            elif index is None:
                print("[Error] No index loaded. Run 'build' or 'load' first.")
            else:
                print(print_word(index, args[0]))

        elif command == "find":
            if not args:
                print("[Error] Usage: find <query>")
            elif index is None:
                print("[Error] No index loaded. Run 'build' or 'load' first.")
            else:
                query = " ".join(args)
                print(find_pages(index, query))

        elif command == "help":
            print(HELP_TEXT)

        elif command in ("quit", "exit"):
            print("Goodbye!")
            break

        else:
            print(f'[Error] Unknown command: "{command}". Type "help" for available commands.')


if __name__ == "__main__":
    run_shell()
