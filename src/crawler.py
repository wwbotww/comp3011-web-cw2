"""
crawler.py — Web crawler for quotes.toscrape.com.

Responsibilities:
- Crawl all quote listing pages (page/1/ through page/10/)
- Crawl author detail pages linked from quote listings
- Observe a mandatory 6-second politeness window between requests
- Return a mapping of URL → page data for the indexer
"""

import time
import logging
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://quotes.toscrape.com"
POLITENESS_WINDOW = 6  # seconds between requests
REQUEST_TIMEOUT = 10   # seconds before a request times out
MAX_RETRIES = 3        # number of retry attempts on failure

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; COMP3011-SearchBot/1.0; "
        "+https://github.com/wwbotww/comp3011-web-cw2)"
    )
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def fetch_page(url: str, session: requests.Session) -> Optional[BeautifulSoup]:
    """Fetch a single URL and return a BeautifulSoup object, or None on failure.

    Retries up to MAX_RETRIES times on transient errors.

    Args:
        url: The URL to fetch.
        session: A requests.Session to reuse connections.

    Returns:
        A BeautifulSoup parsed page, or None if all retries failed.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return BeautifulSoup(response.text, "lxml")
        except requests.exceptions.HTTPError as exc:
            logger.warning("HTTP error fetching %s (attempt %d/%d): %s", url, attempt, MAX_RETRIES, exc)
        except requests.exceptions.ConnectionError as exc:
            logger.warning("Connection error fetching %s (attempt %d/%d): %s", url, attempt, MAX_RETRIES, exc)
        except requests.exceptions.Timeout:
            logger.warning("Timeout fetching %s (attempt %d/%d)", url, attempt, MAX_RETRIES)
        except requests.exceptions.RequestException as exc:
            logger.error("Unexpected error fetching %s: %s", url, exc)
            break

        if attempt < MAX_RETRIES:
            time.sleep(POLITENESS_WINDOW)

    return None


def extract_page_text(soup: BeautifulSoup) -> str:
    """Extract visible text content from a parsed HTML page.

    Removes script and style elements before extracting text.

    Args:
        soup: A BeautifulSoup parsed page.

    Returns:
        A single string of all visible text, whitespace-normalised.
    """
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    return " ".join(text.split())


def extract_author_urls(soup: BeautifulSoup) -> list[str]:
    """Extract all author detail page URLs from a quotes listing page.

    Args:
        soup: A BeautifulSoup parsed quotes listing page.

    Returns:
        A list of absolute author page URLs.
    """
    author_urls: list[str] = []
    for link in soup.select("small.author + a"):
        href = link.get("href", "")
        if href.startswith("/author/"):
            author_urls.append(urljoin(BASE_URL, href))
    return author_urls


def get_next_page_url(soup: BeautifulSoup) -> Optional[str]:
    """Return the URL of the next quotes listing page, or None if there is none.

    Args:
        soup: A BeautifulSoup parsed quotes listing page.

    Returns:
        Absolute URL of the next page, or None.
    """
    next_btn = soup.select_one("li.next > a")
    if next_btn:
        href = next_btn.get("href", "")
        return urljoin(BASE_URL, href)
    return None


def crawl(
    start_url: str = BASE_URL,
    progress_callback=None,
) -> dict[str, dict]:
    """Crawl quotes.toscrape.com and return page data for indexing.

    Crawls all quote listing pages then all discovered author pages.
    Observes a POLITENESS_WINDOW-second delay between every HTTP request.

    Args:
        start_url: URL of the first page to crawl (defaults to BASE_URL).
        progress_callback: Optional callable(url, page_num, total) invoked
            after each page is fetched, useful for CLI progress display.

    Returns:
        A dict mapping URL → {"url": str, "title": str, "text": str}.
    """
    pages: dict[str, dict] = {}
    visited: set[str] = set()
    author_urls: list[str] = []

    session = requests.Session()

    # --- Phase 1: crawl quote listing pages ---
    current_url: Optional[str] = start_url
    page_num = 0

    while current_url and current_url not in visited:
        page_num += 1
        logger.info("[Crawling] %s", current_url)
        if progress_callback:
            progress_callback(current_url, phase="listing", page_num=page_num)

        soup = fetch_page(current_url, session)
        visited.add(current_url)

        if soup is None:
            logger.error("Failed to fetch %s — skipping.", current_url)
        else:
            title = soup.title.string.strip() if soup.title else current_url
            text = extract_page_text(soup)
            pages[current_url] = {"url": current_url, "title": title, "text": text}

            for author_url in extract_author_urls(soup):
                if author_url not in visited:
                    author_urls.append(author_url)

            current_url = get_next_page_url(soup)

        if current_url and current_url not in visited:
            time.sleep(POLITENESS_WINDOW)

    # --- Phase 2: crawl author pages ---
    seen_authors: set[str] = set()
    for author_url in author_urls:
        if author_url in seen_authors or author_url in visited:
            continue
        seen_authors.add(author_url)

        logger.info("[Crawling] %s", author_url)
        if progress_callback:
            progress_callback(author_url, phase="author")

        soup = fetch_page(author_url, session)
        visited.add(author_url)

        if soup is not None:
            title = soup.title.string.strip() if soup.title else author_url
            text = extract_page_text(soup)
            pages[author_url] = {"url": author_url, "title": title, "text": text}

        time.sleep(POLITENESS_WINDOW)

    logger.info("[Crawl complete] %d pages fetched.", len(pages))
    return pages
