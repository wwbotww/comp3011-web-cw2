"""
crawler.py — Web crawler for quotes.toscrape.com.

Crawling strategy:
  Phase 1 — Follow the "Next" pagination links starting from BASE_URL until
             there is no next page or a page fails to load.
  Phase 2 — Visit every unique author detail page discovered in Phase 1.

A mandatory POLITENESS_WINDOW-second sleep is observed *before* every HTTP
request (except the very first one) so the target server is not overloaded.
"""

import time
import logging
from typing import Callable, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://quotes.toscrape.com"
POLITENESS_WINDOW = 6   # seconds between successive requests (hard requirement)
REQUEST_TIMEOUT = 10    # seconds before giving up on a single request
MAX_RETRIES = 3         # how many times to retry a transient failure

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; COMP3011-SearchBot/1.0; "
        "+https://github.com/wwbotww/comp3011-web-cw2)"
    )
}

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Type alias for the progress callback
ProgressCallback = Callable[..., None]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def fetch_page(url: str, session: requests.Session) -> Optional[BeautifulSoup]:
    """Fetch *url* and return a parsed BeautifulSoup, or ``None`` on failure.

    Retries up to ``MAX_RETRIES`` times on transient HTTP / network errors.
    Between retries the politeness window is observed so we never hammer the
    server even during error recovery.

    Args:
        url:     The URL to fetch.
        session: A ``requests.Session`` instance to reuse TCP connections.

    Returns:
        A ``BeautifulSoup`` object on success, ``None`` after all retries fail.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return BeautifulSoup(response.text, "lxml")

        except requests.exceptions.HTTPError as exc:
            logger.warning(
                "HTTP error fetching %s (attempt %d/%d): %s",
                url, attempt, MAX_RETRIES, exc,
            )
        except requests.exceptions.ConnectionError as exc:
            logger.warning(
                "Connection error fetching %s (attempt %d/%d): %s",
                url, attempt, MAX_RETRIES, exc,
            )
        except requests.exceptions.Timeout:
            logger.warning(
                "Timeout fetching %s (attempt %d/%d)", url, attempt, MAX_RETRIES,
            )
        except requests.exceptions.RequestException as exc:
            # Non-transient error — no point retrying
            logger.error("Unexpected request error fetching %s: %s", url, exc)
            return None

        if attempt < MAX_RETRIES:
            time.sleep(POLITENESS_WINDOW)

    logger.error("All %d attempts failed for %s — skipping.", MAX_RETRIES, url)
    return None


def extract_page_text(soup: BeautifulSoup) -> str:
    """Return the visible text of a page as a single whitespace-normalised string.

    Strips ``<script>``, ``<style>``, ``<nav>``, and ``<footer>`` elements
    before extraction to avoid indexing boilerplate.

    Args:
        soup: A parsed BeautifulSoup page.

    Returns:
        Visible text joined by spaces, with runs of whitespace collapsed.
    """
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    raw = soup.get_text(separator=" ")
    return " ".join(raw.split())


def has_quotes(soup: BeautifulSoup) -> bool:
    """Return ``True`` if the page contains at least one quote block.

    Useful for detecting the "No quotes found!" sentinel page that
    quotes.toscrape.com returns beyond the last real page.

    Args:
        soup: A parsed BeautifulSoup page.

    Returns:
        ``True`` when one or more ``div.quote`` elements are found.
    """
    return bool(soup.select_one("div.quote"))


def extract_author_urls(soup: BeautifulSoup) -> list[str]:
    """Return absolute author-detail URLs found on a quote-listing page.

    On quotes.toscrape.com every quote has a ``<small class="author">``
    immediately followed by an ``<a href="/author/…">(about)</a>`` link.

    Args:
        soup: A parsed quote-listing page.

    Returns:
        Deduplicated list of absolute author page URLs.
    """
    seen: set[str] = set()
    urls: list[str] = []

    for link in soup.select("small.author + a"):
        href = link.get("href", "")
        if href.startswith("/author/"):
            absolute = urljoin(BASE_URL, href)
            if absolute not in seen:
                seen.add(absolute)
                urls.append(absolute)

    return urls


def get_next_page_url(soup: BeautifulSoup) -> Optional[str]:
    """Return the absolute URL of the next listing page, or ``None``.

    Looks for the ``<li class="next"><a href="…">`` pagination element.

    Args:
        soup: A parsed quote-listing page.

    Returns:
        Absolute URL of the next page, or ``None`` if this is the last page.
    """
    next_btn = soup.select_one("li.next > a")
    if next_btn:
        href = next_btn.get("href", "")
        return urljoin(BASE_URL, href)
    return None


# ---------------------------------------------------------------------------
# Main crawl entry point
# ---------------------------------------------------------------------------

def crawl(
    start_url: str = BASE_URL,
    progress_callback: Optional[ProgressCallback] = None,
) -> dict[str, dict]:
    """Crawl quotes.toscrape.com and return page data ready for indexing.

    Performs two phases:
      1. Follow "Next" pagination links from *start_url*, collecting author
         URLs along the way.
      2. Visit each unique author detail page discovered in phase 1.

    A ``POLITENESS_WINDOW``-second sleep is inserted between *every* pair of
    successive requests.

    Args:
        start_url:
            First URL to fetch. Defaults to the site root which redirects to
            the first listing page.
        progress_callback:
            Optional callable invoked after each successful (or failed) fetch.
            Signature: ``callback(url, phase, page_num)`` where *phase* is
            ``"listing"`` or ``"author"`` and *page_num* is the 1-based listing
            page counter (0 for author pages).

    Returns:
        ``dict`` mapping URL → ``{"url": str, "title": str, "text": str}``.
        Only successfully fetched pages with content are included.
    """
    pages: dict[str, dict] = {}
    visited: set[str] = set()
    author_queue: list[str] = []   # ordered, deduplicated in extract_author_urls

    session = requests.Session()
    first_request = True

    # -----------------------------------------------------------------------
    # Phase 1: crawl quote listing pages
    # -----------------------------------------------------------------------
    current_url: Optional[str] = start_url
    page_num = 0

    while current_url and current_url not in visited:
        page_num += 1

        if not first_request:
            time.sleep(POLITENESS_WINDOW)
        first_request = False

        logger.info("[Phase 1 | page %d] Fetching %s", page_num, current_url)
        if progress_callback:
            progress_callback(current_url, phase="listing", page_num=page_num)

        soup = fetch_page(current_url, session)
        visited.add(current_url)

        if soup is None:
            logger.error("Failed to fetch listing page %s — stopping listing crawl.", current_url)
            # Cannot determine the next URL without parsing the current page,
            # so we must stop the listing phase here.
            current_url = None
            continue

        # Skip pages that contain no quotes (e.g. the "No quotes found!" page)
        if not has_quotes(soup):
            logger.info("No quotes found on %s — treating as end of listing.", current_url)
            break

        title = soup.title.string.strip() if soup.title else current_url
        text = extract_page_text(soup)
        pages[current_url] = {"url": current_url, "title": title, "text": text}
        logger.info("  → Stored page '%s' (%d chars of text)", title, len(text))

        # Collect author URLs for phase 2
        for author_url in extract_author_urls(soup):
            if author_url not in visited and author_url not in author_queue:
                author_queue.append(author_url)

        current_url = get_next_page_url(soup)

    logger.info(
        "[Phase 1 complete] %d listing page(s) stored; %d author page(s) queued.",
        len(pages), len(author_queue),
    )

    # -----------------------------------------------------------------------
    # Phase 2: crawl author detail pages
    # -----------------------------------------------------------------------
    for idx, author_url in enumerate(author_queue):
        if author_url in visited:
            continue

        time.sleep(POLITENESS_WINDOW)

        logger.info(
            "[Phase 2 | author %d/%d] Fetching %s",
            idx + 1, len(author_queue), author_url,
        )
        if progress_callback:
            progress_callback(author_url, phase="author", page_num=0)

        soup = fetch_page(author_url, session)
        visited.add(author_url)

        if soup is None:
            logger.warning("Failed to fetch author page %s — skipping.", author_url)
            continue

        title = soup.title.string.strip() if soup.title else author_url
        text = extract_page_text(soup)
        pages[author_url] = {"url": author_url, "title": title, "text": text}
        logger.info("  → Stored author page '%s' (%d chars of text)", title, len(text))

    logger.info(
        "[Crawl complete] %d page(s) fetched in total (%d listing + %d author).",
        len(pages),
        sum(1 for u in pages if "/author/" not in u),
        sum(1 for u in pages if "/author/" in u),
    )
    return pages
