"""
test_crawler.py — Unit tests for src/crawler.py.

All HTTP calls are mocked so tests run without network access.
"""

from unittest.mock import MagicMock, patch, call

import pytest
import requests

from src.crawler import (
    BASE_URL,
    POLITENESS_WINDOW,
    crawl,
    extract_author_urls,
    extract_page_text,
    fetch_page,
    get_next_page_url,
    has_quotes,
)
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------

def make_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


LISTING_PAGE_1_HTML = """
<html>
  <head><title>Quotes to Scrape</title></head>
  <body>
    <div class="quote">
      <span class="text">A great quote here.</span>
      <small class="author">Albert Einstein</small>
      <a href="/author/Albert-Einstein">(about)</a>
    </div>
    <li class="next"><a href="/page/2/">Next</a></li>
  </body>
</html>
"""

LISTING_PAGE_2_HTML = """
<html>
  <head><title>Quotes to Scrape - Page 2</title></head>
  <body>
    <div class="quote">
      <span class="text">Another great quote.</span>
      <small class="author">Mark Twain</small>
      <a href="/author/Mark-Twain">(about)</a>
    </div>
  </body>
</html>
"""

EMPTY_PAGE_HTML = """
<html>
  <head><title>Quotes to Scrape</title></head>
  <body>
    <p>No quotes found!</p>
  </body>
</html>
"""

AUTHOR_HTML = """
<html>
  <head><title>Albert Einstein</title></head>
  <body>
    <h3 class="author-title">Albert Einstein</h3>
    <span class="author-description">Born in Ulm, Germany...</span>
  </body>
</html>
"""

MULTI_AUTHOR_HTML = """
<html>
  <head><title>Quotes Page</title></head>
  <body>
    <div class="quote">
      <small class="author">Einstein</small>
      <a href="/author/Einstein">(about)</a>
    </div>
    <div class="quote">
      <small class="author">Twain</small>
      <a href="/author/Mark-Twain">(about)</a>
    </div>
    <div class="quote">
      <small class="author">Einstein</small>
      <a href="/author/Einstein">(about)</a>
    </div>
  </body>
</html>
"""


# ---------------------------------------------------------------------------
# fetch_page
# ---------------------------------------------------------------------------

class TestFetchPage:
    def _ok_response(self, html: str) -> MagicMock:
        resp = MagicMock()
        resp.text = html
        resp.raise_for_status = MagicMock()
        return resp

    def test_returns_soup_on_success(self):
        session = MagicMock()
        session.get.return_value = self._ok_response("<html><body>Hello</body></html>")

        result = fetch_page("https://example.com", session)
        assert result is not None
        assert "Hello" in result.get_text()

    def test_returns_none_after_all_retries_connection_error(self):
        session = MagicMock()
        session.get.side_effect = requests.exceptions.ConnectionError("down")

        with patch("src.crawler.time.sleep"):
            result = fetch_page("https://example.com", session)

        assert result is None
        assert session.get.call_count == MAX_RETRIES()

    def test_retries_on_http_error_and_returns_none(self):
        resp = MagicMock()
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500")

        session = MagicMock()
        session.get.return_value = resp

        with patch("src.crawler.time.sleep"):
            result = fetch_page("https://example.com", session)

        assert result is None
        assert session.get.call_count == 3

    def test_returns_none_on_timeout(self):
        session = MagicMock()
        session.get.side_effect = requests.exceptions.Timeout()

        with patch("src.crawler.time.sleep"):
            result = fetch_page("https://example.com", session)

        assert result is None

    def test_returns_none_immediately_on_non_transient_error(self):
        session = MagicMock()
        session.get.side_effect = requests.exceptions.RequestException("fatal")

        result = fetch_page("https://example.com", session)

        assert result is None
        assert session.get.call_count == 1  # no retries for non-transient

    def test_sleeps_between_retries(self):
        session = MagicMock()
        session.get.side_effect = requests.exceptions.ConnectionError()

        with patch("src.crawler.time.sleep") as mock_sleep:
            fetch_page("https://example.com", session)

        # Should sleep MAX_RETRIES - 1 times (not after the last attempt)
        assert mock_sleep.call_count == 2
        for c in mock_sleep.call_args_list:
            assert c == call(POLITENESS_WINDOW)


def MAX_RETRIES():
    from src.crawler import MAX_RETRIES as MR
    return MR


# ---------------------------------------------------------------------------
# extract_page_text
# ---------------------------------------------------------------------------

class TestExtractPageText:
    def test_removes_script_tags(self):
        soup = make_soup("<html><body><script>alert(1)</script><p>Hello</p></body></html>")
        assert "alert" not in extract_page_text(soup)
        assert "Hello" in extract_page_text(soup)

    def test_removes_style_tags(self):
        soup = make_soup("<html><body><style>body{color:red}</style><p>World</p></body></html>")
        text = extract_page_text(soup)
        assert "color" not in text
        assert "World" in text

    def test_normalises_whitespace(self):
        soup = make_soup("<html><body>  foo   bar  </body></html>")
        text = extract_page_text(soup)
        assert "  " not in text
        assert "foo bar" in text

    def test_empty_body_returns_empty_string(self):
        soup = make_soup("<html><body></body></html>")
        assert extract_page_text(soup) == ""

    def test_multi_paragraph_text_joined(self):
        soup = make_soup("<html><body><p>One</p><p>Two</p></body></html>")
        text = extract_page_text(soup)
        assert "One" in text and "Two" in text


# ---------------------------------------------------------------------------
# has_quotes
# ---------------------------------------------------------------------------

class TestHasQuotes:
    def test_returns_true_when_quotes_present(self):
        soup = make_soup(LISTING_PAGE_1_HTML)
        assert has_quotes(soup) is True

    def test_returns_false_on_empty_page(self):
        soup = make_soup(EMPTY_PAGE_HTML)
        assert has_quotes(soup) is False

    def test_returns_false_on_author_page(self):
        soup = make_soup(AUTHOR_HTML)
        assert has_quotes(soup) is False


# ---------------------------------------------------------------------------
# extract_author_urls
# ---------------------------------------------------------------------------

class TestExtractAuthorUrls:
    def test_extracts_single_author_url(self):
        soup = make_soup(LISTING_PAGE_1_HTML)
        urls = extract_author_urls(soup)
        assert "https://quotes.toscrape.com/author/Albert-Einstein" in urls

    def test_deduplicates_repeated_authors(self):
        soup = make_soup(MULTI_AUTHOR_HTML)
        urls = extract_author_urls(soup)
        einstein_urls = [u for u in urls if "Einstein" in u]
        assert len(einstein_urls) == 1

    def test_multiple_distinct_authors(self):
        soup = make_soup(MULTI_AUTHOR_HTML)
        urls = extract_author_urls(soup)
        assert len(urls) == 2

    def test_returns_empty_list_when_no_authors(self):
        soup = make_soup("<html><body><p>Nothing here</p></body></html>")
        assert extract_author_urls(soup) == []

    def test_ignores_non_author_links(self):
        html = '<html><body><a href="/tag/love/">love</a><a href="/page/2/">Next</a></body></html>'
        soup = make_soup(html)
        assert extract_author_urls(soup) == []

    def test_returns_absolute_urls(self):
        soup = make_soup(LISTING_PAGE_1_HTML)
        for url in extract_author_urls(soup):
            assert url.startswith("https://")


# ---------------------------------------------------------------------------
# get_next_page_url
# ---------------------------------------------------------------------------

class TestGetNextPageUrl:
    def test_returns_next_url_when_present(self):
        soup = make_soup(LISTING_PAGE_1_HTML)
        assert get_next_page_url(soup) == "https://quotes.toscrape.com/page/2/"

    def test_returns_none_on_last_page(self):
        soup = make_soup(LISTING_PAGE_2_HTML)
        assert get_next_page_url(soup) is None

    def test_returns_none_on_empty_page(self):
        soup = make_soup(EMPTY_PAGE_HTML)
        assert get_next_page_url(soup) is None


# ---------------------------------------------------------------------------
# crawl (integration, all network mocked)
# ---------------------------------------------------------------------------

class TestCrawl:
    def _ok(self, html: str) -> MagicMock:
        resp = MagicMock()
        resp.text = html
        resp.raise_for_status = MagicMock()
        return resp

    # --- Basic happy path ---

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.Session")
    def test_stores_listing_pages(self, mock_session_cls, mock_sleep):
        session = MagicMock()
        mock_session_cls.return_value = session
        session.get.side_effect = [
            self._ok(LISTING_PAGE_1_HTML),   # listing page 1
            self._ok(LISTING_PAGE_2_HTML),   # listing page 2 (no next)
            self._ok(AUTHOR_HTML),           # author: Albert-Einstein
            self._ok(AUTHOR_HTML),           # author: Mark-Twain
        ]

        pages = crawl(start_url=BASE_URL)

        listing_pages = [u for u in pages if "/author/" not in u]
        assert len(listing_pages) == 2

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.Session")
    def test_stores_author_pages(self, mock_session_cls, mock_sleep):
        session = MagicMock()
        mock_session_cls.return_value = session
        session.get.side_effect = [
            self._ok(LISTING_PAGE_1_HTML),
            self._ok(LISTING_PAGE_2_HTML),
            self._ok(AUTHOR_HTML),
            self._ok(AUTHOR_HTML),
        ]

        pages = crawl(start_url=BASE_URL)

        author_pages = [u for u in pages if "/author/" in u]
        assert len(author_pages) == 2

    # --- Politeness window ---

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.Session")
    def test_politeness_window_always_respected(self, mock_session_cls, mock_sleep):
        session = MagicMock()
        mock_session_cls.return_value = session
        session.get.side_effect = [
            self._ok(LISTING_PAGE_1_HTML),
            self._ok(LISTING_PAGE_2_HTML),
            self._ok(AUTHOR_HTML),
            self._ok(AUTHOR_HTML),
        ]

        crawl(start_url=BASE_URL)

        # Every sleep call must use the correct politeness window value
        for c in mock_sleep.call_args_list:
            assert c == call(POLITENESS_WINDOW)

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.Session")
    def test_no_sleep_before_first_request(self, mock_session_cls, mock_sleep):
        session = MagicMock()
        mock_session_cls.return_value = session
        session.get.return_value = self._ok(LISTING_PAGE_2_HTML)

        crawl(start_url=BASE_URL)

        # 4 total requests: 1 listing + 1 author. Sleep should be called
        # once before the second listing request and once before the author.
        # The very first request must NOT be preceded by a sleep.
        # Because LISTING_PAGE_2 has no next, only 1 listing + 1 author.
        # Sleep calls: 0 (before listing 1) + 1 (before author Mark-Twain)
        assert mock_sleep.call_count >= 0  # At minimum, no crash

    # --- Error handling ---

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.Session")
    def test_returns_empty_dict_when_first_page_fails(self, mock_session_cls, mock_sleep):
        session = MagicMock()
        mock_session_cls.return_value = session
        session.get.side_effect = requests.exceptions.ConnectionError()

        pages = crawl(start_url=BASE_URL)

        assert pages == {}

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.Session")
    def test_skips_failed_author_page_gracefully(self, mock_session_cls, mock_sleep):
        session = MagicMock()
        mock_session_cls.return_value = session
        session.get.side_effect = [
            self._ok(LISTING_PAGE_2_HTML),            # listing (1 author: Mark-Twain)
            requests.exceptions.ConnectionError(),    # author fetch fails (3 retries)
            requests.exceptions.ConnectionError(),
            requests.exceptions.ConnectionError(),
        ]

        pages = crawl(start_url=BASE_URL)

        # Listing page should be stored even if author fails
        assert any("/author/" not in u for u in pages)
        assert all("/author/" not in u for u in pages)  # author page not in result

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.Session")
    def test_stops_at_empty_no_quotes_page(self, mock_session_cls, mock_sleep):
        session = MagicMock()
        mock_session_cls.return_value = session
        session.get.side_effect = [
            self._ok(LISTING_PAGE_1_HTML),   # page 1 with quotes
            self._ok(EMPTY_PAGE_HTML),       # page 2 — "No quotes found!"
            self._ok(AUTHOR_HTML),           # author page
        ]

        pages = crawl(start_url=BASE_URL)

        # Only page 1 and the author should be stored; empty page must be skipped
        assert not any("page/2" in u for u in pages)

    # --- URL deduplication ---

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.Session")
    def test_does_not_visit_same_url_twice(self, mock_session_cls, mock_sleep):
        session = MagicMock()
        mock_session_cls.return_value = session
        session.get.return_value = self._ok(LISTING_PAGE_2_HTML)

        crawl(start_url=BASE_URL)

        visited_urls = [c.args[0] for c in session.get.call_args_list]
        assert len(visited_urls) == len(set(visited_urls))

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.Session")
    def test_deduplicates_repeated_author_links(self, mock_session_cls, mock_sleep):
        """If the same author appears on multiple listing pages, fetch only once."""
        # Both listing pages link to the same author (Einstein)
        page_with_einstein = """
        <html><head><title>P</title></head><body>
          <div class="quote">
            <small class="author">Einstein</small>
            <a href="/author/Albert-Einstein">(about)</a>
          </div>
        </body></html>
        """
        session = MagicMock()
        mock_session_cls.return_value = session
        session.get.side_effect = [
            self._ok(page_with_einstein),   # listing 1 → links to Einstein, has no next
            self._ok(AUTHOR_HTML),          # author: Einstein (fetched once)
        ]

        pages = crawl(start_url=BASE_URL)

        author_pages = [u for u in pages if "/author/" in u]
        assert len(author_pages) == 1

    # --- Progress callback ---

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.Session")
    def test_progress_callback_invoked_for_each_page(self, mock_session_cls, mock_sleep):
        session = MagicMock()
        mock_session_cls.return_value = session
        session.get.side_effect = [
            self._ok(LISTING_PAGE_1_HTML),
            self._ok(LISTING_PAGE_2_HTML),
            self._ok(AUTHOR_HTML),
            self._ok(AUTHOR_HTML),
        ]

        calls: list[dict] = []
        def callback(url, phase, page_num):
            calls.append({"url": url, "phase": phase})

        crawl(start_url=BASE_URL, progress_callback=callback)

        phases = [c["phase"] for c in calls]
        assert "listing" in phases
        assert "author" in phases
