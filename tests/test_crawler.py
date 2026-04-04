"""
test_crawler.py — Unit tests for src/crawler.py.

All HTTP calls are mocked so tests run without network access.
"""

import time
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
)
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


SIMPLE_LISTING_HTML = """
<html>
  <head><title>Quotes Page 1</title></head>
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

LAST_PAGE_HTML = """
<html>
  <head><title>Quotes Page 10</title></head>
  <body>
    <div class="quote">
      <span class="text">Another quote.</span>
      <small class="author">Mark Twain</small>
      <a href="/author/Mark-Twain">(about)</a>
    </div>
  </body>
</html>
"""

AUTHOR_HTML = """
<html>
  <head><title>Albert Einstein</title></head>
  <body>
    <h3 class="author-title">Albert Einstein</h3>
    <span class="author-description">Born in Germany...</span>
  </body>
</html>
"""


# ---------------------------------------------------------------------------
# fetch_page tests
# ---------------------------------------------------------------------------

class TestFetchPage:
    def test_returns_soup_on_success(self):
        mock_resp = MagicMock()
        mock_resp.text = "<html><body>Hello</body></html>"
        mock_resp.raise_for_status = MagicMock()

        session = MagicMock()
        session.get.return_value = mock_resp

        result = fetch_page("https://example.com", session)
        assert result is not None
        assert result.find("body").text == "Hello"

    def test_returns_none_after_all_retries_fail(self):
        session = MagicMock()
        session.get.side_effect = requests.exceptions.ConnectionError("fail")

        with patch("src.crawler.time.sleep"):
            result = fetch_page("https://example.com", session)

        assert result is None

    def test_retries_on_http_error(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500")

        session = MagicMock()
        session.get.return_value = mock_resp

        with patch("src.crawler.time.sleep"):
            result = fetch_page("https://example.com", session)

        assert result is None
        assert session.get.call_count == 3  # MAX_RETRIES

    def test_returns_none_on_timeout(self):
        session = MagicMock()
        session.get.side_effect = requests.exceptions.Timeout()

        with patch("src.crawler.time.sleep"):
            result = fetch_page("https://example.com", session)

        assert result is None


# ---------------------------------------------------------------------------
# extract_page_text tests
# ---------------------------------------------------------------------------

class TestExtractPageText:
    def test_removes_script_and_style(self):
        soup = make_soup("<html><body><script>alert(1)</script><p>Hello</p></body></html>")
        text = extract_page_text(soup)
        assert "alert" not in text
        assert "Hello" in text

    def test_normalises_whitespace(self):
        soup = make_soup("<html><body>  foo   bar  </body></html>")
        text = extract_page_text(soup)
        assert "  " not in text
        assert "foo bar" in text

    def test_empty_page(self):
        soup = make_soup("<html><body></body></html>")
        text = extract_page_text(soup)
        assert text == ""


# ---------------------------------------------------------------------------
# extract_author_urls tests
# ---------------------------------------------------------------------------

class TestExtractAuthorUrls:
    def test_extracts_author_urls(self):
        soup = make_soup(SIMPLE_LISTING_HTML)
        urls = extract_author_urls(soup)
        assert "https://quotes.toscrape.com/author/Albert-Einstein" in urls

    def test_returns_empty_list_when_no_authors(self):
        soup = make_soup("<html><body><p>No authors here</p></body></html>")
        urls = extract_author_urls(soup)
        assert urls == []

    def test_does_not_include_non_author_links(self):
        html = '<html><body><a href="/tag/love/">love</a></body></html>'
        soup = make_soup(html)
        urls = extract_author_urls(soup)
        assert urls == []


# ---------------------------------------------------------------------------
# get_next_page_url tests
# ---------------------------------------------------------------------------

class TestGetNextPageUrl:
    def test_returns_next_url(self):
        soup = make_soup(SIMPLE_LISTING_HTML)
        url = get_next_page_url(soup)
        assert url == "https://quotes.toscrape.com/page/2/"

    def test_returns_none_on_last_page(self):
        soup = make_soup(LAST_PAGE_HTML)
        url = get_next_page_url(soup)
        assert url is None


# ---------------------------------------------------------------------------
# crawl integration tests (all network calls mocked)
# ---------------------------------------------------------------------------

class TestCrawl:
    def _make_response(self, html: str) -> MagicMock:
        resp = MagicMock()
        resp.text = html
        resp.raise_for_status = MagicMock()
        return resp

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.Session")
    def test_crawl_two_listing_pages_and_one_author(self, mock_session_cls, mock_sleep):
        session = MagicMock()
        mock_session_cls.return_value = session

        session.get.side_effect = [
            self._make_response(SIMPLE_LISTING_HTML),   # page 1
            self._make_response(AUTHOR_HTML),            # author page
            self._make_response(LAST_PAGE_HTML),         # page 2 (no next)
        ]

        pages = crawl(start_url=BASE_URL)

        assert len(pages) >= 1
        assert any("page" in url or "author" in url for url in pages)

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.Session")
    def test_politeness_window_respected(self, mock_session_cls, mock_sleep):
        session = MagicMock()
        mock_session_cls.return_value = session

        session.get.side_effect = [
            self._make_response(SIMPLE_LISTING_HTML),
            self._make_response(AUTHOR_HTML),
            self._make_response(LAST_PAGE_HTML),
        ]

        crawl(start_url=BASE_URL)

        # time.sleep must have been called with POLITENESS_WINDOW each time
        for c in mock_sleep.call_args_list:
            assert c == call(POLITENESS_WINDOW)

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.Session")
    def test_skips_failed_pages_gracefully(self, mock_session_cls, mock_sleep):
        session = MagicMock()
        mock_session_cls.return_value = session

        # First page fails, second page succeeds (no next link)
        session.get.side_effect = [
            requests.exceptions.ConnectionError(),
            requests.exceptions.ConnectionError(),
            requests.exceptions.ConnectionError(),
        ]

        pages = crawl(start_url=BASE_URL)
        assert pages == {}

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.Session")
    def test_does_not_visit_same_url_twice(self, mock_session_cls, mock_sleep):
        session = MagicMock()
        mock_session_cls.return_value = session

        session.get.return_value = self._make_response(LAST_PAGE_HTML)

        crawl(start_url=BASE_URL)

        visited_urls = [c.args[0] for c in session.get.call_args_list]
        assert len(visited_urls) == len(set(visited_urls))
