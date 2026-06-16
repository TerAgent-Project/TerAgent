"""teragent.tools.builtin.web — Web interaction tools

Provides two web tools:
  - WebSearchTool: Search the web via configurable search API (READ_ONLY)
  - WebScrapeTool: Fetch and extract web page content (READ_ONLY)

Both tools gracefully handle missing dependencies (httpx) and
unconfigured APIs by returning informative error messages rather
than raising exceptions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any
from urllib.parse import quote_plus

from teragent.core.types import ToolSafety
from teragent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

__all__ = [
    "WebSearchTool",
    "WebScrapeTool",
]


def _check_httpx() -> Any:
    """Check if httpx is available and return the module.

    Returns:
        httpx module if available, None otherwise.
    """
    try:
        import httpx
        return httpx
    except ImportError:
        return None


class WebSearchTool(BaseTool):
    """Search the web via configurable search API — READ_ONLY, concurrency safe.

    Supports multiple search backends:
      - SearXNG (self-hosted, default if SEARXNG_URL is set)
      - SerpAPI (via SERPAPI_KEY environment variable)
      - Generic search API (configurable via SEARCH_API_URL)

    Falls back gracefully when no API is configured.

    Usage::

        tool = WebSearchTool()
        result = await tool.execute({"query": "Python async best practices"})
    """

    name = "web_search"
    description = (
        "Search the web for information. "
        "Requires a configured search API (SearXNG, SerpAPI, or custom). "
        "Returns a list of search results with titles, URLs, and snippets."
    )
    _safety = ToolSafety.READ_ONLY
    _concurrency_safe = True

    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    async def execute(
        self,
        params: dict,
        progress_callback: Any | None = None,
    ) -> ToolResult:
        """Search the web for information.

        Args:
            params: Must include 'query'. Optional 'max_results' (default 5).
            progress_callback: Not used.

        Returns:
            ToolResult with search results in data['results'].
        """
        query = params.get("query", "")
        max_results = params.get("max_results", 5)

        if not query:
            return ToolResult(
                success=False,
                error="Parameter 'query' is required",
                safety=self._safety,
            )

        httpx = _check_httpx()
        if httpx is None:
            return ToolResult(
                success=False,
                error=(
                    "httpx is not installed. Install it with: pip install httpx. "
                    "Web search requires an HTTP client."
                ),
                safety=self._safety,
            )

        # Try SearXNG first
        searxng_url = os.environ.get("SEARXNG_URL")
        if searxng_url:
            return await self._search_searxng(httpx, query, max_results, searxng_url)

        # Try SerpAPI
        serpapi_key = os.environ.get("SERPAPI_KEY")
        if serpapi_key:
            return await self._search_serpapi(httpx, query, max_results, serpapi_key)

        # Try generic search API
        search_api_url = os.environ.get("SEARCH_API_URL")
        if search_api_url:
            return await self._search_generic(httpx, query, max_results, search_api_url)

        return ToolResult(
            success=False,
            error=(
                "No search API configured. Set one of: "
                "SEARXNG_URL (SearXNG instance), SERPAPI_KEY (SerpAPI), "
                "or SEARCH_API_URL (custom search endpoint)."
            ),
            safety=self._safety,
        )

    async def _search_searxng(
        self,
        httpx: Any,
        query: str,
        max_results: int,
        base_url: str,
    ) -> ToolResult:
        """Search using SearXNG instance."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    f"{base_url.rstrip('/')}/search",
                    params={
                        "q": query,
                        "format": "json",
                        "categories": "general",
                    },
                )
                response.raise_for_status()
                data = response.json()

                results = []
                for item in data.get("results", [])[:max_results]:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("content", ""),
                    })

                return ToolResult(
                    success=True,
                    data={
                        "results": results,
                        "query": query,
                        "engine": "searxng",
                    },
                    safety=self._safety,
                )

        except Exception as e:
            logger.error(f"WebSearchTool SearXNG search failed: {e}")
            return ToolResult(
                success=False,
                error=f"SearXNG search failed: {e}",
                safety=self._safety,
            )

    async def _search_serpapi(
        self,
        httpx: Any,
        query: str,
        max_results: int,
        api_key: str,
    ) -> ToolResult:
        """Search using SerpAPI."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    "https://serpapi.com/search",
                    params={
                        "q": query,
                        "api_key": api_key,
                        "engine": "google",
                        "num": max_results,
                    },
                )
                response.raise_for_status()
                data = response.json()

                results = []
                for item in data.get("organic_results", [])[:max_results]:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                    })

                return ToolResult(
                    success=True,
                    data={
                        "results": results,
                        "query": query,
                        "engine": "serpapi",
                    },
                    safety=self._safety,
                )

        except Exception as e:
            logger.error(f"WebSearchTool SerpAPI search failed: {e}")
            return ToolResult(
                success=False,
                error=f"SerpAPI search failed: {e}",
                safety=self._safety,
            )

    async def _search_generic(
        self,
        httpx: Any,
        query: str,
        max_results: int,
        api_url: str,
    ) -> ToolResult:
        """Search using a generic search API endpoint."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    api_url,
                    params={"q": query, "max_results": max_results},
                )
                response.raise_for_status()
                data = response.json()

                # Assume the API returns {"results": [...]}
                results = data.get("results", [])[:max_results]

                return ToolResult(
                    success=True,
                    data={
                        "results": results,
                        "query": query,
                        "engine": "generic",
                    },
                    safety=self._safety,
                )

        except Exception as e:
            logger.error(f"WebSearchTool generic search failed: {e}")
            return ToolResult(
                success=False,
                error=f"Search API failed: {e}",
                safety=self._safety,
            )

    def describe_usage(self, params: dict) -> str:
        query = params.get("query", "?")
        return f"搜索: {query[:50]}"


class WebScrapeTool(BaseTool):
    """Fetch and extract web page content — READ_ONLY, concurrency safe.

    Uses httpx to fetch web pages and extracts text content from HTML.
    Handles basic HTML parsing without requiring BeautifulSoup (falls back
    to regex-based extraction).

    Usage::

        tool = WebScrapeTool()
        result = await tool.execute({"url": "https://example.com"})
    """

    name = "web_scrape"
    description = (
        "Fetch and extract text content from a web page. "
        "Returns the page title, text content, and metadata."
    )
    _safety = ToolSafety.READ_ONLY
    _concurrency_safe = True

    parameters_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL of the web page to scrape",
            },
            "max_length": {
                "type": "integer",
                "description": "Maximum content length in characters (default: 50000)",
                "default": 50000,
            },
        },
        "required": ["url"],
    }

    async def execute(
        self,
        params: dict,
        progress_callback: Any | None = None,
    ) -> ToolResult:
        """Fetch and extract web page content.

        Args:
            params: Must include 'url'. Optional 'max_length' (default 50000).
            progress_callback: Not used.

        Returns:
            ToolResult with page content in data['content'].
        """
        url = params.get("url", "")
        max_length = params.get("max_length", 50000)

        if not url:
            return ToolResult(
                success=False,
                error="Parameter 'url' is required",
                safety=self._safety,
            )

        # Basic URL validation
        if not url.startswith(("http://", "https://")):
            return ToolResult(
                success=False,
                error=f"Invalid URL: {url}. Must start with http:// or https://",
                safety=self._safety,
            )

        httpx = _check_httpx()
        if httpx is None:
            return ToolResult(
                success=False,
                error=(
                    "httpx is not installed. Install it with: pip install httpx. "
                    "Web scraping requires an HTTP client."
                ),
                safety=self._safety,
            )

        try:
            async with httpx.AsyncClient(
                timeout=15.0,
                follow_redirects=True,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; TerAgent/1.0; +https://github.com/teragent)"
                    ),
                },
            ) as client:
                response = await client.get(url)
                response.raise_for_status()

                html = response.text

                # Try BeautifulSoup first, fall back to regex extraction
                title, text_content = self._extract_content(html)

                # Truncate if needed
                truncated = False
                if len(text_content) > max_length:
                    text_content = text_content[:max_length] + "\n... [truncated]"
                    truncated = True

                data = {
                    "url": str(response.url),
                    "title": title,
                    "content": text_content,
                    "status_code": response.status_code,
                    "content_type": response.headers.get("content-type", ""),
                }
                if truncated:
                    data["truncated"] = True

                return ToolResult(
                    success=True,
                    data=data,
                    safety=self._safety,
                )

        except httpx.HTTPStatusError as e:
            return ToolResult(
                success=False,
                error=f"HTTP error {e.response.status_code}: {e.response.reason_phrase}",
                safety=self._safety,
            )
        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                error=f"Request timed out for {url}",
                safety=self._safety,
            )
        except Exception as e:
            logger.error(f"WebScrapeTool failed for {url}: {e}")
            return ToolResult(
                success=False,
                error=f"Failed to fetch page: {e}",
                safety=self._safety,
            )

    @staticmethod
    def _extract_content(html: str) -> tuple[str, str]:
        """Extract title and text content from HTML.

        Tries BeautifulSoup first, falls back to regex-based extraction.

        Args:
            html: Raw HTML string

        Returns:
            Tuple of (title, text_content)
        """
        # Try BeautifulSoup
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            title = soup.title.string.strip() if soup.title and soup.title.string else ""

            # Get text content
            text = soup.get_text(separator="\n", strip=True)

            # Clean up multiple blank lines
            text = re.sub(r"\n{3,}", "\n\n", text)

            return title, text

        except ImportError:
            # Fall back to regex-based extraction
            pass

        # Regex-based extraction
        # Extract title
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""

        # Remove script and style blocks
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.IGNORECASE | re.DOTALL)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Decode HTML entities
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        text = text.replace("&#39;", "'")
        text = text.replace("&nbsp;", " ")

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return title, text

    def describe_usage(self, params: dict) -> str:
        url = params.get("url", "?")
        return f"抓取网页: {url[:60]}"
