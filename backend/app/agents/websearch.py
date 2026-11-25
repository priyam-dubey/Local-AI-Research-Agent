"""
Web search for the research agent (NOT used by document QA).

Uses the maintained `ddgs` package, with the legacy `duckduckgo_search` package as a
fallback for existing installs. Returns normalised dicts: {"title", "href", "body"}.
Note: the query text is sent to DuckDuckGo even when the LLM provider is local.
"""

from __future__ import annotations

import importlib
from typing import Any

from loguru import logger


class WebSearchUnavailable(Exception):
    pass


def _client() -> Any:
    for module in ("ddgs", "duckduckgo_search"):
        try:
            return importlib.import_module(module).DDGS()
        except ImportError:
            continue
    raise WebSearchUnavailable("Web search needs the `ddgs` package (pip install -r requirements.txt).")


def search_web(query: str, max_results: int = 5) -> list[dict]:
    raw = _client().text(query, max_results=max_results)
    out: list[dict] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        href = str(item.get("href") or item.get("link") or item.get("url") or "").strip()
        body = str(item.get("body") or item.get("snippet") or "").strip()
        if title or body:
            out.append({"title": title, "href": href, "body": body})
    logger.debug(f"Web search returned {len(out)} result(s).")
    return out[:max_results]
