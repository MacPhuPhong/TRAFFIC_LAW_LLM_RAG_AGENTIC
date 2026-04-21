# -*- coding: utf-8 -*-
"""
tools.py — External tool wrappers (currently just Tavily web search).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


# Vietnamese legal + government + traffic-police domains. Tavily will only
# surface results from this set when include_domains is passed, keeping the
# web fallback anchored to authoritative Vietnamese traffic-law sources.
VN_LEGAL_DOMAINS: tuple[str, ...] = (
    "thuvienphapluat.vn",
    "luatvietnam.vn",
    "vbpl.vn",
    "chinhphu.vn",
    "xaydungchinhsach.chinhphu.vn",
    "baochinhphu.vn",
    "moj.gov.vn",
    "mt.gov.vn",
    "csgt.vn",
    "luatgiaothong.vn",
    "tapchitoaan.vn",
    "quochoi.vn",
)


class TavilySearchTool:
    """Thin wrapper around the Tavily Search API, locked to VN legal domains."""

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 5,
        search_depth: str = "advanced",
        include_domains: tuple[str, ...] | list[str] | None = VN_LEGAL_DOMAINS,
    ):
        try:
            from tavily import TavilyClient
        except ImportError as e:
            raise ImportError(
                "tavily-python not installed. Add `tavily-python` to requirements."
            ) from e

        key = api_key or os.environ.get("TAVILY_API_KEY")
        if not key:
            raise ValueError(
                "TavilySearchTool requires TAVILY_API_KEY env var or api_key arg."
            )
        self.client = TavilyClient(api_key=key)
        self.max_results = max_results
        self.search_depth = search_depth
        self.include_domains = list(include_domains) if include_domains else None

    def search(
        self,
        query: str,
        max_results: int | None = None,
        include_domains: list[str] | None = None,
    ) -> dict:
        """Run Tavily search restricted to authoritative Vietnamese legal sites.

        Shape: {"answer": Optional[str], "results": [{"title","url","content"}, ...]}
        """
        domains = include_domains if include_domains is not None else self.include_domains
        return self.client.search(
            query=query,
            search_depth=self.search_depth,
            max_results=max_results or self.max_results,
            include_domains=domains,
        )
