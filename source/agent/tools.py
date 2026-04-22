# -*- coding: utf-8 -*-
"""
tools.py — External tool wrappers (currently just Tavily web search).
"""

from __future__ import annotations

import logging
import os
import time

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


# Errors worth retrying (transient network / server-side). Anything else
# (auth, malformed request) should surface immediately.
_TRANSIENT_EXC_NAMES: tuple[str, ...] = (
    "Timeout",
    "ConnectTimeout",
    "ReadTimeout",
    "ConnectionError",
    "ConnectionResetError",
    "RemoteDisconnected",
    "ProtocolError",
    "ChunkedEncodingError",
)


def _is_transient(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in _TRANSIENT_EXC_NAMES:
        return True
    msg = str(exc).lower()
    return "timeout" in msg or "timed out" in msg or "connection" in msg


class TavilySearchTool:
    """Thin wrapper around the Tavily Search API, locked to VN legal domains.

    Adds a configurable per-request `timeout` (forwarded to the SDK) and a
    small retry loop for transient network errors so a flaky connection
    doesn't immediately degrade the web-fallback path.
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 5,
        search_depth: str = "advanced",
        include_domains: tuple[str, ...] | list[str] | None = VN_LEGAL_DOMAINS,
        timeout: float = 120.0,
        max_retries: int = 2,
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
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))

    def search(
        self,
        query: str,
        max_results: int | None = None,
        include_domains: list[str] | None = None,
    ) -> dict:
        """Run Tavily search restricted to authoritative Vietnamese legal sites.

        Shape: {"answer": Optional[str], "results": [{"title","url","content"}, ...]}

        Retries transient network errors (timeout / connection reset) up to
        `max_retries` times with linear backoff (1s, 2s, ...). Non-transient
        errors are raised immediately so the caller can surface them.
        """
        domains = include_domains if include_domains is not None else self.include_domains
        attempts = self.max_retries + 1
        last_exc: BaseException | None = None

        for attempt in range(1, attempts + 1):
            try:
                return self.client.search(
                    query=query,
                    search_depth=self.search_depth,
                    max_results=max_results or self.max_results,
                    include_domains=domains,
                    timeout=self.timeout,
                )
            except Exception as exc:
                last_exc = exc
                if not _is_transient(exc) or attempt >= attempts:
                    raise
                backoff = float(attempt)
                logger.warning(
                    "Tavily transient error (%s: %s) — retrying %d/%d after %.1fs",
                    type(exc).__name__,
                    exc,
                    attempt,
                    self.max_retries,
                    backoff,
                )
                time.sleep(backoff)

        # Unreachable — loop either returns or raises — but keep mypy happy.
        assert last_exc is not None
        raise last_exc
