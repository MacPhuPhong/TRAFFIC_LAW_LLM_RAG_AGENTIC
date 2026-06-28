# -*- coding: utf-8 -*-
"""Enable LangSmith tracing from a notebook in one line.

    from research.utils.langsmith_setup import enable_tracing
    enable_tracing(project="traffic-rag-research", run_name="rq1-gemini-only")

Honours `LANGCHAIN_API_KEY` / `LANGSMITH_API_KEY` from the process env or .env.
Silently no-ops if no key is set.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    here = Path(__file__).resolve()
    for candidate in (
        here.parent.parent.parent.parent / ".env",   # repo root
        here.parent.parent.parent / ".env",          # traffic_rag/.env
    ):
        if candidate.exists():
            load_dotenv(candidate, override=False)


def enable_tracing(
    project: str = "traffic-rag-research",
    run_name: str | None = None,
    endpoint: str = "https://api.smith.langchain.com",
) -> bool:
    """Turn on LangChain tracing v2. Returns True if enabled, False otherwise."""
    _load_env()
    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        logger.warning("LangSmith key missing; tracing disabled. "
                       "Set LANGCHAIN_API_KEY in .env to enable.")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_ENDPOINT"] = endpoint
    os.environ["LANGCHAIN_PROJECT"] = project
    if run_name:
        os.environ["LANGCHAIN_RUN_NAME"] = run_name
    logger.info("LangSmith tracing ON · project=%s · run=%s", project, run_name or "-")
    return True


def disable_tracing() -> None:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


def verify_connection() -> dict:
    """Smoke-test the LangSmith connection.

    Returns {"ok": bool, "project": str, "url": str | None, "error": str | None}.
    Useful from a notebook to confirm credentials are wired correctly before
    burning a long eval run.
    """
    _load_env()
    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    project = os.getenv("LANGCHAIN_PROJECT", "default")
    if not api_key:
        return {"ok": False, "project": project, "url": None,
                "error": "LANGCHAIN_API_KEY missing"}
    try:
        from langsmith import Client
    except ImportError:
        return {"ok": False, "project": project, "url": None,
                "error": "pip install langsmith"}
    try:
        client = Client(api_key=api_key)
        # Cheap call that exercises auth without listing every run.
        list(client.list_projects(limit=1))
        return {"ok": True, "project": project,
                "url": f"https://smith.langchain.com/o/-/projects/p/{project}",
                "error": None}
    except Exception as exc:
        return {"ok": False, "project": project, "url": None,
                "error": f"{type(exc).__name__}: {exc}"}
