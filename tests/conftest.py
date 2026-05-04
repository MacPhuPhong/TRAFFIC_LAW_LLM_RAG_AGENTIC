# -*- coding: utf-8 -*-
"""Pytest fixtures + Qdrant availability guard.

Tests marked with `@pytest.mark.requires_qdrant` are auto-skipped when Qdrant
is unreachable (i.e. in CI without a docker-compose). Other tests (imports,
config) run unconditionally so the smoke layer always executes in CI.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import pytest

# Make `source.*` importable regardless of CWD.
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent  # traffic_rag/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _qdrant_reachable(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def qdrant_ready() -> bool:
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return _qdrant_reachable(host, port)


@pytest.fixture(autouse=True)
def _skip_if_no_qdrant(request, qdrant_ready):
    """Auto-skip @requires_qdrant tests when Qdrant is unreachable."""
    if request.node.get_closest_marker("requires_qdrant") and not qdrant_ready:
        pytest.skip("Qdrant not reachable — skipping retrieval test")
