# -*- coding: utf-8 -*-
"""Tiny launcher that enables Gemini throttle before running a script.

Usage:
    python research/scripts/_run_throttled.py research/scripts/rq5_prompts_v2.py
"""
import logging
import runpy
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT = Path(__file__).resolve().parent.parent.parent  # traffic_rag/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.utils.throttle import enable  # noqa: E402

enable(sleep_between_calls=2.0, max_retries=5)

if len(sys.argv) < 2:
    print("usage: _run_throttled.py <script.py> [args...]", file=sys.stderr)
    sys.exit(2)

target = sys.argv[1]
sys.argv = sys.argv[1:]  # mimic direct invocation
runpy.run_path(target, run_name="__main__")
