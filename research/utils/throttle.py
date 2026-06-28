# -*- coding: utf-8 -*-
"""Runtime throttle/retry for LegalAnswerGenerator against the Gemini free tier.

The free tier limits `gemini-3.1-flash-lite` to 250k input tokens/min. Long-running
RQ scripts (RQ1, RQ5, RQ7, RQ9) easily breach that. Importing this module and
calling `enable()` monkey-patches `LegalAnswerGenerator.generate` to:
  - sleep `sleep_between_calls` seconds before every call (smooth out RPM),
  - on 429 / ResourceExhausted, back off (15s, 30s, 60s, ...) and retry up to
    `max_retries` times.

Usage (top of any research script, or via the launcher):
    from research.utils.throttle import enable
    enable()
"""
from __future__ import annotations

import logging
import time

log = logging.getLogger("throttle")
_PATCHED = False


def _is_rate_limit(err: BaseException) -> bool:
    msg = str(err)
    return any(t in msg for t in ("429", "ResourceExhausted", "RATE_LIMIT_EXCEEDED", "quota"))


def enable(sleep_between_calls: float = 2.0, max_retries: int = 5) -> None:
    """Patch `LegalAnswerGenerator.generate` and `ChatGoogleGenerativeAI.invoke`
    so any Gemini call automatically pauses + retries on 429.
    Idempotent: safe to call multiple times.
    """
    global _PATCHED
    if _PATCHED:
        return

    try:
        from source.rag_core.generator import LegalAnswerGenerator  # noqa: F401
    except ImportError:
        log.warning("LegalAnswerGenerator import failed — throttle skipped.")
        return

    _orig_generate = LegalAnswerGenerator.generate

    def _wrapped_generate(self, *args, **kwargs):
        # Forward all args/kwargs so callers passing intent=, vehicle=, etc.
        # (introduced in v6 agentic pipeline) still work.
        for attempt in range(max_retries):
            if sleep_between_calls > 0:
                time.sleep(sleep_between_calls)
            try:
                return _orig_generate(self, *args, **kwargs)
            except Exception as e:
                if not _is_rate_limit(e):
                    raise
                wait = 15 * (2 ** attempt)
                log.warning("[throttle] 429 hit on attempt %d, sleeping %ds: %s",
                            attempt + 1, wait, str(e)[:120])
                time.sleep(wait)
        raise RuntimeError(f"LegalAnswerGenerator.generate exceeded {max_retries} retries on 429")

    LegalAnswerGenerator.generate = _wrapped_generate  # type: ignore[assignment]

    # Also patch the raw ChatGoogleGenerativeAI client used by GeminiOnlyPipeline /
    # AgenticRAGPipeline / RQ5 prompt variants.
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        log.warning("langchain_google_genai not installed — only generator patched.")
        _PATCHED = True
        return

    _orig_invoke = ChatGoogleGenerativeAI.invoke

    def _wrapped_invoke(self, *args, **kwargs):
        for attempt in range(max_retries):
            if sleep_between_calls > 0:
                time.sleep(sleep_between_calls)
            try:
                return _orig_invoke(self, *args, **kwargs)
            except Exception as e:
                if not _is_rate_limit(e):
                    raise
                wait = 15 * (2 ** attempt)
                log.warning("[throttle] 429 on ChatGoogleGenerativeAI.invoke attempt %d, sleeping %ds",
                            attempt + 1, wait)
                time.sleep(wait)
        raise RuntimeError(f"ChatGoogleGenerativeAI.invoke exceeded {max_retries} retries on 429")

    ChatGoogleGenerativeAI.invoke = _wrapped_invoke  # type: ignore[assignment]
    _PATCHED = True
    log.info("[throttle] enabled: sleep=%.1fs, retries=%d", sleep_between_calls, max_retries)
