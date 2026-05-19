# -*- coding: utf-8 -*-
"""
state.py — Agent shared state schema.
"""

from __future__ import annotations

from typing import Literal, TypedDict

Category = Literal["legal_rag", "chit_chat", "web_legal_search", "out_of_scope"]

# Fine-grained intent inside legal_rag — drives downstream gating:
#   penalty     → confidence judge demands money amount in chunks; generator highlights số tiền
#   fault       → activate Rule 14 (phân định lỗi 4 bước)
#   procedure   → allow procedural chunks (TT 72, Luật 36 procedural articles)
#   definition  → simpler retrieval, no need to enumerate cases
#   list        → broad top_k=25, drop keyword-overlap check
#   mixed       → multiple intents in one question (treat union)
Intent = Literal["penalty", "fault", "procedure", "definition", "list", "mixed"]


class AgentState(TypedDict, total=False):
    query: str
    raw_query: str
    expanded_query: str          # joined form, kept for API back-compat + broad-query detection
    expanded_queries: list[str]  # multi-query rewrites — primary input to multi-retrieval
    chat_history: list[dict]
    category: Category
    intent: Intent               # fine-grained intent within legal_rag (or None)
    multi_frame: bool            # True when ≥2 distinct (Điều, Khoản) in final chunks
    chunks: list[dict]
    answer: str
    draft_answer: str
    sources: list[dict]
    refused: bool
    web_results: list[dict]
    warning_prefix: str
    requires_approval: bool
    model_info: str
    error: str
