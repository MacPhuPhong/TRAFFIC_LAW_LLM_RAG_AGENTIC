# -*- coding: utf-8 -*-
"""
state.py — Agent shared state schema.
"""

from __future__ import annotations

from typing import Literal, TypedDict

Category = Literal["legal_rag", "chit_chat", "web_legal_search", "out_of_scope"]


class AgentState(TypedDict, total=False):
    query: str
    raw_query: str
    expanded_query: str
    chat_history: list[dict]
    category: Category
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
