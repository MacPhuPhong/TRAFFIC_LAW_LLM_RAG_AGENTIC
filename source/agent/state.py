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

# Vehicle type detection — binds retrieval + generator to specific Điều:
#   o_to       → Điều 6 NĐ 168 (ô tô, xe tương tự, xe tải, xe khách)
#   mo_to      → Điều 7 NĐ 168 (xe mô tô, gắn máy, xe máy điện)
#   chuyen_dung → Điều 8 NĐ 168 (xe máy chuyên dùng, máy kéo)
#   xe_dap     → Điều 9 NĐ 168 (xe đạp, xe thô sơ)
#   any        → no constraint (default; câu hỏi không nêu loại xe)
VehicleType = Literal["o_to", "mo_to", "chuyen_dung", "xe_dap", "any"]


class AgentState(TypedDict, total=False):
    query: str
    raw_query: str
    expanded_query: str          # joined form, kept for API back-compat + broad-query detection
    expanded_queries: list[str]  # multi-query rewrites — primary input to multi-retrieval
    chat_history: list[dict]
    category: Category
    intent: Intent               # fine-grained intent within legal_rag (or None)
    vehicle_type: VehicleType    # which Điều in NĐ 168 to bias retrieval + generator
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
