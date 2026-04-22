# -*- coding: utf-8 -*-
"""Pydantic request/response schemas for the FastAPI layer."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., description="User question in Vietnamese")
    thread_id: str | None = Field(
        None,
        description="Optional thread id to continue a conversation / resume an HITL pause",
    )


ChatStatus = Literal[
    "completed",
    "pending_web_review",
    "rejected",
]


class ChatResponse(BaseModel):
    thread_id: str
    status: ChatStatus
    answer: str | None = None
    draft_answer: str | None = Field(
        None,
        description="Populated when status=pending_web_review — the synthesised web answer "
        "waiting for human approval before being shown to the user.",
    )
    sources: list[dict[str, Any]] = Field(default_factory=list)
    category: str | None = None
    requires_approval: bool = False
    model_info: str | None = None
    expanded_query: str | None = None
    error: str | None = None


class ResumeRequest(BaseModel):
    approved: bool = Field(..., description="True to continue, False to reject")
    override: dict[str, Any] | None = Field(
        None,
        description="Optional partial-state dict to patch before resuming. "
        "Useful for editing `draft_answer` before approval.",
    )


class PendingResponse(BaseModel):
    thread_id: str
    next_nodes: list[str]
    values: dict[str, Any]
    requires_approval: bool
