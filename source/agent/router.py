# -*- coding: utf-8 -*-
"""
router.py — Intent classification (4-route Router).
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .state import Category

logger = logging.getLogger(__name__)


ROUTER_SYSTEM_PROMPT = """Bạn là AI Điều phối viên (Router) cho Hệ thống Tư vấn Luật Giao thông Việt Nam.
Nhiệm vụ của bạn là phân tích câu hỏi của người dùng và phân loại vào đúng 1 trong 4 danh mục:

1. "legal_rag": Câu hỏi về luật lệ giao thông Việt Nam, mức phạt, quy định đăng ký, đăng kiểm,
   trừ điểm, các Điều/Khoản trong Luật Giao thông, Nghị định, Thông tư của Việt Nam.

2. "chit_chat": Chào hỏi (xin chào, hello), cảm ơn, khen chê xã giao, hỏi thăm,
   hỏi về bản thân trợ lý (bạn là ai, bạn làm được gì).

3. "web_legal_search": Luật giao thông ngoài lãnh thổ Việt Nam (Nhật, Mỹ, châu Âu...),
   tin tức giao thông rất mới chưa có trong corpus nội bộ, hoặc quy định vừa ban hành
   trong vòng 1-2 tháng gần đây.

4. "out_of_scope": Hoàn toàn không liên quan đến luật và xe cộ (nấu ăn, giải trí,
   lập trình, y tế, học hành, công việc khác, v.v.).

Chỉ trả về một trong 4 giá trị: "legal_rag", "chit_chat", "web_legal_search", "out_of_scope".
"""


class RouterOutput(BaseModel):
    """Structured router response."""

    category: Category = Field(
        description=(
            "One of: legal_rag, chit_chat, web_legal_search, out_of_scope"
        )
    )


def build_router(llm):
    """Return a callable `classify(query: str) -> Category`.

    Uses LangChain's `with_structured_output` so we get a typed Pydantic result
    instead of parsing raw JSON text.
    """
    structured = llm.with_structured_output(RouterOutput)

    def classify(query: str) -> Category:
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"User Query: {query}"),
        ]
        try:
            result: RouterOutput = structured.invoke(messages)
            return result.category
        except Exception as exc:
            logger.warning("Router classification failed (%s); defaulting to legal_rag", exc)
            return "legal_rag"

    return classify
