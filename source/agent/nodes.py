# -*- coding: utf-8 -*-
"""
nodes.py — LangGraph node callables for the agentic RAG workflow.

Each node receives AgentState and returns a partial-state dict that
LangGraph merges back into the global state.
"""

from __future__ import annotations

import logging
import re
from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .router import build_router
from .state import AgentState, Category


logger = logging.getLogger(__name__)


# --- Soft risk tagging -------------------------------------------------------
#
# Risk detection no longer blocks: the legal_rag path trusts the cited corpus
# (Nghị định 168/2024 + Luật ATGT) to return authoritative answers. Instead,
# `risk_tag_node` sets a UI-only warning flag so the frontend can display a
# cautionary pill above the answer. The REAL HITL gate now sits on the web
# fallback path (see `web_finalize_node`) where content quality is uncertain.

RISK_PATTERNS: tuple[str, ...] = (
    r"tịch thu",
    r"hình sự",
    r"truy cứu trách nhiệm",
    r"tạm giữ phương tiện",
    r"tước quyền sử dụng",
)


def risk_tag_node(state: AgentState) -> dict:
    """Tag queries that touch heavy sanctions with `risk_flag=True` so the UI
    can warn users that the answer involves serious penalties. Non-blocking:
    the graph continues straight to legal_rag regardless."""
    q = (state.get("query") or "").lower()
    flagged = any(re.search(p, q) for p in RISK_PATTERNS)
    if flagged:
        logger.info("Risk tag: query touches heavy-sanction topics")
    return {"risk_flag": flagged}


# --- Prompts -----------------------------------------------------------------

CHIT_CHAT_SYSTEM_PROMPT = """Bạn là Trợ lý Luật Giao thông Việt Nam thân thiện.
Trả lời ngắn gọn 1-2 câu, thân mật, bằng tiếng Việt. Gợi ý người dùng có thể
hỏi về luật giao thông (mức phạt, quy định đăng ký/đăng kiểm, v.v.) nếu phù hợp.
KHÔNG trích dẫn Điều/Khoản luật ở đây; không bịa thông tin pháp lý cụ thể."""


WEB_SEARCH_SYSTEM_PROMPT = """Bạn là chuyên gia tổng hợp thông tin pháp lý từ Internet.
Tổng hợp câu trả lời dựa trên các kết quả web cung cấp.

QUY TẮC BẮT BUỘC:
1. Ghi LUÔN Ở ĐẦU CÂU dòng cảnh báo: "⚠️ Lưu ý: Thông tin tra cứu từ Internet mở, hãy cẩn trọng."
2. Nếu kết quả web không đủ để trả lời, nói dứt khoát: "Xin lỗi, không có thông tin online xác thực."
3. Trích xuất và liệt kê URLs nguồn ở cuối câu trả lời dưới mục "Nguồn:".
4. Trả lời bằng tiếng Việt, ngắn gọn và rõ ràng."""


OUT_OF_SCOPE_MESSAGE = (
    "Xin lỗi, tôi chỉ tư vấn về Luật Giao thông Việt Nam và không giải quyết các "
    "vấn đề ngoài phạm vi này. Bạn có thể hỏi tôi về mức phạt vi phạm, quy định "
    "đăng ký xe, đăng kiểm, cấp giấy phép lái xe, v.v."
)


class AnalyzerOutput(BaseModel):
    """Consolidated output for the pre-processing analyzer."""
    category: Category = Field(description="Intent category (legal_rag, chit_chat, etc.)")
    standalone_query: str = Field(description="Self-contained rewrite of the query using history.")
    expanded_query: str = Field(description="Formal legal terminology expansion for retrieval.")

ANALYZER_SYSTEM_PROMPT = """Bạn là chuyên gia phân tích truy vấn Luật Giao thông Việt Nam.
Nhiệm vụ: Chấp nhận lịch sử hội thoại + câu hỏi mới, thực hiện 3 bước trong 1:

1. PHÂN LOẠI (Category):
   - 'legal_rag': Hỏi về luật, mức phạt, đăng kiểm, đăng ký xe...
   - 'chit_chat': Xã giao, xin chào...
   - 'web_legal_search': Hỏi về luật nước ngoài hoặc tin tức quốc tế/quá mới.
   - 'out_of_scope': Không liên quan giao thông.

2. CHUẨN HOÁ (Standalone):
   - Viết lại câu hỏi thành câu ĐỘC LẬP dựa trên lịch sử.
   - Ví dụ: "Còn ô tô thì sao?" -> "Mức phạt vượt đèn đỏ đối với xe ô tô là bao nhiêu?"

3. MỞ RỘNG (Expanded):
   - Chuyển đổi ngôn ngữ dân dã sang thuật ngữ chuyên môn pháp lý Việt Nam.
   - CHIẾN LƯỢC MỞ RỘNG TỔNG QUÁT:
     * Với câu hỏi 'Mức phạt/Bị gì': Mở rộng thành "{Hành vi} + mức xử phạt + Nghị định 168/2024".
     * Với câu hỏi 'Khi nào/Trường hợp nào': Mở rộng thành "{Chủ đề} + các hành vi vi phạm + hình thức xử phạt bổ sung + biện pháp khắc phục hậu quả".
     * Với câu hỏi 'Thủ tục/Đâu': Mở rộng thành "{Thủ tục} + trình tự + thẩm quyền + hồ sơ".

QUY TẮC:
- Trả về JSON theo đúng định dạng yêu cầu.
- Luôn giữ nguyên loại phương tiện (ô tô/xe máy/...) trong câu mở rộng.
- Mục tiêu là tạo ra truy vấn có độ phủ (recall) cao nhất trong CSDL luật.
"""


# --- Node factories ----------------------------------------------------------

CONTEXTUALIZE_HISTORY_TURNS = 6


def _format_history_for_prompt(history: list[dict]) -> str:
    lines = []
    for msg in history[-CONTEXTUALIZE_HISTORY_TURNS:]:
        role = msg.get("role", "?")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        label = "user" if role == "user" else "assistant"
        lines.append(f"  {label}: {content}")
        
        # New: If there are sources, list them so the LLM can resolve "Điều số X"
        sources = msg.get("sources") or []
        if role == "assistant" and sources:
            source_labels = []
            for i, s in enumerate(sources, 1):
                parts = []
                if s.get("dieu"): parts.append(f"Điều {s['dieu']}")
                if s.get("khoan"): parts.append(f"Khoan {s['khoan']}")
                if s.get("ten_van_ban"): parts.append(s['ten_van_ban'])
                label_str = " · ".join(parts) or s.get("doc_id", "Nguồn")
                source_labels.append(f"[{i}] {label_str}")
            lines.append(f"    (Nguồn đã trích dẫn: {', '.join(source_labels)})")
            
    return "\n".join(lines) if lines else "(chưa có)"


def make_analyzer_node(llm) -> Callable[[AgentState], dict]:

    """Targeted optimization: consolidates Contextualize, Router, and Expansion
    into one LLM call to solve API timeout issues.
    """
    structured = llm.with_structured_output(AnalyzerOutput)

    def analyzer_node(state: AgentState) -> dict:
        raw = state.get("query", "")
        history = state.get("chat_history") or []
        history_text = _format_history_for_prompt(history)

        try:
            result: AnalyzerOutput = structured.invoke(
                [
                    SystemMessage(content=ANALYZER_SYSTEM_PROMPT),
                    HumanMessage(content=f"Lịch sử hội thoại:\n{history_text}\n\nCâu hỏi mới nhất: {raw}"),
                ]
            )
            return {
                "category": result.category,
                "query": result.standalone_query,
                "expanded_query": result.expanded_query,
                "raw_query": raw,
            }
        except Exception as exc:
            logger.error("Analyzer failed (%s); falling back to raw defaults", exc)
            return {
                "category": "legal_rag", 
                "query": raw, 
                "expanded_query": raw, 
                "raw_query": raw
            }

    return analyzer_node


def route_by_category(state: AgentState) -> str:
    """Edge condition after analyzer_node."""
    return state.get("category", "out_of_scope")




# Listing-type queries ("khi nào", "các trường hợp", "liệt kê", ...) need a
# wider retrieval window than pinpoint queries. Regex is enough — these phrases
# are distinctive in Vietnamese and avoid an extra LLM hop.
BROAD_QUERY_PATTERNS: tuple[str, ...] = (
    r"\bkhi nào\b",
    r"\btrường hợp\b",
    r"\bcác trường hợp\b",
    r"\bnhững trường hợp\b",
    r"\bliệt kê\b",
    r"\bcác lỗi\b",
    r"\blỗi nào\b",
    r"\bnhững lỗi\b",
    r"\bcác hành vi\b",
    r"\bdanh sách\b",
    r"\bđiều kiện\b",
)

LEGAL_RAG_TOP_K_DEFAULT = 10
LEGAL_RAG_TOP_K_BROAD = 20


def _is_broad_query(*texts: str) -> bool:
    blob = " ".join(t for t in texts if t).lower()
    return any(re.search(p, blob) for p in BROAD_QUERY_PATTERNS)


def make_legal_rag_node(retriever, generator) -> Callable[[AgentState], dict]:
    def legal_rag_node(state: AgentState) -> dict:
        query = state["query"]
        retrieval_query = state.get("expanded_query") or query
        top_k = (
            LEGAL_RAG_TOP_K_BROAD
            if _is_broad_query(state.get("raw_query", ""), query, retrieval_query)
            else LEGAL_RAG_TOP_K_DEFAULT
        )
        try:
            # Tăng top_k lên 20 để bao quát đủ các trường hợp cụ thể (tịch thu, tước quyền sử dụng)
            chunks = retriever.get_relevant_chunks(retrieval_query, top_k=20)

        except Exception as exc:
            logger.exception("Retriever failed: %s", exc)
            return {
                "chunks": [],
                "answer": "",
                "sources": [],
                "refused": False,
                "model_info": "",
                "error": f"retriever_error: {exc}",
            }

        chunk_dicts = [c.to_dict() for c in chunks]
        if not chunk_dicts:
            # No context at all in the corpus -> legitimate fallback candidate.
            return {
                "chunks": [],
                "answer": "",
                "sources": [],
                "refused": True,
                "model_info": "",
            }

        try:
            result = generator.generate(query, chunk_dicts)
        except Exception as exc:
            logger.exception("Generator failed: %s", exc)
            return {
                "chunks": chunk_dicts,
                "answer": "",
                "sources": [],
                "refused": False,
                "model_info": "",
                "error": f"generator_error: {exc}",
            }

        return {
            "chunks": chunk_dicts,
            "answer": result["answer"],
            "sources": result["sources"],
            "refused": bool(result.get("refused")),
            "model_info": result.get("model", ""),
        }

    return legal_rag_node


def legal_fallback_router(state: AgentState) -> str:
    """Edge condition after legal_rag_node.

    Fall back to web_search ONLY when the generator genuinely could not answer
    from the local corpus (no chunks, or generator returned refused=True).
    Infra errors (retriever/generator exceptions) short-circuit to END so the
    user sees the real cause instead of a misleading web fallback."""
    if state.get("error"):
        logger.info("legal_rag error -> end (no web fallback for infra errors)")
        return "end"
    if state.get("refused") or not state.get("chunks"):
        logger.info("legal_rag refused/empty -> falling back to web_search")
        return "web_search"
    return "end"


def make_chit_chat_node(llm) -> Callable[[AgentState], dict]:
    def chit_chat_node(state: AgentState) -> dict:
        query = state["query"]
        try:
            resp = llm.invoke(
                [
                    SystemMessage(content=CHIT_CHAT_SYSTEM_PROMPT),
                    HumanMessage(content=query),
                ]
            )
            answer = resp.content.strip() if hasattr(resp, "content") else str(resp)
        except Exception as exc:
            logger.exception("chit_chat LLM call failed: %s", exc)
            answer = (
                "Xin chào! Tôi là Trợ lý Luật Giao thông Việt Nam. "
                "Bạn có thể hỏi tôi về mức phạt, quy định đăng ký xe, đăng kiểm, v.v."
            )
        return {
            "answer": answer,
            "sources": [],
            "refused": False,
        }

    return chit_chat_node


def out_of_scope_node(state: AgentState) -> dict:
    return {
        "answer": OUT_OF_SCOPE_MESSAGE,
        "sources": [],
        "refused": False,
    }


WEB_SEARCH_LEGAL_PREFIX = "Luật giao thông Việt Nam"


def _build_web_search_query(state: AgentState) -> str:
    """Pick the best query for Tavily and anchor it to Vietnamese traffic law.

    Prefers the expanded legal-terminology query; falls back to raw. Always
    prepends a legal-domain prefix so the search stays on-topic even when the
    user input is ambiguous (e.g. a follow-up like "đã lỡ bỏ chạy rồi").
    """
    base = (state.get("expanded_query") or state.get("query") or "").strip()
    if not base:
        return WEB_SEARCH_LEGAL_PREFIX
    if WEB_SEARCH_LEGAL_PREFIX.lower() in base.lower():
        return base
    return f"{WEB_SEARCH_LEGAL_PREFIX}: {base}"


WEB_WARNING_PREFIX = "⚠️ Lưu ý: Thông tin tra cứu từ Internet mở, hãy cẩn trọng."


def make_web_search_node(tavily_tool, llm) -> Callable[[AgentState], dict]:
    """Produce a DRAFT answer (stored as `draft_answer`, not `answer`).

    The graph will then pause at `web_finalize` so a human reviewer can
    approve/edit/reject the draft before it becomes the final `answer` shown
    to the user. Does NOT overwrite `category` — the router's original decision
    (legal_rag vs web_legal_search) is preserved so the UI can distinguish
    "planned web search" vs "fell back from RAG".
    """

    def web_search_node(state: AgentState) -> dict:
        original_query = state.get("raw_query") or state.get("query", "")
        search_query = _build_web_search_query(state)
        logger.info("Web search query: %r", search_query)
        try:
            resp = tavily_tool.search(search_query)
        except Exception as exc:
            logger.exception("Tavily search failed: %s", exc)
            return {
                "draft_answer": (
                    f"{WEB_WARNING_PREFIX}\n"
                    "Xin lỗi, không có thông tin online xác thực (dịch vụ tìm kiếm lỗi)."
                ),
                "sources": [],
                "web_results": [],
                "warning_prefix": WEB_WARNING_PREFIX,
                "refused": False,
                "error": f"tavily_error: {exc}",
            }

        results = resp.get("results", []) if isinstance(resp, dict) else []

        if not results:
            return {
                "draft_answer": (
                    f"{WEB_WARNING_PREFIX}\n"
                    "Xin lỗi, không có thông tin online xác thực về chủ đề này "
                    "trên các nguồn luật Việt Nam."
                ),
                "sources": [],
                "web_results": [],
                "warning_prefix": WEB_WARNING_PREFIX,
                "refused": False,
            }

        web_context = "\n\n".join(
            f"[{i}] {r.get('title','(no title)')}\n{r.get('url','')}\n{r.get('content','')}"
            for i, r in enumerate(results, 1)
        )

        try:
            synth = llm.invoke(
                [
                    SystemMessage(content=WEB_SEARCH_SYSTEM_PROMPT),
                    HumanMessage(
                        content=(
                            f"Web Context:\n{web_context}\n\n"
                            f"Query: {original_query}"
                        )
                    ),
                ]
            )
            draft = synth.content.strip() if hasattr(synth, "content") else str(synth)
        except Exception as exc:
            logger.exception("Web-search synth LLM call failed: %s", exc)
            draft = (
                f"{WEB_WARNING_PREFIX}\n"
                "Xin lỗi, không tổng hợp được kết quả online lúc này."
            )

        sources = [
            {"url": r.get("url", ""), "title": r.get("title", "")}
            for r in results
            if r.get("url")
        ]

        return {
            "draft_answer": draft,
            "sources": sources,
            "web_results": results,
            "warning_prefix": WEB_WARNING_PREFIX,
            "refused": False,
            "requires_approval": True,
        }

    return web_search_node


def web_finalize_node(state: AgentState) -> dict:
    """Commit the reviewed draft as the final answer.

    The graph is compiled with `interrupt_before=["web_finalize"]`, so this
    node only runs AFTER a human reviewer has resumed the thread (either
    approving the draft as-is or supplying an override via /resume).
    Moves `draft_answer` -> `answer` and clears `requires_approval`.
    """
    draft = state.get("draft_answer") or state.get("answer") or ""
    return {
        "answer": draft,
        "requires_approval": False,
    }
