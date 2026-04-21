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
from .state import AgentState

logger = logging.getLogger(__name__)


# --- HITL risk detection -----------------------------------------------------

RISK_PATTERNS: tuple[str, ...] = (
    r"tịch thu",
    r"hình sự",
    r"truy cứu trách nhiệm",
    r"tạm giữ phương tiện",
    r"tước quyền sử dụng",
)


def hitl_gate_node(state: AgentState) -> dict:
    """Passthrough node: flip `requires_approval` when the query contains
    high-risk keywords. Sits between the router and legal_rag so the
    StateGraph can `interrupt_before=["legal_rag"]` conditionally."""
    q = (state.get("query") or "").lower()
    requires = any(re.search(p, q) for p in RISK_PATTERNS)
    if requires:
        logger.info("HITL gate: query flagged for human approval")
    return {"requires_approval": requires}


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


CONTEXTUALIZE_SYSTEM_PROMPT = """Bạn là bộ chuẩn hoá câu hỏi trong cuộc hội thoại về
Luật Giao thông Việt Nam.

Nhiệm vụ: đọc lịch sử hội thoại + câu hỏi mới nhất của người dùng, sau đó viết
lại câu hỏi thành MỘT câu HOÀN CHỈNH, ĐỘC LẬP — người đọc không cần xem lịch
sử cũng hiểu được đầy đủ bối cảnh.

QUY TẮC:
1. Nếu câu mới đã độc lập (không dùng đại từ, tham chiếu, hoặc tiếp nối), giữ nguyên.
2. Nếu câu mới phụ thuộc ngữ cảnh ("cái đó", "vậy", "nếu", "thế còn", "tiếp", các câu ngắn mô tả hành động tiếp theo), hãy gộp ngữ cảnh vào để thành câu đầy đủ.
3. KHÔNG thêm giả định user chưa nói (không tự thêm loại xe/tình huống mới).
4. Đầu ra là 1 câu duy nhất, tiếng Việt, không giải thích, không bullet.

VÍ DỤ 1:
Lịch sử:
  user: "vượt đèn đỏ xe máy bị phạt bao nhiêu?"
  assistant: "Phạt từ 4–6 triệu đồng, trừ 4 điểm GPLX."
  user: "nếu gây tai nạn thì sao"
→ "nếu vượt đèn đỏ bằng xe máy mà gây tai nạn thì bị xử phạt thế nào"

VÍ DỤ 2:
Lịch sử:
  user: "vượt đèn đỏ gây tai nạn thì bị gì?"
  assistant: "..."
  user: "đã lỡ bỏ chạy rồi"
→ "gây tai nạn giao thông rồi bỏ chạy khỏi hiện trường bị xử phạt như thế nào"

VÍ DỤ 3 (đã độc lập):
Lịch sử: (có vài lượt)
  user: "chu kỳ đăng kiểm ô tô dưới 9 chỗ là bao lâu?"
→ "chu kỳ đăng kiểm ô tô dưới 9 chỗ là bao lâu?"
"""


class ContextualizedQuery(BaseModel):
    """Self-contained rewrite of the latest user question."""

    standalone: str = Field(
        description="Câu hỏi mới đã được viết lại thành câu hoàn chỉnh, độc lập, 1 câu duy nhất."
    )


QUERY_EXPANSION_SYSTEM_PROMPT = """Bạn là chuyên gia thuật ngữ pháp lý giao thông Việt Nam.
Nhiệm vụ: viết lại câu hỏi người dùng thành truy vấn dùng thuật ngữ chuẩn
trong Luật Trật tự, An toàn giao thông đường bộ và các Nghị định 168/2024,
100/2019, 46/2016.

QUY TẮC:
1. Giữ nguyên ý định và đối tượng (xe máy/ô tô/người đi bộ) mà user đề cập.
2. Thay từ ngữ dân dã bằng thuật ngữ pháp lý chuẩn.
3. Nếu câu hỏi đã dùng thuật ngữ chuẩn, trả về gần như nguyên văn.
4. KHÔNG thêm thông tin user chưa cung cấp (không tự thêm "xe máy" nếu user chỉ nói "vượt đèn đỏ").
5. Đầu ra là 1 câu truy vấn duy nhất, không giải thích, không bullet.

VÍ DỤ:
- "vượt đèn đỏ xe máy phạt bao nhiêu?"
  → "mức xử phạt hành vi không chấp hành hiệu lệnh đèn tín hiệu giao thông đối với xe mô tô, xe gắn máy"
- "nhậu rồi lái xe bị gì?"
  → "xử phạt điều khiển phương tiện giao thông khi trong máu hoặc hơi thở có nồng độ cồn"
- "không đội mũ bảo hiểm"
  → "xử phạt hành vi không đội mũ bảo hiểm khi tham gia giao thông"
- "bị tịch thu xe khi nào?"
  → "các trường hợp bị tịch thu phương tiện theo quy định giao thông đường bộ"
- "Xử phạt hành vi vượt đèn đỏ đối với xe máy là gì?"
  → "Xử phạt hành vi không chấp hành hiệu lệnh đèn tín hiệu giao thông đối với xe mô tô, xe gắn máy"
"""


class ExpandedQuery(BaseModel):
    """Structured output for query expansion node."""

    expanded: str = Field(
        description=(
            "Truy vấn đã viết lại bằng thuật ngữ pháp lý chuẩn, 1 câu duy nhất, "
            "không giải thích."
        )
    )


# --- Node factories ----------------------------------------------------------


def make_router_node(llm) -> Callable[[AgentState], dict]:
    classify = build_router(llm)

    def router_node(state: AgentState) -> dict:
        category = classify(state["query"])
        logger.info("Router: %s -> %s", state.get("query", "")[:60], category)
        return {"category": category}

    return router_node


def route_by_category(state: AgentState) -> str:
    """Edge condition after router_node."""
    return state.get("category", "out_of_scope")


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



def make_contextualize_node(llm) -> Callable[[AgentState], dict]:
    """Rewrite the latest user question into a self-contained one using prior
    conversation history (if any). Runs BEFORE the router so follow-ups like
    "đã lỡ bỏ chạy rồi" get enriched with context from the previous turn
    ("gây tai nạn") before routing/expansion/retrieval.

    For the first turn (no history) this is a no-op. On LLM failure we keep
    the original query — worst case we regress to the pre-memory baseline.
    """

    structured = llm.with_structured_output(ContextualizedQuery)

    def contextualize_node(state: AgentState) -> dict:
        raw = state.get("query", "")
        history = state.get("chat_history") or []

        if not history:
            return {"raw_query": raw}

        history_text = _format_history_for_prompt(history)
        try:
            result: ContextualizedQuery = structured.invoke(
                [
                    SystemMessage(content=CONTEXTUALIZE_SYSTEM_PROMPT),
                    HumanMessage(
                        content=(
                            f"Lịch sử hội thoại:\n{history_text}\n\n"
                            f"Câu hỏi mới: {raw}"
                        )
                    ),
                ]
            )
            rewritten = (result.standalone or "").strip()
        except Exception as exc:
            logger.warning("Contextualize failed (%s); using raw query", exc)
            return {"raw_query": raw}

        if rewritten and rewritten != raw:
            logger.info("Contextualize: %r -> %r", raw, rewritten)
            return {"query": rewritten, "raw_query": raw}
        return {"raw_query": raw}

    return contextualize_node


def make_query_expansion_node(llm) -> Callable[[AgentState], dict]:
    """Rewrite colloquial Vietnamese queries into formal legal terminology so
    the BM25/dense retriever can hit the corpus chunks, which use statutory
    wording (e.g. "không chấp hành hiệu lệnh đèn tín hiệu" vs. the user's
    "vượt đèn đỏ"). Expansion is used ONLY for retrieval; the generator still
    sees the original query so the answer sounds natural to the user.

    If the LLM call fails, fall back to the original query rather than break
    the flow — retrieval with the original string is still the existing
    pre-expansion baseline.
    """

    structured = llm.with_structured_output(ExpandedQuery)

    def query_expansion_node(state: AgentState) -> dict:
        query = state["query"]
        try:
            result: ExpandedQuery = structured.invoke(
                [
                    SystemMessage(content=QUERY_EXPANSION_SYSTEM_PROMPT),
                    HumanMessage(content=f"Câu hỏi người dùng: {query}"),
                ]
            )
            expanded = (result.expanded or "").strip()
            if not expanded:
                expanded = query
        except Exception as exc:
            logger.warning("Query expansion failed (%s); using original query", exc)
            expanded = query

        if expanded != query:
            logger.info("Query expansion: %r -> %r", query, expanded)
        return {"expanded_query": expanded}

    return query_expansion_node


def make_legal_rag_node(retriever, generator) -> Callable[[AgentState], dict]:
    def legal_rag_node(state: AgentState) -> dict:
        query = state["query"]
        retrieval_query = state.get("expanded_query") or query
        try:
            # Tăng top_k lên 10 để bao quát đủ các loại phương tiện (ô tô, xe máy, xe chuyên dùng)
            chunks = retriever.get_relevant_chunks(retrieval_query, top_k=10)

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


def make_web_search_node(tavily_tool, llm) -> Callable[[AgentState], dict]:
    def web_search_node(state: AgentState) -> dict:
        original_query = state.get("query", "")
        search_query = _build_web_search_query(state)
        logger.info("Web search query: %r", search_query)
        try:
            resp = tavily_tool.search(search_query)
        except Exception as exc:
            logger.exception("Tavily search failed: %s", exc)
            return {
                "answer": (
                    "⚠️ Lưu ý: Thông tin tra cứu từ Internet mở, hãy cẩn trọng.\n"
                    "Xin lỗi, không có thông tin online xác thực (dịch vụ tìm kiếm lỗi)."
                ),
                "sources": [],
                "web_results": [],
                "warning_prefix": (
                    "⚠️ Lưu ý: Thông tin tra cứu từ Internet mở, hãy cẩn trọng."
                ),
                "refused": False,
                "error": f"tavily_error: {exc}",
            }

        results = resp.get("results", []) if isinstance(resp, dict) else []

        if not results:
            return {
                "answer": (
                    "⚠️ Lưu ý: Thông tin tra cứu từ Internet mở, hãy cẩn trọng.\n"
                    "Xin lỗi, không có thông tin online xác thực về chủ đề này "
                    "trên các nguồn luật Việt Nam."
                ),
                "sources": [],
                "web_results": [],
                "warning_prefix": (
                    "⚠️ Lưu ý: Thông tin tra cứu từ Internet mở, hãy cẩn trọng."
                ),
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
            answer = synth.content.strip() if hasattr(synth, "content") else str(synth)
        except Exception as exc:
            logger.exception("Web-search synth LLM call failed: %s", exc)
            answer = (
                "⚠️ Lưu ý: Thông tin tra cứu từ Internet mở, hãy cẩn trọng.\n"
                "Xin lỗi, không tổng hợp được kết quả online lúc này."
            )

        sources = [
            {"url": r.get("url", ""), "title": r.get("title", "")}
            for r in results
            if r.get("url")
        ]

        return {
            "answer": answer,
            "sources": sources,
            "web_results": results,
            "warning_prefix": (
                "⚠️ Lưu ý: Thông tin tra cứu từ Internet mở, hãy cẩn trọng."
            ),
            "refused": False,
            "category": "web_legal_search",
        }

    return web_search_node
