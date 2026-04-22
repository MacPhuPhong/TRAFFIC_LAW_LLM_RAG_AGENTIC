# -*- coding: utf-8 -*-
"""
Streamlit UI for the Agentic Traffic-Law RAG.

Start the FastAPI backend first:
    uvicorn traffic_rag.api.main:app --port 8000

Then run the UI:
    streamlit run traffic_rag/ui/app.py
"""

from __future__ import annotations

import os
from uuid import uuid4

import httpx
import streamlit as st

# ---------------------------------------------------------------------------

API_BASE = os.getenv("TRAFFIC_RAG_API", "http://localhost:8000")
REQUEST_TIMEOUT = 120.0

CATEGORY_LABELS = {
    "legal_rag": ("📘", "Luật VN"),
    "chit_chat": ("💬", "Trò chuyện"),
    "web_legal_search": ("🌐", "Web search"),
    "out_of_scope": ("🚫", "Ngoài phạm vi"),
}

st.set_page_config(
    page_title="Trợ lý Luật Giao thông VN",
    page_icon="🚗",
    layout="wide",
)

# --- Modern UI Styles (Glass Legal Pro v5.0) ---------------------------------
st.markdown(
    """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Outfit:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Main App Background */
    .stApp {
        background-color: #0e1117;
    }

    /* Glass Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(17, 25, 40, 0.75) !important;
        backdrop-filter: blur(12px) saturate(180%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Chat Input Styling */
    .stChatInputContainer {
        padding-bottom: 20px;
    }

    /* Custom Message Container */
    .message-container {
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .user-msg {
        background: rgba(255, 255, 255, 0.03);
    }

    .assistant-msg {
        background: rgba(94, 106, 210, 0.05);
        border-left: 4px solid #5e6ad2;
    }

    /* Sources Glass Cards */
    .source-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 8px 12px;
        margin-top: 6px;
        transition: all 0.2s ease;
    }
    .source-card:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(94, 106, 210, 0.4);
    }

    /* Status Pills */
    .status-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
        background: rgba(94, 106, 210, 0.2);
        color: #8c9eff;
        border: 1px solid rgba(94, 106, 210, 0.3);
    }
    
    .stWarning {
        background: rgba(255, 152, 0, 0.1) !important;
        border: 1px solid rgba(255, 152, 0, 0.2) !important;
        color: #ff9800 !important;
        border-radius: 10px;
    }
    
    .stSuccess {
        background: rgba(76, 175, 80, 0.1) !important;
        border: 1px solid rgba(76, 175, 80, 0.2) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- session-state bootstrap ------------------------------------------------


def _init_state():
    st.session_state.setdefault("thread_id", str(uuid4()))
    st.session_state.setdefault("history", [])  # list[dict]
    st.session_state.setdefault("pending_thread", None)
    st.session_state.setdefault("pending_category", None)


_init_state()


# --- HTTP helpers -----------------------------------------------------------


def _api_health() -> tuple[bool, str]:
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=5.0)
        return r.status_code == 200, r.text
    except Exception as exc:
        return False, str(exc)


def _api_chat(query: str, thread_id: str | None) -> dict:
    r = httpx.post(
        f"{API_BASE}/chat",
        json={"query": query, "thread_id": thread_id},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def _api_resume(thread_id: str, approved: bool, override: dict | None = None) -> dict:
    payload = {"approved": approved}
    if override:
        payload["override"] = override
    r = httpx.post(
        f"{API_BASE}/resume/{thread_id}",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def _api_pending(thread_id: str) -> dict:
    r = httpx.get(f"{API_BASE}/pending/{thread_id}", timeout=30.0)
    r.raise_for_status()
    return r.json()


# --- sidebar ---------------------------------------------------------------

with st.sidebar:
    st.title("🚗 Trợ lý Luật GT")
    st.caption("LangGraph + HITL + Tavily fallback")

    ok, detail = _api_health()
    if ok:
        st.success(f"API: online — `{API_BASE}`")
    else:
        st.error(f"API: offline — `{API_BASE}`\n{detail}")

    st.divider()

    st.markdown("**Thread hiện tại**")
    st.code(st.session_state.thread_id, language=None)

    if st.button("🔄 Bắt đầu cuộc hội thoại mới", use_container_width=True):
        st.session_state.thread_id = str(uuid4())
        st.session_state.history = []
        st.session_state.pending_thread = None
        st.session_state.pending_category = None
        st.rerun()

    with st.expander("ℹ️ Các route của Router", expanded=False):
        st.markdown(
            "- **📘 Luật VN** → truy xuất corpus + generator\n"
            "- **💬 Trò chuyện** → LLM trả lời xã giao\n"
            "- **🌐 Web search** → Tavily (quốc tế / tin mới)\n"
            "- **🚫 Ngoài phạm vi** → từ chối\n"
            "\n_Câu hỏi chứa từ khoá rủi ro cao (tịch thu, hình sự, tước GPLX...) "
            "sẽ bị chặn chờ phê duyệt (HITL)._"
        )

    with st.expander("🧪 Ví dụ câu hỏi", expanded=False):
        examples = [
            ("📘", "Vượt đèn đỏ xe máy phạt bao nhiêu?"),
            ("📘", "Chu kỳ đăng kiểm ô tô dưới 9 chỗ?"),
            ("⚠️", "Khi nào bị tịch thu phương tiện?"),
            ("💬", "Xin chào, bạn là ai?"),
            ("🌐", "Luật giao thông ở Nhật Bản thế nào?"),
            ("🚫", "Cách nấu phở bò ngon?"),
        ]
        for icon, q in examples:
            if st.button(f"{icon} {q}", key=f"ex_{q}", use_container_width=True):
                st.session_state._pending_input = q
                st.rerun()


# --- main column ------------------------------------------------------------

st.markdown(
    """
    <h1 style='text-align: left; background: -webkit-linear-gradient(#fff, #999); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        🚗 Trợ lý Pháp lý Giao thông VN
    </h1>
    """,
    unsafe_allow_html=True
)
st.caption("Agentic RAG · Cinematic Dark · HITL v4.1 · Glass UI v5.0")


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 Nguồn trích dẫn ({len(sources)})", expanded=False):
        for i, s in enumerate(sources, 1):
            st.markdown('<div class="source-card">', unsafe_allow_html=True)
            if s.get("url"):
                st.markdown(
                    f"**[{i}]** [{s.get('title') or s['url']}]({s['url']})"
                )
            else:
                parts = []
                if s.get("dieu"): parts.append(f"Điều {s['dieu']}")
                if s.get("khoan"): parts.append(f"Khoản {s['khoan']}")
                if s.get("diem"): parts.append(f"Điểm {s['diem']}")
                loc = " · ".join(parts)
                doc = s.get("ten_van_ban") or s.get("doc_id", "")
                did = s.get("doc_id", "")
                st.markdown(f"**[{i}]** {loc} — {doc} <br><small style='opacity:0.6'>{did}</small>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


def _render_message(msg: dict) -> None:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f'<div class="message-container user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        return

    with st.chat_message("assistant"):
        cat = msg.get("category")
        if cat and cat in CATEGORY_LABELS:
            icon, label = CATEGORY_LABELS[cat]
            st.markdown(
                f'<span class="status-pill">{icon} {label}</span>'
                f'<small style="opacity:0.5">model: {msg.get("model_info") or "gemini-flash"}</small>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="message-container assistant-msg">', unsafe_allow_html=True)
        
        if msg.get("pending"):
            st.warning(
                "🕵️ **Câu trả lời được tổng hợp từ Internet — đang chờ chuyên gia pháp lý duyệt.**\n\n"
                "Bản nháp bên dưới CHƯA gửi cho người dùng. "
                "Duyệt để phát hành, Từ chối để thay bằng lời khuyên mặc định, "
                "hoặc Chỉnh sửa để biên tập trước khi phát hành."
            )
            draft = msg.get("draft_answer") or "_(không có nội dung nháp)_"
            st.markdown("**📝 Bản nháp (chưa phát hành):**")
            edit_key = f"edit_{msg['thread_id']}_{msg['idx']}"
            edited = st.text_area(
                "Biên tập bản nháp trước khi phát hành (tùy chọn):",
                value=draft,
                key=edit_key,
                height=180,
                label_visibility="collapsed",
            )
            _render_sources(msg.get("sources", []))

            col1, col2, col3 = st.columns(3)
            tid = msg["thread_id"]
            with col1:
                if st.button("✅ Duyệt", key=f"ok_{tid}_{msg['idx']}", use_container_width=True):
                    try:
                        override = None
                        if edited.strip() and edited.strip() != draft.strip():
                            override = {"draft_answer": edited.strip()}
                        result = _api_resume(tid, approved=True, override=override)
                    except Exception as exc:
                        st.error(f"Resume lỗi: {exc}")
                        return
                    _finalize_pending(msg["idx"], result)
                    st.rerun()
            with col2:
                if st.button("❌ Từ chối", key=f"no_{tid}_{msg['idx']}", use_container_width=True):
                    try:
                        result = _api_resume(tid, approved=False)
                    except Exception as exc:
                        st.error(f"Resume lỗi: {exc}")
                        return
                    _finalize_pending(msg["idx"], result)
                    st.rerun()
            with col3:
                st.caption("💡 Chỉnh sửa trong ô trên rồi bấm Duyệt để phát hành bản đã sửa.")
            st.markdown('</div>', unsafe_allow_html=True)
            return

        if msg.get("risk_flag"):
            st.warning(
                "🛑 Câu hỏi liên quan chế tài nặng (tịch thu / hình sự / tước GPLX). "
                "Câu trả lời dưới đây trích nguyên văn văn bản pháp luật; "
                "chỉ mang tính tham khảo, không thay thế tư vấn của luật sư."
            )
        if msg.get("answer"):
            st.markdown(msg["answer"])
        if msg.get("error"):
            st.error(msg["error"])
        
        st.markdown('</div>', unsafe_allow_html=True)
        _render_sources(msg.get("sources", []))


def _finalize_pending(idx: int, result: dict) -> None:
    """Replace the pending message at index `idx` with the resumed result."""
    st.session_state.history[idx] = {
        "role": "assistant",
        "answer": result.get("answer"),
        "sources": result.get("sources", []),
        "category": result.get("category") or st.session_state.pending_category,
        "risk_flag": bool(result.get("risk_flag")),
        "model_info": result.get("model_info"),
        "error": result.get("error"),
        "pending": False,
        "thread_id": result.get("thread_id"),
        "idx": idx,
    }
    st.session_state.pending_thread = None
    st.session_state.pending_category = None


# Render existing history
for msg in st.session_state.history:
    _render_message(msg)


def _handle_user_query(query: str) -> None:
    st.session_state.history.append({"role": "user", "content": query})

    try:
        resp = _api_chat(query, st.session_state.thread_id)
    except Exception as exc:
        st.session_state.history.append(
            {
                "role": "assistant",
                "answer": None,
                "error": f"Lỗi gọi API: {exc}",
                "pending": False,
                "idx": len(st.session_state.history),
            }
        )
        return

    st.session_state.thread_id = resp.get("thread_id", st.session_state.thread_id)
    idx = len(st.session_state.history)

    if resp.get("status") == "pending_web_review":
        st.session_state.pending_thread = resp["thread_id"]
        st.session_state.pending_category = resp.get("category")
        st.session_state.history.append(
            {
                "role": "assistant",
                "pending": True,
                "draft_answer": resp.get("draft_answer"),
                "sources": resp.get("sources", []),
                "thread_id": resp["thread_id"],
                "category": resp.get("category"),
                "risk_flag": bool(resp.get("risk_flag")),
                "model_info": resp.get("model_info"),
                "idx": idx,
            }
        )
        return

    st.session_state.history.append(
        {
            "role": "assistant",
            "answer": resp.get("answer"),
            "sources": resp.get("sources", []),
            "category": resp.get("category"),
            "risk_flag": bool(resp.get("risk_flag")),
            "model_info": resp.get("model_info"),
            "error": resp.get("error"),
            "pending": False,
            "thread_id": resp.get("thread_id"),
            "idx": idx,
        }
    )


# --- input handling ---------------------------------------------------------

_pending_input = st.session_state.pop("_pending_input", None)
user_input = st.chat_input("Hỏi về luật giao thông, đăng kiểm, mức phạt...")
if _pending_input and not user_input:
    user_input = _pending_input

if user_input:
    _handle_user_query(user_input)
    st.rerun()
