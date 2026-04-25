# -*- coding: utf-8 -*-
"""
Streamlit UI — Trợ lý Luật Giao thông Việt Nam.

Start the FastAPI backend first:
    uvicorn api.main:app --port 8000

Then run the UI:
    streamlit run ui/app.py
"""

from __future__ import annotations

import os
from uuid import uuid4

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = os.getenv("TRAFFIC_RAG_API", "http://localhost:8000")
REQUEST_TIMEOUT = 120.0

st.set_page_config(
    page_title="Trợ lý Luật Giao thông",
    page_icon="⚖️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark-accent legal theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global font and background */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc !important; /* Unified Slate-50 background */
    }

    /* Target the chat input container specifically */
    div[data-testid="stChatInput"] {
        background-color: #f8fafc !important;
        border: 2px solid #cbd5e1 !important; /* Thicker, defined border */
        border-radius: 14px;
    }
    
    /* Input interaction focus */
    div[data-testid="stChatInput"]:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }

    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 12px;
        border: 1px solid #e2e8f0;
        background-color: white;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    /* Sidebar styling — Synchronized with main area */
    section[data-testid="stSidebar"] {
        background-color: #f1f5f9 !important; /* Slightly darker slate for depth */
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] * {
        color: #1e293b !important; /* Dark text for light sidebar */
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: #3b82f6;
        color: white !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #2563eb;
    }

    /* Source expander styling */
    .stExpander {
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
        background: #f8fafc !important;
    }
    .streamlit-expanderHeader {
        font-size: 0.85rem;
        font-weight: 600;
        color: #475569 !important;
    }

    /* Title area */
    .main-title {
        display: flex;
        align-items: center;
        gap: 15px;
        padding: 20px 0 10px 0;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 20px;
    }
    .main-title h1 {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
    }
    .subtitle {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 25px;
    }

    /* Thinking indicator */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    .thinking-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #3b82f6;
        margin: 0 3px;
        animation: pulse 1.4s infinite;
    }
    .thinking-dot:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dot:nth-child(3) { animation-delay: 0.4s; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state():
    st.session_state.setdefault("thread_id", str(uuid4()))
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("pending_thread", None)
    st.session_state.setdefault("pending_category", None)
    st.session_state.setdefault("waiting_for_response", False)


_init_state()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Sidebar — clean & minimal
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 16px 0 8px 0;">
            <div style="font-size: 2.5rem;">⚖️</div>
            <div style="font-size: 1.2rem; font-weight: 700; margin-top: 4px;">
                Trợ lý Luật Giao thông
            </div>
            <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 2px;">
                Hệ thống tra cứu thông minh
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # API status — compact
    ok, _ = _api_health()
    if ok:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:6px;font-size:0.8rem;">'
            '<span style="width:8px;height:8px;border-radius:50%;background:#22c55e;display:inline-block;"></span>'
            'Hệ thống đang hoạt động</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:6px;font-size:0.8rem;">'
            '<span style="width:8px;height:8px;border-radius:50%;background:#ef4444;display:inline-block;"></span>'
            'Hệ thống ngoại tuyến</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    if st.button("🔄  Cuộc hội thoại mới", use_container_width=True):
        st.session_state.thread_id = str(uuid4())
        st.session_state.history = []
        st.session_state.pending_thread = None
        st.session_state.pending_category = None
        st.session_state.waiting_for_response = False
        st.rerun()

    st.divider()

    # Capability hints
    st.markdown(
        """
        <div style="font-size:0.78rem; line-height:1.6; color:#94a3b8;">
            <div style="margin-bottom:8px; font-weight:600; color:#cbd5e1;">Có thể hỏi về</div>
            <div>⚖️ Mức phạt vi phạm giao thông</div>
            <div>📋 Thủ tục đăng ký, đăng kiểm xe</div>
            <div>🪪 Giấy phép lái xe & điểm trừ</div>
            <div>📜 Nghị định, Thông tư mới nhất</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main content area — title
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="main-title">
        <div style="font-size:2rem;">⚖️</div>
        <h1>Trợ lý Pháp lý Giao thông Việt Nam</h1>
    </div>
    <div class="subtitle">
        Tra cứu mức phạt, quy định đăng ký xe, đăng kiểm, giấy phép lái xe theo Nghị định 168/2024
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📎 Nguồn trích dẫn ({len(sources)})", expanded=False):
        for i, s in enumerate(sources, 1):
            if s.get("url"):
                st.markdown(
                    f"**[{i}]** [{s.get('title') or s['url']}]({s['url']})"
                )
            else:
                parts = []
                if s.get("dieu"):
                    parts.append(f"Điều {s['dieu']}")
                if s.get("khoan"):
                    parts.append(f"Khoản {s['khoan']}")
                if s.get("diem"):
                    parts.append(f"Điểm {s['diem']}")
                loc = " · ".join(parts)
                doc = s.get("ten_van_ban") or s.get("doc_id", "")
                st.markdown(f"**[{i}]** {loc} — {doc}")


def _render_message(msg: dict) -> None:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
        return

    with st.chat_message("assistant", avatar="⚖️"):
        # HITL pending state
        if msg.get("pending"):
            st.warning(
                "🔍 **Câu trả lời từ Internet — đang chờ xác nhận.**"
            )
            draft = msg.get("draft_answer") or "_(không có nội dung nháp)_"
            st.markdown("**📝 Bản nháp:**")
            edit_key = f"edit_{msg['thread_id']}_{msg['idx']}"
            edited = st.text_area(
                "Chỉnh sửa bản nháp:",
                value=draft,
                key=edit_key,
                height=180,
                label_visibility="collapsed",
            )
            _render_sources(msg.get("sources", []))

            col1, col2 = st.columns(2)
            tid = msg["thread_id"]
            with col1:
                if st.button("✅ Phê duyệt", key=f"ok_{tid}_{msg['idx']}", use_container_width=True):
                    try:
                        override = None
                        if edited.strip() and edited.strip() != draft.strip():
                            override = {"draft_answer": edited.strip()}
                        result = _api_resume(tid, approved=True, override=override)
                    except Exception as exc:
                        st.error(f"Lỗi: {exc}")
                        return
                    _finalize_pending(msg["idx"], result)
                    st.rerun()
            with col2:
                if st.button("❌ Từ chối", key=f"no_{tid}_{msg['idx']}", use_container_width=True):
                    try:
                        result = _api_resume(tid, approved=False)
                    except Exception as exc:
                        st.error(f"Lỗi: {exc}")
                        return
                    _finalize_pending(msg["idx"], result)
                    st.rerun()
            return

        # Normal answer
        if msg.get("answer"):
            st.markdown(msg["answer"])
        if msg.get("error"):
            st.error(msg["error"])
        _render_sources(msg.get("sources", []))


def _finalize_pending(idx: int, result: dict) -> None:
    st.session_state.history[idx] = {
        "role": "assistant",
        "answer": result.get("answer"),
        "sources": result.get("sources", []),
        "category": result.get("category") or st.session_state.pending_category,
        "model_info": result.get("model_info"),
        "error": result.get("error"),
        "pending": False,
        "thread_id": result.get("thread_id"),
        "idx": idx,
    }
    st.session_state.pending_thread = None
    st.session_state.pending_category = None


# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.history:
    _render_message(msg)


# ---------------------------------------------------------------------------
# Handle user input — show message IMMEDIATELY, then fetch response
# ---------------------------------------------------------------------------

def _handle_user_query(query: str) -> None:
    # 1. Append user message to history IMMEDIATELY
    st.session_state.history.append({"role": "user", "content": query})


def _fetch_response(query: str) -> None:
    # 2. Call API and append assistant response
    try:
        resp = _api_chat(query, st.session_state.thread_id)
    except Exception as exc:
        st.session_state.history.append(
            {
                "role": "assistant",
                "answer": None,
                "error": f"Không thể kết nối đến hệ thống: {exc}",
                "pending": False,
                "idx": len(st.session_state.history),
            }
        )
        st.session_state.waiting_for_response = False
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
                "model_info": resp.get("model_info"),
                "idx": idx,
            }
        )
    else:
        st.session_state.history.append(
            {
                "role": "assistant",
                "answer": resp.get("answer"),
                "sources": resp.get("sources", []),
                "category": resp.get("category"),
                "model_info": resp.get("model_info"),
                "error": resp.get("error"),
                "pending": False,
                "thread_id": resp.get("thread_id"),
                "idx": idx,
            }
        )
    st.session_state.waiting_for_response = False


# --- Input handling ---------------------------------------------------------

user_input = st.chat_input("Nhập câu hỏi về luật giao thông...")

if user_input:
    # Show user message immediately
    _handle_user_query(user_input)
    st.session_state.waiting_for_response = True
    st.session_state._pending_query = user_input
    st.rerun()

# If waiting for response, show thinking indicator and fetch
if st.session_state.get("waiting_for_response") and st.session_state.get("_pending_query"):
    # Render the user message that was just added
    with st.chat_message("assistant", avatar="⚖️"):
        with st.spinner("🔍 Đang tra cứu văn bản pháp luật..."):
            query = st.session_state.pop("_pending_query")
            _fetch_response(query)
    st.rerun()
