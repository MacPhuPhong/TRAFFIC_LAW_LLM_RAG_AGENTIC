# -*- coding: utf-8 -*-
"""
api/main.py — FastAPI entry point for the Agentic Traffic-Law RAG.

Endpoints:
    POST /chat                 Kick off a new (or resume existing) conversation.
    POST /resume/{thread_id}   Human approval/rejection of a paused HITL step.
    GET  /pending/{thread_id}  Inspect the current snapshot of a thread.
    GET  /health               Liveness probe.

Run:
    uvicorn traffic_rag.api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException

# --- sys.path injection so `source.*` is importable regardless of CWD -------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent  # traffic_rag/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.agent import TavilySearchTool, build_graph  # noqa: E402
from source.rag_core import LegalAnswerGenerator, TrafficHybridRetriever  # noqa: E402

from api.schemas import ChatRequest, ChatResponse, PendingResponse, ResumeRequest  # noqa: E402

# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _load_dotenv_if_present() -> None:
    """Load .env from the repo root if python-dotenv is available.

    Also maps the existing project convention `API_KEY` -> `GOOGLE_API_KEY`
    so users who already have a .env from the notebook flow don't have to
    duplicate their Gemini key.
    """
    try:
        from dotenv import load_dotenv

        for candidate in (
            PROJECT_ROOT.parent / ".env",  # repo root (user's existing .env)
            PROJECT_ROOT / ".env",
        ):
            if candidate.exists():
                load_dotenv(candidate, override=False)
                logger.info("Loaded env from %s", candidate)
    except ImportError:
        logger.debug("python-dotenv not installed; relying on process env")

    if not os.getenv("GOOGLE_API_KEY") and os.getenv("API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["API_KEY"]


def _validate_env() -> None:
    missing = [k for k in ("GOOGLE_API_KEY", "TAVILY_API_KEY") if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            f"Missing required env vars: {missing}. "
            f"Set them in .env or the process environment before starting the API."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_dotenv_if_present()
    _validate_env()

    logger.info("Initialising retriever...")
    retriever = TrafficHybridRetriever()

    model_name = os.getenv("GENERATOR_MODEL", "gemini-3.1-flash-lite-preview")
    logger.info("Initialising generator (%s)...", model_name)
    generator = LegalAnswerGenerator(provider="google", model=model_name)

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.1,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        request_timeout=30,
    )

    tavily = TavilySearchTool()

    ckpt_db = Path(
        os.getenv("CHECKPOINT_DB", str(PROJECT_ROOT / "checkpoints" / "graph.db"))
    )
    ckpt_db.parent.mkdir(parents=True, exist_ok=True)

    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    conn = await aiosqlite.connect(str(ckpt_db))
    checkpointer = AsyncSqliteSaver(conn)
    await checkpointer.setup()
    logger.info("AsyncSqliteSaver at %s", ckpt_db)

    logger.info("Building graph...")
    graph = build_graph(
        retriever, generator, llm, tavily,
        checkpointer=checkpointer,
    )

    app.state.retriever = retriever
    app.state.generator = generator
    app.state.llm = llm
    app.state.tavily = tavily
    app.state.graph = graph
    app.state._ckpt_conn = conn
    logger.info("API ready.")

    yield

    logger.info("Shutting down API.")
    try:
        await conn.close()
    except Exception:
        pass


app = FastAPI(
    title="Traffic Law Agentic RAG",
    description=(
        "LangGraph-powered Vietnamese traffic-law assistant with HITL approval "
        "and Tavily web-search fallback."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# --- endpoints --------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


MAX_HISTORY_MESSAGES = 20  # ~10 turns


async def _append_chat_history(graph, config, user_text: str, assistant_text: str, sources: list[dict] = None) -> None:
    """Persist a completed turn into the graph's checkpointed chat_history.
    Includes sources so the contextualizer can resolve 'Nguồn số X' references.
    """
    snap = await graph.aget_state(config)
    history = list(snap.values.get("chat_history") or []) if snap else []
    if user_text:
        history.append({"role": "user", "content": user_text})
    if assistant_text:
        # We store the sources alongside the content in the history dict
        history.append({
            "role": "assistant", 
            "content": assistant_text,
            "sources": sources or []
        })
    history = history[-MAX_HISTORY_MESSAGES:]
    await graph.aupdate_state(config, {"chat_history": history})



def _snapshot_to_response(thread_id: str, snap) -> ChatResponse:
    values = snap.values if snap else {}
    return ChatResponse(
        thread_id=thread_id,
        status="completed",
        answer=values.get("answer"),
        sources=values.get("sources", []),
        category=values.get("category"),
        requires_approval=False,
        model_info=values.get("model_info"),
        expanded_query=values.get("expanded_query"),
        error=values.get("error"),
    )


def _is_paused_on_web_review(snap) -> bool:
    """Graph pauses at `web_finalize` whenever a web answer needs approval."""
    return bool(snap and snap.next and "web_finalize" in snap.next)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid4())
    logger.info("CHAT [%s]: %s", thread_id, req.query[:50])
    config = {"configurable": {"thread_id": thread_id}}
    graph = app.state.graph

    await graph.ainvoke(
        {"query": req.query, "raw_query": req.query},
        config,
    )
    snap = await graph.aget_state(config)

    if _is_paused_on_web_review(snap):
        logger.info("CHAT [%s]: paused at web_finalize — awaiting human review", thread_id)
        values = snap.values or {}
        return ChatResponse(
            thread_id=thread_id,
            status="pending_web_review",
            draft_answer=values.get("draft_answer"),
            sources=values.get("sources", []),
            category=values.get("category"),
            requires_approval=True,
            model_info=values.get("model_info"),
            expanded_query=values.get("expanded_query"),
            error=values.get("error"),
        )

    resp = _snapshot_to_response(thread_id, snap)
    await _append_chat_history(graph, config, req.query, resp.answer or "", resp.sources)
    return resp


@app.post("/resume/{thread_id}", response_model=ChatResponse)
async def resume(thread_id: str, req: ResumeRequest):
    logger.info("RESUME [%s]: approved=%s", thread_id, req.approved)
    config = {"configurable": {"thread_id": thread_id}}
    graph = app.state.graph

    snap = await graph.aget_state(config)
    if not snap or not snap.next:
        logger.warning("RESUME [%s]: No pending state found", thread_id)
        raise HTTPException(
            status_code=404,
            detail=f"Thread {thread_id} has no pending interrupt to resume.",
        )

    user_text = snap.values.get("raw_query") or snap.values.get("query") or ""

    if not req.approved:
        rejection_msg = (
            "Câu trả lời từ Internet đã bị người phê duyệt từ chối. "
            "Bạn nên tham khảo trực tiếp văn bản pháp luật hoặc luật sư để được "
            "tư vấn chính xác."
        )
        await graph.aupdate_state(
            config,
            {
                "answer": rejection_msg,
                "draft_answer": "",
                "sources": [],
                "refused": True,
                "requires_approval": False,
            },
            as_node="web_finalize",
        )
        await _append_chat_history(graph, config, user_text, rejection_msg, [])
        snap = await graph.aget_state(config)
        return ChatResponse(
            thread_id=thread_id,
            status="rejected",
            answer=rejection_msg,
            sources=[],
            category=snap.values.get("category") if snap.values else None,
            requires_approval=False,
        )

    # Approved — optionally patch draft_answer via override, then let
    # web_finalize commit draft_answer -> answer.
    if req.override:
        await graph.aupdate_state(config, req.override)

    await graph.ainvoke(None, config)
    snap = await graph.aget_state(config)
    resp = _snapshot_to_response(thread_id, snap)
    await _append_chat_history(graph, config, user_text, resp.answer or "", resp.sources)
    return resp



@app.get("/pending/{thread_id}", response_model=PendingResponse)
async def pending(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    graph = app.state.graph
    snap = await graph.aget_state(config)
    if not snap:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found.")

    return PendingResponse(
        thread_id=thread_id,
        next_nodes=list(snap.next) if snap.next else [],
        values=dict(snap.values) if snap.values else {},
        requires_approval=bool(snap.values.get("requires_approval")) if snap.values else False,
    )
