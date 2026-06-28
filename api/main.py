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

    # Always-on LangSmith tracing when LANGCHAIN_API_KEY is present.
    # No-op if the key is missing (logs a warning, no exception).
    try:
        from research.utils.langsmith_setup import enable_tracing
        traced = enable_tracing(
            project=os.getenv("LANGCHAIN_PROJECT", "traffic-rag-prod"),
            run_name=None,
        )
        logger.info("LangSmith tracing: %s", "ON" if traced else "OFF (no key)")
    except Exception as exc:
        logger.warning("LangSmith setup failed (non-fatal): %s", exc)

    # Lightweight runtime metrics — exposed via GET /metrics.
    import time as _time
    app.state.metrics = {
        "started_at":          _time.time(),
        "total_chat_requests": 0,
        "total_resume_requests": 0,
        "total_chat_errors":   0,
        "total_chat_latency_ms": 0.0,
        "last_request_at":     None,
        "tracing_enabled":     bool(os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"),
        "reranker_enabled":    False,  # populated below
    }

    logger.info("Initialising retriever...")
    enable_reranker = os.getenv("ENABLE_RERANKER", "false").lower() in ("1", "true", "yes")
    reranker_model = os.getenv("RERANKER_MODEL") or None
    rerank_top_n = int(os.getenv("RERANK_TOP_N", "30"))
    # Strip whitespace — HF Spaces UI sometimes appends trailing newline when
    # pasting secrets, which causes httpx "Illegal header value" errors.
    vector_db = (os.getenv("VECTOR_DB") or "qdrant").strip().lower()
    if vector_db == "pgvector":
        from source.rag_core.retriever_pgvector import PgVectorHybridRetriever
        pg_url = (os.getenv("VECTOR_DB_URL") or "").strip()
        if not pg_url:
            raise RuntimeError("VECTOR_DB=pgvector requires VECTOR_DB_URL env var.")
        if pg_url.startswith("postgres://"):
            pg_url = pg_url.replace("postgres://", "postgresql://", 1)
        retriever = PgVectorHybridRetriever(
            pg_url=pg_url,
            enable_reranker=enable_reranker,
            reranker_model=reranker_model,
            rerank_top_n=rerank_top_n,
        )
    else:
        _qdrant_url = (os.getenv("QDRANT_URL") or "").strip() or None
        _qdrant_api_key = (os.getenv("QDRANT_API_KEY") or "").strip() or None
        retriever = TrafficHybridRetriever(
            qdrant_host=os.getenv("QDRANT_HOST", "localhost").strip(),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            qdrant_url=_qdrant_url,
            qdrant_api_key=_qdrant_api_key,
            enable_reranker=enable_reranker,
            reranker_model=reranker_model,
            rerank_top_n=rerank_top_n,
        )
    if enable_reranker:
        logger.info("Reranker enabled (model=%s, top_n=%d)",
                    reranker_model or "BAAI/bge-reranker-v2-m3", rerank_top_n)
    app.state.metrics["reranker_enabled"] = bool(enable_reranker)

    model_name = os.getenv("GENERATOR_MODEL", "gemini-3.1-flash-lite-preview")
    logger.info("Initialising generator (%s)...", model_name)
    generator = LegalAnswerGenerator(provider="google", model=model_name)

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        request_timeout=30,
    )

    tavily = TavilySearchTool()

    # Checkpoint backend: Postgres (Supabase) if CHECKPOINT_DB_URL starts with
    # postgres:// or postgresql://; otherwise local SQLite file.
    ckpt_url = os.getenv("CHECKPOINT_DB_URL", "")
    if ckpt_url.startswith(("postgres://", "postgresql://")):
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        # langgraph expects "postgresql://" not "postgres://"
        pg_url = ckpt_url.replace("postgres://", "postgresql://", 1)
        cm = AsyncPostgresSaver.from_conn_string(pg_url)
        checkpointer = await cm.__aenter__()
        app.state._ckpt_cm = cm  # keep alive for shutdown
        await checkpointer.setup()
        logger.info("AsyncPostgresSaver connected (Supabase / Postgres)")
    else:
        ckpt_db = Path(
            os.getenv("CHECKPOINT_DB", str(PROJECT_ROOT / "checkpoints" / "graph.db"))
        )
        ckpt_db.parent.mkdir(parents=True, exist_ok=True)
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        conn = await aiosqlite.connect(str(ckpt_db))
        checkpointer = AsyncSqliteSaver(conn)
        await checkpointer.setup()
        app.state._ckpt_conn = conn
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
    # In-memory registry of threads paused at web_finalize awaiting admin
    # review. Populated by /chat, drained by /resume. Each entry holds the
    # info an admin needs to make a decision.
    app.state.pending_threads = {}
    logger.info("API ready.")

    yield

    logger.info("Shutting down API.")
    conn = getattr(app.state, "_ckpt_conn", None)
    if conn is not None:
        try:
            await conn.close()
        except Exception:
            pass
    cm = getattr(app.state, "_ckpt_cm", None)
    if cm is not None:
        try:
            await cm.__aexit__(None, None, None)
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

# CORS — when frontend is hosted on a different origin (e.g. Vercel) and calls
# the backend directly. Set CORS_ALLOWED_ORIGINS="https://x.vercel.app,https://y"
# in env. Use "*" to allow all (dev only).
_cors = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
if _cors:
    from fastapi.middleware.cors import CORSMiddleware
    origins = ["*"] if _cors == "*" else [o.strip() for o in _cors.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# --- endpoints --------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    """Lightweight runtime metrics for monitoring — JSON, not Prometheus.

    Counters are updated by /chat and /resume handlers. Latency is the
    rolling sum (divide by total_chat_requests for the mean).
    """
    import time as _time
    m = app.state.metrics
    started = m.get("started_at") or _time.time()
    n = max(int(m.get("total_chat_requests") or 0), 0)
    total_lat = float(m.get("total_chat_latency_ms") or 0.0)
    return {
        "uptime_seconds":        round(_time.time() - started, 1),
        "total_chat_requests":   n,
        "total_resume_requests": int(m.get("total_resume_requests") or 0),
        "total_chat_errors":     int(m.get("total_chat_errors") or 0),
        "chat_avg_latency_ms":   round(total_lat / n, 1) if n else None,
        "last_request_at":       m.get("last_request_at"),
        "tracing_enabled":       bool(m.get("tracing_enabled")),
        "reranker_enabled":      bool(m.get("reranker_enabled")),
    }


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

    import time as _time
    _t0 = _time.time()
    app.state.metrics["last_request_at"] = _t0
    app.state.metrics["total_chat_requests"] = int(app.state.metrics.get("total_chat_requests") or 0) + 1
    try:
        await graph.ainvoke(
            {"query": req.query, "raw_query": req.query},
            config,
        )
    except Exception:
        app.state.metrics["total_chat_errors"] = int(app.state.metrics.get("total_chat_errors") or 0) + 1
        raise
    finally:
        app.state.metrics["total_chat_latency_ms"] = float(app.state.metrics.get("total_chat_latency_ms") or 0.0) + (_time.time() - _t0) * 1000.0
    snap = await graph.aget_state(config)

    if _is_paused_on_web_review(snap):
        logger.info("CHAT [%s]: paused at web_finalize — awaiting human review", thread_id)
        values = snap.values or {}
        import time as _time
        app.state.pending_threads[thread_id] = {
            "thread_id": thread_id,
            "query": req.query,
            "draft_answer": values.get("draft_answer"),
            "sources": values.get("sources", []),
            "category": values.get("category"),
            "model_info": values.get("model_info"),
            "expanded_query": values.get("expanded_query"),
            "created_at": _time.time(),
        }
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
    app.state.metrics["total_resume_requests"] = int(app.state.metrics.get("total_resume_requests") or 0) + 1
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
        app.state.pending_threads.pop(thread_id, None)
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
    app.state.pending_threads.pop(thread_id, None)
    return resp



@app.get("/pending")
async def pending_list():
    """List all threads currently paused at web_finalize awaiting admin review.

    Sorted by creation time descending (newest first).
    """
    items = sorted(
        app.state.pending_threads.values(),
        key=lambda x: x.get("created_at", 0.0),
        reverse=True,
    )
    return {"count": len(items), "items": items}


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
