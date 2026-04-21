# -*- coding: utf-8 -*-
"""
graph.py — Assemble the LangGraph StateGraph.

Flow:
    START -> contextualize -> router ─┬─> hitl_gate -> query_expansion -> legal_rag ─┬─> END
                                      │                                               └─> web_search -> END
                                      ├─> chit_chat    -> END
                                      ├─> web_search   -> END
                                      └─> out_of_scope -> END

`interrupt_before=["legal_rag"]` pauses EVERY path entering legal_rag.
The API layer decides whether to auto-resume (when requires_approval=False)
or wait for a human (when requires_approval=True). query_expansion runs
BEFORE the interrupt so the reviewer sees both the original and expanded
query in the state snapshot.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from .nodes import (
    hitl_gate_node,
    legal_fallback_router,
    make_chit_chat_node,
    make_contextualize_node,
    make_legal_rag_node,
    make_query_expansion_node,
    make_router_node,
    make_web_search_node,
    out_of_scope_node,
    route_by_category,
)
from .state import AgentState

logger = logging.getLogger(__name__)


def _default_sync_checkpointer(checkpoint_db: str | Path):
    db_path = Path(checkpoint_db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    logger.info("Checkpointer SqliteSaver (sync) at %s", db_path)
    return SqliteSaver(conn)


def build_graph(
    retriever,
    generator,
    llm,
    tavily_tool,
    checkpoint_db: str | Path = "traffic_rag/checkpoints/graph.db",
    checkpointer=None,
    enable_hitl_interrupt: bool = True,
):
    """Compile and return the agentic RAG graph.

    Args:
        retriever: TrafficHybridRetriever instance.
        generator: LegalAnswerGenerator instance.
        llm: LangChain chat model (used by router, chit_chat, web_search synth).
        tavily_tool: TavilySearchTool instance.
        checkpoint_db: Path to SQLite file (used only when `checkpointer` is None).
        checkpointer: Optional pre-built checkpointer. Use `AsyncSqliteSaver`
            for async callers (FastAPI); if None, defaults to a sync `SqliteSaver`
            (good for tests/notebooks).
        enable_hitl_interrupt: If False, skip interrupt_before (useful for tests).

    Returns:
        Compiled LangGraph (supports .invoke / .ainvoke / .get_state / .aget_state).
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("contextualize", make_contextualize_node(llm))
    workflow.add_node("router", make_router_node(llm))
    workflow.add_node("hitl_gate", hitl_gate_node)
    workflow.add_node("query_expansion", make_query_expansion_node(llm))
    workflow.add_node("legal_rag", make_legal_rag_node(retriever, generator))
    workflow.add_node("chit_chat", make_chit_chat_node(llm))
    workflow.add_node("out_of_scope", out_of_scope_node)
    workflow.add_node("web_search", make_web_search_node(tavily_tool, llm))

    workflow.add_edge(START, "contextualize")
    workflow.add_edge("contextualize", "router")

    workflow.add_conditional_edges(
        "router",
        route_by_category,
        {
            "legal_rag": "hitl_gate",
            "chit_chat": "chit_chat",
            "web_legal_search": "web_search",
            "out_of_scope": "out_of_scope",
        },
    )
    workflow.add_edge("hitl_gate", "query_expansion")
    workflow.add_edge("query_expansion", "legal_rag")
    workflow.add_conditional_edges(
        "legal_rag",
        legal_fallback_router,
        {"web_search": "web_search", "end": END},
    )
    workflow.add_edge("chit_chat", END)
    workflow.add_edge("web_search", END)
    workflow.add_edge("out_of_scope", END)

    if checkpointer is None:
        checkpointer = _default_sync_checkpointer(checkpoint_db)

    compile_kwargs = {"checkpointer": checkpointer}
    if enable_hitl_interrupt:
        compile_kwargs["interrupt_before"] = ["legal_rag"]

    return workflow.compile(**compile_kwargs)
