# -*- coding: utf-8 -*-
"""
graph.py — Assemble the LangGraph StateGraph.

Flow:
    START -> analyzer ─┬─▶ legal_rag ─┬─▶ END            (trusted corpus)
                       │               └─▶ web_search -> web_finalize -> END
                       ├─▶ chit_chat -> END
                       ├─▶ web_search -> web_finalize -> END
                       └─▶ out_of_scope -> END

HITL design:
    - Local RAG answers cite the official corpus (NĐ 168/2024, Luật ATGT) and
      are trusted by default — they flow straight to the user.
    - `interrupt_before=["web_finalize"]` is the ACTUAL gate: any answer
      synthesised from open-internet content pauses for human review. The
      reviewer sees the `draft_answer` + source URLs and approves / rejects
      before the user ever receives it.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from .nodes import (
    legal_fallback_router,
    make_analyzer_node,
    make_chit_chat_node,
    make_legal_rag_node,
    make_web_search_node,
    out_of_scope_node,
    route_by_category,
    web_finalize_node,
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
        llm: LangChain chat model (used by analyzer, chit_chat, web_search synth).
        tavily_tool: TavilySearchTool instance.
        checkpoint_db: Path to SQLite file (used only when `checkpointer` is None).
        checkpointer: Optional pre-built checkpointer. Use `AsyncSqliteSaver`
            for async callers (FastAPI); if None, defaults to a sync `SqliteSaver`
            (good for tests/notebooks).
        enable_hitl_interrupt: If False, skip `interrupt_before` on `web_finalize`
            (useful for tests that want end-to-end runs without manual approval).

    Returns:
        Compiled LangGraph (supports .invoke / .ainvoke / .get_state / .aget_state).
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("analyzer", make_analyzer_node(llm))
    workflow.add_node("legal_rag", make_legal_rag_node(retriever, generator, llm=llm))
    workflow.add_node("chit_chat", make_chit_chat_node(llm))
    workflow.add_node("out_of_scope", out_of_scope_node)
    workflow.add_node("web_search", make_web_search_node(tavily_tool, llm))
    workflow.add_node("web_finalize", web_finalize_node)

    workflow.add_edge(START, "analyzer")

    workflow.add_conditional_edges(
        "analyzer",
        route_by_category,
        {
            "legal_rag": "legal_rag",
            "chit_chat": "chit_chat",
            "web_legal_search": "web_search",
            "out_of_scope": "out_of_scope",
        },
    )
    workflow.add_conditional_edges(
        "legal_rag",
        legal_fallback_router,
        {"web_search": "web_search", "end": END},
    )
    workflow.add_edge("web_search", "web_finalize")
    workflow.add_edge("web_finalize", END)
    workflow.add_edge("chit_chat", END)
    workflow.add_edge("out_of_scope", END)

    if checkpointer is None:
        checkpointer = _default_sync_checkpointer(checkpoint_db)

    compile_kwargs = {"checkpointer": checkpointer}
    if enable_hitl_interrupt:
        compile_kwargs["interrupt_before"] = ["web_finalize"]

    return workflow.compile(**compile_kwargs)
