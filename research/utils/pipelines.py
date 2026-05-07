# -*- coding: utf-8 -*-
"""Pipeline variants for RQ1 (Gemini-only vs RAG baseline vs Agentic RAG).

Every variant exposes the same shape:
    run(question: str) -> dict
        {
            "answer": str,
            "contexts": list[str],        # chunk texts used (empty for Gemini-only)
            "citations": list[dict],      # list of {doc_id, dieu, khoan, diem}
            "category": str | None,       # router decision (None for Gemini-only)
            "latency_s": float,
            "cost_usd": float | None,     # None if not tracked
        }

This is a compatibility layer so the eval runner + metrics can treat all three
variants identically. Production code (source/) is imported, NOT forked.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --- sys.path so source.* is importable from notebooks ---------------------
HERE = Path(__file__).resolve().parent
TRAFFIC_RAG_ROOT = HERE.parent.parent  # .../traffic_rag
if str(TRAFFIC_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAFFIC_RAG_ROOT))


@dataclass
class PipelineResult:
    answer: str
    contexts: list[str]
    citations: list[dict]
    category: str | None
    latency_s: float
    cost_usd: float | None = None
    raw: dict | None = None

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "contexts": self.contexts,
            "citations": self.citations,
            "category": self.category,
            "latency_s": self.latency_s,
            "cost_usd": self.cost_usd,
        }


# ---------------------------------------------------------------------------
# Variant 1 — Gemini only (no retrieval)
# ---------------------------------------------------------------------------

class GeminiOnlyPipeline:
    """Straight LLM call, no retrieval. Baseline for RQ1."""

    def __init__(self, model: str = "gemini-3.1-flash-lite-preview", temperature: float = 0.1):
        from langchain_google_genai import ChatGoogleGenerativeAI

        self.model = model
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("API_KEY"),
            request_timeout=60,
        )

    def run(self, question: str) -> PipelineResult:
        from langchain_core.messages import HumanMessage, SystemMessage

        system = (
            "Bạn là trợ lý pháp lý về Luật Giao thông đường bộ Việt Nam. "
            "Trả lời ngắn gọn, chính xác. Nếu không chắc, nói rõ là không chắc chắn."
        )
        t0 = time.perf_counter()
        resp = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=question)])
        dt = time.perf_counter() - t0
        return PipelineResult(
            answer=getattr(resp, "content", str(resp)),
            contexts=[],
            citations=[],
            category=None,
            latency_s=dt,
        )


# ---------------------------------------------------------------------------
# Variant 2 — Vanilla RAG (retrieve + single-pass generate, no router/HITL)
# ---------------------------------------------------------------------------

class VanillaRAGPipeline:
    """Minimal RAG: TrafficHybridRetriever → LegalAnswerGenerator. No agent, no router."""

    def __init__(
        self,
        top_k: int = 10,
        model: str = "gemini-3.1-flash-lite-preview",
        collection_name: str | None = None,
        jsonl_path: str | None = None,
        embedding_model: str | None = None,
    ):
        from source.rag_core import LegalAnswerGenerator, TrafficHybridRetriever

        kwargs = {}
        if collection_name:
            kwargs["collection_name"] = collection_name
        if jsonl_path:
            kwargs["jsonl_path"] = jsonl_path
        if embedding_model:
            kwargs["embedding_model"] = embedding_model
        self.retriever = TrafficHybridRetriever(**kwargs)
        self.generator = LegalAnswerGenerator(provider="google", model=model)
        self.top_k = top_k

    def run(self, question: str) -> PipelineResult:
        t0 = time.perf_counter()
        chunks = self.retriever.get_relevant_chunks(question, top_k=self.top_k)
        chunk_dicts = [c.to_dict() for c in chunks]
        gen_out = self.generator.generate(question, chunk_dicts)
        dt = time.perf_counter() - t0

        contexts = [c.get("content", "") for c in chunk_dicts]
        # Citations = các nguồn generator thực sự trích (có trong answer),
        # không phải toàn bộ chunks retrieve. So sánh công bằng với agentic_rag.
        citations = gen_out.get("sources") or []
        # Surface chunks (with metadata) in `raw` so downstream eval/inspection
        # can match by (doc_id, dieu, ...) instead of regex on free text.
        raw = dict(gen_out)
        raw["chunks"] = chunk_dicts
        return PipelineResult(
            answer=gen_out.get("answer", ""),
            contexts=contexts,
            citations=citations,
            category="legal_rag",
            latency_s=dt,
            raw=raw,
        )


# ---------------------------------------------------------------------------
# Variant 3 — Agentic RAG (production graph: router + query expansion + HITL)
# ---------------------------------------------------------------------------

class AgenticRAGPipeline:
    """Full LangGraph pipeline from source/agent/.

    For eval we run synchronously and auto-approve any HITL interrupt (so the
    web-fallback branch yields a comparable answer). If you want to measure the
    strict `legal_rag`-only path, pass `auto_approve=False` and treat paused
    runs as failures.
    """

    def __init__(
        self,
        model: str = "gemini-3.1-flash-lite-preview",
        auto_approve: bool = True,
    ):
        from langchain_google_genai import ChatGoogleGenerativeAI
        from source.agent import TavilySearchTool, build_graph
        from source.rag_core import LegalAnswerGenerator, TrafficHybridRetriever

        self.retriever = TrafficHybridRetriever()
        self.generator = LegalAnswerGenerator(provider="google", model=model)
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("API_KEY"),
            request_timeout=60,
        )
        self.tavily = TavilySearchTool()
        # In-memory checkpointer so each question is isolated.
        from langgraph.checkpoint.memory import MemorySaver
        self.graph = build_graph(
            self.retriever, self.generator, self.llm, self.tavily,
            checkpointer=MemorySaver(),
        )
        self.auto_approve = auto_approve

    def run(self, question: str) -> PipelineResult:
        import uuid
        config = {"configurable": {"thread_id": f"eval-{uuid.uuid4()}"}}
        t0 = time.perf_counter()
        self.graph.invoke({"query": question, "raw_query": question}, config)
        snap = self.graph.get_state(config)
        # Resume through HITL if paused.
        if snap and snap.next and self.auto_approve:
            self.graph.invoke(None, config)
            snap = self.graph.get_state(config)
        dt = time.perf_counter() - t0

        values: dict[str, Any] = dict(snap.values) if snap and snap.values else {}
        answer = values.get("answer") or values.get("draft_answer") or ""
        raw_chunks = values.get("chunks") or []
        contexts = []
        for c in raw_chunks:
            if isinstance(c, dict):
                contexts.append(c.get("content", ""))
            elif hasattr(c, "content"):
                contexts.append(c.content)
        return PipelineResult(
            answer=answer,
            contexts=contexts,
            citations=values.get("sources") or [],
            category=values.get("category"),
            latency_s=dt,
            raw=values,
        )


# ---------------------------------------------------------------------------
# Convenience registry
# ---------------------------------------------------------------------------

def _vanilla_fixed512(**kwargs):
    defaults = dict(
        collection_name="traffic_law_fixed512",
        jsonl_path="/media/pphong/D:/Do_An_Tot_Nghiep/GitHub1/traffic_rag/Data/all_chunks_fixed512.jsonl",
        embedding_model="intfloat/multilingual-e5-small",
    )
    defaults.update(kwargs)
    return VanillaRAGPipeline(**defaults)


PIPELINE_REGISTRY = {
    "gemini_only": GeminiOnlyPipeline,
    "vanilla_rag": VanillaRAGPipeline,
    "agentic_rag": AgenticRAGPipeline,
    "vanilla_rag_fixed512": _vanilla_fixed512,
}


def build_pipeline(name: str, **kwargs):
    if name not in PIPELINE_REGISTRY:
        raise ValueError(f"Unknown pipeline '{name}'. Choose from {list(PIPELINE_REGISTRY)}")
    return PIPELINE_REGISTRY[name](**kwargs)
