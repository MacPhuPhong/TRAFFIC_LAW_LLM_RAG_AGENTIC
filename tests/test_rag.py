# -*- coding: utf-8 -*-
"""
test_rag.py — End-to-End RAG smoke test
=======================================
Runs the 3 validation queries + 1 out-of-scope refusal test through the
retriever → generator pipeline.

Usage:
    # OpenAI (default)
    export OPENAI_API_KEY=sk-...
    python traffic_rag/test_rag.py

    # Google Gemini
    export GOOGLE_API_KEY=...
    python traffic_rag/test_rag.py --provider google

    # Retrieval-only (no LLM call; does not require an API key)
    python traffic_rag/test_rag.py --no-generate
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Make `source.rag_core` importable when running from repo root or tests folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))

from rag_core import TrafficHybridRetriever, LegalAnswerGenerator  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("test_rag")


TEST_QUERIES = [
    {
        "query": "Mức trừ điểm GPLX cho hành vi vượt đèn đỏ theo quy định mới 2024?",
        "expected_doc": "168/2024/NĐ-CP",
        "description": "Xử phạt — NĐ 168/2024",
        "should_answer": True,
    },
    {
        "query": "Chu kỳ đăng kiểm lần đầu cho xe ô tô con không kinh doanh vận tải sản xuất năm 2025?",
        "expected_doc": "47/2024/TT-BGTVT",
        "description": "Kỹ thuật — TT 47/2024",
        "should_answer": True,
    },
    {
        "query": "Hồ sơ đăng ký xe trực tuyến cần những giấy tờ gì?",
        "expected_doc": "79/2024/TT-BCA",
        "description": "Thủ tục — TT 79/2024",
        "should_answer": True,
    },
    {
        "query": "Thời tiết Hà Nội hôm nay thế nào?",
        "expected_doc": None,
        "description": "Out-of-scope (phải từ chối)",
        "should_answer": False,
    },
]


def print_separator(char: str = "=", width: int = 72) -> None:
    print(char * width)


def run_case(
    case: dict,
    retriever: TrafficHybridRetriever,
    generator: LegalAnswerGenerator | None,
    top_k: int = 5,
) -> dict:
    print()
    print_separator()
    print(f"[{case['description']}]  should_answer={case['should_answer']}")
    print(f"Q: {case['query']}")
    print_separator("-")

    chunks = retriever.get_relevant_chunks(case["query"], top_k=top_k)
    found_docs = [c.metadata.get("doc_id", "") for c in chunks]
    print(f"Retrieved top-{top_k} doc_ids: {found_docs}")

    retrieval_pass = (
        case["expected_doc"] in found_docs if case["expected_doc"] else True
    )
    print(f"Retrieval check: {'✅' if retrieval_pass else '❌'}")

    result = {
        "query": case["query"],
        "expected_doc": case["expected_doc"],
        "retrieved_docs": found_docs,
        "retrieval_pass": retrieval_pass,
    }

    if generator is None:
        return result

    print()
    print("Generating answer...")
    chunk_dicts = [c.to_dict() for c in chunks]
    out = generator.generate(case["query"], chunk_dicts)

    print_separator("-")
    print(f"ANSWER [{out['model']}]:\n{out['answer']}")
    print_separator("-")
    print(f"Cited sources: {len(out['sources'])}")
    for s in out["sources"]:
        loc = f"Điều {s['dieu']}"
        if s.get("khoan"):
            loc += f", Khoản {s['khoan']}"
        if s.get("diem"):
            loc += f", Điểm {s['diem']}"
        print(f"  - {loc} — {s['ten_van_ban']} ({s['doc_id']})")

    print(f"Refused: {out['refused']}")

    if case["should_answer"]:
        generation_pass = not out["refused"] and len(out["answer"]) > 30
    else:
        generation_pass = out["refused"]
    print(f"Generation check: {'✅' if generation_pass else '❌'}")

    result.update({
        "answer": out["answer"],
        "sources": out["sources"],
        "refused": out["refused"],
        "generation_pass": generation_pass,
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider", choices=["openai", "google"], default="openai"
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--no-generate", action="store_true",
        help="Skip LLM generation (retrieval only; no API key needed).",
    )
    args = parser.parse_args()

    print_separator()
    print("TRAFFIC LAW RAG — END-TO-END SMOKE TEST")
    print_separator()

    retriever = TrafficHybridRetriever()

    generator = None
    if not args.no_generate:
        key_env = "OPENAI_API_KEY" if args.provider == "openai" else "GOOGLE_API_KEY"
        if not os.environ.get(key_env):
            print(
                f"\n⚠️  {key_env} not set — skipping generation. "
                f"Re-run with --no-generate or set the env var."
            )
        else:
            generator = LegalAnswerGenerator(
                provider=args.provider, model=args.model
            )

    results = [run_case(c, retriever, generator, top_k=args.top_k) for c in TEST_QUERIES]

    print()
    print_separator()
    print("SUMMARY")
    print_separator()
    retrieval_passes = sum(r["retrieval_pass"] for r in results)
    print(f"Retrieval: {retrieval_passes}/{len(results)} passed")
    if generator is not None:
        gen_passes = sum(r.get("generation_pass", False) for r in results)
        print(f"Generation: {gen_passes}/{len(results)} passed")
    print_separator()

    all_pass = all(r["retrieval_pass"] for r in results) and (
        generator is None
        or all(r.get("generation_pass", False) for r in results)
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
