# -*- coding: utf-8 -*-
"""LangSmith-native evaluators for the Traffic-RAG project.

Each evaluator follows LangSmith's expected signature:

    def my_eval(run, example) -> dict:
        return {"key": "metric_name", "score": 0.0..1.0, "comment": str}

`run`     — the produced run (has `outputs={"answer", "contexts", "citations", ...}`)
`example` — the dataset example (has `inputs={"question"}` and
            `outputs={"gold_answer", "gold_citations"}`)

Pass any of these to `langsmith.evaluation.evaluate(...)`:

    from langsmith.evaluation import evaluate
    from research.utils.langsmith_evaluators import ALL_EVALUATORS

    evaluate(
        target_fn,                        # question -> dict (answer, contexts, citations)
        data="traffic-rag-eval-v1",       # dataset name on LangSmith
        evaluators=ALL_EVALUATORS,
        experiment_prefix="agentic-rag",
    )

Two LLM-judge metrics (faithfulness, answer_correctness) need a Gemini key.
The other three (citation_correctness, refusal_appropriateness, context_recall)
are purely heuristic — fast, free, deterministic.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REFUSAL_PHRASE = "Thông tin này không có trong tài liệu được cung cấp."


def _outputs(run) -> dict:
    """Tolerant accessor — `run` may be a dict or a langsmith Run object."""
    if isinstance(run, dict):
        return run.get("outputs") or {}
    return getattr(run, "outputs", None) or {}


def _example_outputs(example) -> dict:
    if isinstance(example, dict):
        return example.get("outputs") or {}
    return getattr(example, "outputs", None) or {}


def _example_inputs(example) -> dict:
    if isinstance(example, dict):
        return example.get("inputs") or {}
    return getattr(example, "inputs", None) or {}


def _parse_score(text: str) -> float:
    """Pull `SCORE: <float>` out of a judge response. NaN if missing."""
    text = text or ""
    m = re.search(r"SCORE\s*[:=]\s*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            return float("nan")
    floats = re.findall(r"-?\d+(?:\.\d+)?", text)
    if floats:
        try:
            return max(0.0, min(1.0, float(floats[-1])))
        except ValueError:
            return float("nan")
    return float("nan")


@lru_cache(maxsize=1)
def _judge_llm():
    """Build the judge LLM once per process (LRU-cached)."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY missing — cannot run LLM-judge evaluators")
    model = os.getenv("LANGSMITH_JUDGE_MODEL", "gemini-flash-lite-latest")
    return ChatGoogleGenerativeAI(model=model, temperature=0.0, google_api_key=key)


def _judge(prompt: str) -> tuple[float, str]:
    """Call the judge once. Returns (score, raw_reasoning)."""
    try:
        resp = _judge_llm().invoke(prompt)
        text = getattr(resp, "content", str(resp))
        return _parse_score(text), text
    except Exception as exc:
        return float("nan"), f"judge error: {type(exc).__name__}: {exc}"


def _format_contexts(contexts: list[Any], max_chars: int = 600) -> str:
    parts = []
    for i, c in enumerate(contexts, 1):
        if isinstance(c, dict):
            text = c.get("content") or c.get("page_content") or ""
        else:
            text = str(c)
        parts.append(f"[{i}] {text[:max_chars]}")
    return "\n\n".join(parts) if parts else "(không có ngữ cảnh)"


# ---------------------------------------------------------------------------
# 1) Faithfulness — câu trả lời có grounded vào contexts không?
# ---------------------------------------------------------------------------

_PROMPT_FAITHFULNESS = """Bạn là chuyên gia đánh giá. Đánh giá CÂU TRẢ LỜI có trung thực với NGỮ CẢNH không.

Quy trình:
1. Tách CÂU TRẢ LỜI thành claims (số tiền phạt, điều/khoản, hành vi, hệ quả).
2. Mỗi claim đánh dấu: supported / partial / contradicted / unsupported.
3. faithfulness = (#supported + 0.5*#partial) / total_claims.

Trả về CHÍNH XÁC định dạng:
REASONING: <1-3 câu>
SCORE: <0.00-1.00>

NGỮ CẢNH:
{contexts}

CÂU TRẢ LỜI:
{answer}"""


def faithfulness_evaluator(run, _example) -> dict:
    # `_example` is required by the LangSmith evaluator signature but unused —
    # faithfulness compares answer ↔ contexts, not against the gold reference.
    out = _outputs(run)
    answer = (out.get("answer") or "").strip()
    contexts = out.get("contexts") or []
    if not answer or not contexts:
        return {"key": "faithfulness", "score": 0.0,
                "comment": "no answer or no contexts"}
    score, reasoning = _judge(
        _PROMPT_FAITHFULNESS.format(
            contexts=_format_contexts(contexts), answer=answer,
        )
    )
    return {"key": "faithfulness", "score": score, "comment": reasoning[:500]}


# ---------------------------------------------------------------------------
# 2) Answer correctness — câu trả lời có khớp gold không?
# ---------------------------------------------------------------------------

_PROMPT_ANSWER_CORRECTNESS = """Bạn là chuyên gia luật giao thông. So sánh CÂU TRẢ LỜI của hệ thống với CÂU TRẢ LỜI ĐÚNG (gold).

Tiêu chí:
- Mức phạt tiền có khớp khoảng giá trị gold không (chấp nhận sai số nhỏ)?
- Số điểm GPLX bị trừ có khớp không?
- Hành vi vi phạm có được mô tả đúng không?

Quy đổi:
- 1.0 = đầy đủ và chính xác mọi facts trong gold.
- 0.7-0.9 = đúng ý chính, thiếu 1 chi tiết phụ (vd: không nêu số điểm trừ).
- 0.4-0.6 = đúng một phần, sai 1 fact quan trọng.
- 0.0-0.3 = sai hoặc không trả lời.

REASONING: <giải thích các điểm khớp/sai>
SCORE: <0.00-1.00>

CÂU TRẢ LỜI ĐÚNG:
{gold}

CÂU TRẢ LỜI CỦA HỆ THỐNG:
{answer}"""


def answer_correctness_evaluator(run, example) -> dict:
    out = _outputs(run)
    answer = (out.get("answer") or "").strip()
    gold = (_example_outputs(example).get("gold_answer") or "").strip()
    if not gold:
        return {"key": "answer_correctness", "score": float("nan"),
                "comment": "no gold answer in example"}
    if not answer:
        return {"key": "answer_correctness", "score": 0.0,
                "comment": "system produced empty answer"}
    score, reasoning = _judge(
        _PROMPT_ANSWER_CORRECTNESS.format(gold=gold, answer=answer),
    )
    return {"key": "answer_correctness", "score": score, "comment": reasoning[:500]}


# ---------------------------------------------------------------------------
# 3) Citation correctness — heuristic, domain-specific (Vietnamese legal)
# ---------------------------------------------------------------------------

def _normalize_citation(c: dict) -> tuple:
    """Tuple form for citation comparison.

    Empty/missing fields collapse to None so a gold citation that omits
    `khoan`/`diem` (because the question is "loose") still matches any
    predicted citation under the same (doc_id, dieu).
    """
    def _norm(v, lower=False):
        if v in (None, ""):
            return None
        s = str(v).strip()
        return s.lower() if lower else s
    return (
        _norm(c.get("doc_id"), lower=True),
        _norm(c.get("dieu")),
        _norm(c.get("khoan")),
        _norm(c.get("diem"), lower=True),
    )


def _citation_match(gold: tuple, pred: tuple) -> str:
    """Return 'exact' / 'partial' / 'none'.

    'exact'   = every non-empty gold field equals pred (gold ⊆ pred semantics)
    'partial' = doc_id + dieu match, but a gold-specified khoan/diem differs
    'none'    = doc_id or dieu mismatch
    """
    gd, gdi, gk, gdiem = gold
    pd, pdi, pk, pdiem = pred
    if gd is None or pd is None or gd != pd:
        return "none"
    if gdi is None or pdi is None or gdi != pdi:
        return "none"
    # doc_id + dieu match. Now check tighter fields ONLY where gold specifies them.
    if gk is not None and gk != pk:
        return "partial"
    if gdiem is not None and gdiem != pdiem:
        return "partial"
    return "exact"


def citation_correctness_evaluator(run, example) -> dict:
    """For each gold citation: best match across all predictions = exact (1.0)
    / partial (0.5) / none (0). Score = mean over gold citations.

    Tolerant of `khoan`/`diem` being omitted from gold — those fields only
    constrain matching when the gold actually specifies them. Avoids penalising
    the model for being more specific than gold.
    """
    out = _outputs(run)
    pred = out.get("citations") or out.get("sources") or []
    gold = _example_outputs(example).get("gold_citations") or []
    if not gold:
        return {"key": "citation_correctness", "score": float("nan"),
                "comment": "no gold citations"}
    if not pred:
        return {"key": "citation_correctness", "score": 0.0,
                "comment": "model emitted no citations"}

    gold_tuples = [_normalize_citation(c) for c in gold]
    pred_tuples = [_normalize_citation(c) for c in pred]

    n_exact = 0
    n_partial = 0
    n_miss = 0
    for g in gold_tuples:
        best = "none"
        for p in pred_tuples:
            m = _citation_match(g, p)
            if m == "exact":
                best = "exact"
                break
            if m == "partial" and best != "exact":
                best = "partial"
        if best == "exact":
            n_exact += 1
        elif best == "partial":
            n_partial += 1
        else:
            n_miss += 1

    score = (n_exact + 0.5 * n_partial) / max(len(gold_tuples), 1)
    score = max(0.0, min(1.0, score))
    return {
        "key": "citation_correctness",
        "score": round(score, 3),
        "comment": (f"exact={n_exact}/{len(gold_tuples)}, "
                    f"partial={n_partial}, miss={n_miss}, "
                    f"pred_total={len(pred_tuples)}"),
    }


# ---------------------------------------------------------------------------
# 4) Refusal appropriateness — heuristic
# ---------------------------------------------------------------------------

def refusal_appropriateness_evaluator(run, example) -> dict:
    """Did the model refuse correctly?

    Logic:
      - If gold expects an answer (gold_answer present, non-empty)
        and model REFUSED → score 0 (false refusal — bad).
      - If gold itself is a refusal (no gold_answer or expected_category=='out_of_scope')
        and model ANSWERED → score 0 (hallucination — bad).
      - Otherwise → score 1.
    """
    out = _outputs(run)
    answer = (out.get("answer") or "").strip().rstrip(".")
    gold_ans = (_example_outputs(example).get("gold_answer") or "").strip()
    expected_cat = _example_inputs(example).get("expected_category") \
                   or _example_outputs(example).get("expected_category")

    model_refused = answer == REFUSAL_PHRASE.rstrip(".") or out.get("refused") is True
    gold_expects_answer = bool(gold_ans) and expected_cat not in ("out_of_scope", "chit_chat")

    if gold_expects_answer and model_refused:
        return {"key": "refusal_appropriateness", "score": 0.0,
                "comment": "false refusal — gold has an answer"}
    if (not gold_expects_answer) and (not model_refused):
        return {"key": "refusal_appropriateness", "score": 0.0,
                "comment": "should have refused (out-of-scope) but answered"}
    return {"key": "refusal_appropriateness", "score": 1.0, "comment": "ok"}


# ---------------------------------------------------------------------------
# 5) Context recall — heuristic (gold doc_id+dieu found in retrieved chunks?)
# ---------------------------------------------------------------------------

_CITATION_RX = re.compile(
    r"\b(?P<doc_id>\d+/\d{4}/[A-ZĐ\-]+)\b"
    r".*?\bĐiều\s*(?P<dieu>\d+)",
    re.IGNORECASE | re.DOTALL,
)


def _ids_in_contexts(contexts: list[Any]) -> set[tuple[str, str]]:
    """Extract (doc_id, dieu) pairs visible inside the retrieved chunks."""
    found: set[tuple[str, str]] = set()
    for c in contexts:
        if isinstance(c, dict):
            text = c.get("content") or c.get("page_content") or ""
            meta = c.get("metadata") or {}
            if meta.get("doc_id") and meta.get("dieu") not in (None, ""):
                found.add((str(meta["doc_id"]).lower(), str(meta["dieu"])))
        else:
            text = str(c)
        for m in _CITATION_RX.finditer(text or ""):
            found.add((m.group("doc_id").lower(), m.group("dieu")))
    return found


def context_recall_evaluator(run, example) -> dict:
    """Fraction of gold (doc_id, dieu) pairs present in retrieved contexts."""
    out = _outputs(run)
    contexts = out.get("contexts") or []
    gold = _example_outputs(example).get("gold_citations") or []
    if not gold:
        return {"key": "context_recall", "score": float("nan"),
                "comment": "no gold citations"}
    if not contexts:
        return {"key": "context_recall", "score": 0.0, "comment": "no retrieval"}

    gold_pairs = {(str(c.get("doc_id", "")).lower(), str(c.get("dieu", "")))
                  for c in gold if c.get("doc_id") and c.get("dieu") not in (None, "")}
    if not gold_pairs:
        return {"key": "context_recall", "score": float("nan"),
                "comment": "gold citations malformed"}

    found = _ids_in_contexts(contexts)
    hit = gold_pairs & found
    return {
        "key": "context_recall",
        "score": round(len(hit) / len(gold_pairs), 3),
        "comment": f"hit={len(hit)}/{len(gold_pairs)} (gold={sorted(gold_pairs)})",
    }


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------

LLM_JUDGE_EVALUATORS = [
    faithfulness_evaluator,
    answer_correctness_evaluator,
]

HEURISTIC_EVALUATORS = [
    citation_correctness_evaluator,
    refusal_appropriateness_evaluator,
    context_recall_evaluator,
]

ALL_EVALUATORS = LLM_JUDGE_EVALUATORS + HEURISTIC_EVALUATORS
