# -*- coding: utf-8 -*-
"""Metric helpers for the research notebooks.

Covers the four columns used in the thesis comparison table:
    F1 (token-level) · ROUGE-L · cosine similarity (Google embeddings) · cost.

Plus retrieval metrics (Recall@k, MRR, nDCG) and citation precision/recall.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

__all__ = [
    "token_f1",
    "rouge_l",
    "cosine_sim",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "citation_precision_recall",
    "normalize_vn",
]


# --- text normalization -----------------------------------------------------

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_vn(text: str) -> str:
    """Lower, strip punctuation, collapse whitespace. Keep Vietnamese diacritics."""
    if not text:
        return ""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    return normalize_vn(text).split()


# --- token-level F1 ---------------------------------------------------------

def token_f1(pred: str, gold: str) -> float:
    """SQuAD-style token F1."""
    pred_toks = _tokenize(pred)
    gold_toks = _tokenize(gold)
    if not pred_toks or not gold_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_toks)
    recall = n_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


# --- ROUGE-L ----------------------------------------------------------------

def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            tmp = dp[j]
            if x == y:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[-1]


def rouge_l(pred: str, gold: str, beta: float = 1.2) -> float:
    """ROUGE-L F-measure (Lin 2004 default β=1.2)."""
    pred_toks = _tokenize(pred)
    gold_toks = _tokenize(gold)
    if not pred_toks or not gold_toks:
        return 0.0
    lcs = _lcs_len(pred_toks, gold_toks)
    if lcs == 0:
        return 0.0
    p = lcs / len(pred_toks)
    r = lcs / len(gold_toks)
    if p == 0 and r == 0:
        return 0.0
    return ((1 + beta**2) * p * r) / (r + (beta**2) * p)


# --- cosine similarity ------------------------------------------------------

def cosine_sim(a: Iterable[float], b: Iterable[float]) -> float:
    """Plain cosine sim between two vectors (any iterable of floats)."""
    a = list(a)
    b = list(b)
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# --- retrieval metrics ------------------------------------------------------

def _norm_cite_val(v):
    """Normalize a citation field to string form.

    Gold data stores `khoan`/`dieu` as int; chunk payloads store `khoan` as str
    and `dieu` as int. Without normalization the tuples never compare equal.
    """
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return str(int(v)) if float(v).is_integer() else str(v)
    return str(v).strip().lower()


def _citation_key(c: dict, level: str = "khoan") -> tuple:
    """Turn a citation dict into a hashable key at the requested granularity."""
    parts = [_norm_cite_val(c.get("doc_id")), _norm_cite_val(c.get("dieu"))]
    if level in ("khoan", "diem"):
        parts.append(_norm_cite_val(c.get("khoan")))
    if level == "diem":
        parts.append(_norm_cite_val(c.get("diem")))
    return tuple(parts)


def recall_at_k(retrieved: list[dict], gold: list[dict], k: int, level: str = "khoan") -> float:
    """What fraction of gold citations are present in the top-k retrieved chunks?"""
    if not gold:
        return 1.0
    gold_keys = {_citation_key(g, level) for g in gold}
    top = retrieved[:k]
    hit_keys = {_citation_key(r, level) for r in top}
    return len(gold_keys & hit_keys) / len(gold_keys)


def mrr(retrieved: list[dict], gold: list[dict], level: str = "khoan") -> float:
    """Mean reciprocal rank of the first relevant hit (1/position, 0 if absent)."""
    if not gold:
        return 0.0
    gold_keys = {_citation_key(g, level) for g in gold}
    for idx, r in enumerate(retrieved, 1):
        if _citation_key(r, level) in gold_keys:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved: list[dict], gold: list[dict], k: int, level: str = "khoan") -> float:
    """Binary relevance nDCG@k."""
    if not gold:
        return 0.0
    gold_keys = {_citation_key(g, level) for g in gold}
    dcg = 0.0
    for i, r in enumerate(retrieved[:k], 1):
        if _citation_key(r, level) in gold_keys:
            dcg += 1.0 / math.log2(i + 1)
    # ideal DCG with min(|gold|, k) perfect hits
    ideal_hits = min(len(gold_keys), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# --- citation P/R (for answer-level citation evaluation) --------------------

def citation_precision_recall(
    pred: list[dict], gold: list[dict], level: str = "khoan"
) -> tuple[float, float, float]:
    """Precision, recall, F1 of citations emitted by the generator."""
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    pred_keys = {_citation_key(p, level) for p in pred}
    gold_keys = {_citation_key(g, level) for g in gold}
    if not pred_keys:
        return 0.0, 0.0, 0.0
    tp = len(pred_keys & gold_keys)
    p = tp / len(pred_keys)
    r = tp / len(gold_keys) if gold_keys else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1
