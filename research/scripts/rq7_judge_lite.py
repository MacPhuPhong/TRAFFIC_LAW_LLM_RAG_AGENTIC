# -*- coding: utf-8 -*-
"""RQ7-lite — Custom RAGAS-style judge (1 call/metric/sample, free-tier safe).

Why this exists:
  RAGAS 0.2.0 calls the judge LLM ~60 times per sample because it internally
  decomposes the answer into atomic claims, scores each claim independently,
  then aggregates. On free-tier Gemini (15 RPM), 25 samples × 4 metrics
  × ~60 calls = throttled to 100% NaN (the original RQ7 failure mode).

This script implements the SAME 4 metric concepts as RAGAS but with **1
single-prompt judge call per (sample, metric)** instead of ~60. Total cost
for n=5 subset: 1 + 1 + 1 + 1 = 4 calls/sample × 5 = 20 calls. With 4s
sleep between calls = ~80s wall time, well under any rate limit.

Trade-off vs real RAGAS:
  - We trust the judge to do the decomposition internally in 1 prompt
    instead of orchestrating it externally. Less rigorous but defensible
    when the judge is good at structured reasoning (Gemini Flash is).
  - Score scale stays compatible [0, 1] so numbers are interpretable in
    the same way as RAGAS output.

Outputs:
  research/results/metrics/rq7_judge_lite.csv         — per-question per-metric
  research/results/metrics/rq7_judge_lite_summary.csv — aggregate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rq7-lite")

EVAL_PATH = ROOT / "research" / "data" / "eval_qa.jsonl"
AGENTIC_RESULTS = ROOT / "research" / "results" / "metrics" / "rq1_agentic_rag.jsonl"
METRICS_DIR = ROOT / "research" / "results" / "metrics"

DEFAULT_JUDGE = "gemini-flash-lite-latest"


PROMPT_FAITHFULNESS = """Bạn là chuyên gia đánh giá câu trả lời pháp lý. Đánh giá CÂU TRẢ LỜI có trung thực với NGỮ CẢNH không.

Quy trình:
1. Tách CÂU TRẢ LỜI thành các khẳng định nhỏ (claims): số tiền phạt, điều/khoản trích dẫn, hành vi vi phạm, hệ quả.
2. Với mỗi claim, đánh dấu:
   - "supported" nếu tìm thấy trực tiếp trong NGỮ CẢNH (chấp nhận paraphrase nhẹ)
   - "partial" nếu thông tin tương tự nhưng khác chi tiết nhỏ
   - "contradicted" nếu trái với NGỮ CẢNH
   - "unsupported" nếu không có trong NGỮ CẢNH (suy diễn / bịa)
3. Tính: faithfulness = (#supported + 0.5×#partial) / total_claims, làm tròn 2 chữ số.

Trả về theo định dạng:
REASONING: <giải thích ngắn gọn 1-3 câu>
SCORE: <số float 0.00-1.00>

NGỮ CẢNH:
{contexts}

CÂU TRẢ LỜI:
{answer}"""


PROMPT_ANSWER_RELEVANCY = """Bạn là chuyên gia đánh giá. Cho CÂU HỎI và CÂU TRẢ LỜI, đo mức câu trả lời hướng thẳng vào câu hỏi.

Cách tính:
- 1.0 = trả lời đầy đủ mọi ý của câu hỏi, không lan man.
- 0.7-0.9 = trả lời đúng ý chính, có thông tin phụ liên quan.
- 0.4-0.6 = trả lời một phần, thiếu khía cạnh hoặc lan man rõ.
- 0.0-0.3 = không liên quan hoặc né tránh câu hỏi.

LƯU Ý: tiêu chí là "có địa chỉ câu hỏi không" — không quan tâm câu trả lời có ĐÚNG fact hay không (cái đó là faithfulness).

Trả về:
REASONING: <1-2 câu>
SCORE: <0.00-1.00>

CÂU HỎI:
{question}

CÂU TRẢ LỜI:
{answer}"""


PROMPT_CONTEXT_PRECISION = """Bạn là chuyên gia đánh giá retrieval.

Cho CÂU HỎI và DANH SÁCH NGỮ CẢNH (đánh số 1..N theo thứ tự rank):
- Mỗi đoạn ngữ cảnh có liên quan trực tiếp đến việc trả lời câu hỏi không?
- Đếm số đoạn "relevant" (có thông tin giúp trả lời câu hỏi, dù chỉ một phần).
- Tính: precision = #relevant / N.

Lưu ý: "relevant" theo nghĩa rộng — đoạn cùng chủ đề + cùng đối tượng (vd. câu hỏi về xe máy thì đoạn về xe máy = relevant; đoạn về ô tô = không relevant).

Trả về:
REASONING: <liệt kê đánh giá relevant/irrelevant cho từng đoạn>
SCORE: <0.00-1.00>

CÂU HỎI:
{question}

DANH SÁCH NGỮ CẢNH:
{contexts}"""


PROMPT_CONTEXT_RECALL = """Bạn là chuyên gia đánh giá retrieval.

Cho CÂU TRẢ LỜI ĐÚNG (gold, đáp án chuẩn của chuyên gia) và DANH SÁCH NGỮ CẢNH:
1. Tách các fact trong gold answer: số tiền phạt, điều/khoản, hành vi, hậu quả phụ.
2. Với mỗi fact, kiểm tra xem có thể TÌM THẤY trong ngữ cảnh không (chấp nhận paraphrase, số gần đúng).
3. Tính: recall = #facts_tìm_thấy / #total_facts.

Trả về:
REASONING: <liệt kê facts + tìm thấy/không>
SCORE: <0.00-1.00>

CÂU TRẢ LỜI ĐÚNG:
{ground_truth}

DANH SÁCH NGỮ CẢNH:
{contexts}"""


def _parse_score(text: str) -> float:
    """Extract score from judge output. Prefers `SCORE: <float>` line; falls
    back to last float in text. NaN if no float found.
    """
    text = text or ""
    m = re.search(r"SCORE\s*[:=]\s*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass
    floats = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not floats:
        return float("nan")
    try:
        return max(0.0, min(1.0, float(floats[-1])))
    except ValueError:
        return float("nan")


def _format_contexts(contexts: list[str], max_per: int = 600) -> str:
    parts = []
    for i, c in enumerate(contexts, 1):
        parts.append(f"[{i}] {c[:max_per]}")
    return "\n\n".join(parts)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def build_subset(n: int) -> list[dict]:
    agentic = load_jsonl(AGENTIC_RESULTS)
    gold_by_id = {g["id"]: g for g in load_jsonl(EVAL_PATH)}
    out: list[dict] = []
    for r in agentic:
        if r.get("error") or not r.get("answer") or not r.get("contexts"):
            continue
        gold = gold_by_id.get(r["id"])
        if not gold:
            continue
        out.append({
            "id": r["id"],
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"][:5],
            "ground_truth": gold["gold_answer"],
        })
        if len(out) >= n:
            break
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--sleep", type=float, default=4.0,
                   help="Seconds to sleep between judge calls (default 4s = 15 RPM)")
    p.add_argument("--judge", default=DEFAULT_JUDGE)
    args = p.parse_args()

    load_dotenv(ROOT.parent / ".env", override=False)
    load_dotenv(ROOT / ".env", override=False)
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("API_KEY")
    if not key:
        log.error("GOOGLE_API_KEY missing")
        return 1
    os.environ["GOOGLE_API_KEY"] = key

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    subset = build_subset(args.n)
    log.info("Subset size: %d (ids=%s)", len(subset), [r["id"] for r in subset])

    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model=args.judge, temperature=0.0, google_api_key=key)

    def call_judge(prompt: str) -> float:
        for attempt in range(1, 4):
            try:
                resp = llm.invoke(prompt)
                time.sleep(args.sleep)
                return _parse_score(resp.content)
            except Exception as exc:
                wait = args.sleep * (2 ** attempt)
                log.warning("attempt=%d failed (%s) — sleep %.1fs", attempt, exc.__class__.__name__, wait)
                time.sleep(wait)
        return float("nan")

    rows = []
    for i, r in enumerate(subset, 1):
        ctx_block = _format_contexts(r["contexts"])
        log.info("[%d/%d] %s — judging…", i, len(subset), r["id"])

        s_faith = call_judge(PROMPT_FAITHFULNESS.format(contexts=ctx_block, answer=r["answer"]))
        s_relev = call_judge(PROMPT_ANSWER_RELEVANCY.format(question=r["question"], answer=r["answer"]))
        s_cprec = call_judge(PROMPT_CONTEXT_PRECISION.format(question=r["question"], contexts=ctx_block))
        s_crec  = call_judge(PROMPT_CONTEXT_RECALL.format(ground_truth=r["ground_truth"], contexts=ctx_block))

        log.info("  faith=%.3f relev=%.3f cprec=%.3f crec=%.3f",
                 s_faith, s_relev, s_cprec, s_crec)
        rows.append({
            "id":                r["id"],
            "faithfulness":      s_faith,
            "answer_relevancy":  s_relev,
            "context_precision": s_cprec,
            "context_recall":    s_crec,
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    df_path = METRICS_DIR / "rq7_judge_lite.csv"
    df.to_csv(df_path, index=False)

    summary_rows = []
    for col in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        vals = [v for v in df[col].tolist() if isinstance(v, float) and v == v]
        summary_rows.append({
            "metric": col,
            "n":      len(vals),
            "mean":   round(sum(vals) / len(vals), 4) if vals else None,
            "min":    round(min(vals), 4) if vals else None,
            "max":    round(max(vals), 4) if vals else None,
        })
    df_sum = pd.DataFrame(summary_rows)
    sum_path = METRICS_DIR / "rq7_judge_lite_summary.csv"
    df_sum.to_csv(sum_path, index=False)

    log.info("Per-question → %s", df_path)
    log.info("Summary       → %s", sum_path)
    print("\n=== RQ7 judge-lite summary (n=%d, judge=%s) ===" % (len(subset), args.judge))
    print(df_sum.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
