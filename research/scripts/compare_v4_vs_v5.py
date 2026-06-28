"""Compare v4 baseline vs v5 (vehicle binding + parent L2 + skip-HyDE-mixed).

Reads:
  - research/results/metrics_v4_baseline/rq1_agentic_rag_v4.jsonl (25 câu cũ)
  - research/results/metrics/rq1_agentic_rag.jsonl                  (40 câu mới)

Outputs:
  - Stdout summary: F1/ROUGE-L per-category, drop/gain counts
  - research/results/metrics/rq1_v4_vs_v5_diff.csv
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from research.utils.metrics import token_f1, rouge_l  # noqa: E402


def load(p: Path) -> dict[str, dict]:
    out = {}
    if not p.exists():
        return out
    for line in p.open(encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        out[d["id"]] = d
    return out


def main():
    v4 = load(ROOT / "research" / "results" / "metrics_v4_baseline" / "rq1_agentic_rag_v4.jsonl")
    v5 = load(ROOT / "research" / "results" / "metrics" / "rq1_agentic_rag.jsonl")

    if not v5:
        print("No v5 results yet. Run rq1_pipelines.py first.")
        return

    rows = []
    for qid, r5 in sorted(v5.items()):
        gold = r5.get("gold_answer") or ""
        a5 = r5.get("answer") or ""
        f1_5 = token_f1(a5, gold)
        rl_5 = rouge_l(a5, gold)
        lat_5 = r5.get("latency_s", 0)
        cat = r5.get("category", "?")

        r4 = v4.get(qid)
        if r4:
            a4 = r4.get("answer") or ""
            f1_4 = token_f1(a4, gold)
            rl_4 = rouge_l(a4, gold)
            lat_4 = r4.get("latency_s", 0)
        else:
            f1_4 = rl_4 = lat_4 = None  # new question (no v4 baseline)

        rows.append({
            "id": qid, "category": cat,
            "f1_v4": f1_4, "f1_v5": f1_5,
            "rouge_v4": rl_4, "rouge_v5": rl_5,
            "lat_v4": lat_4, "lat_v5": lat_5,
            "delta_f1": (f1_5 - f1_4) if f1_4 is not None else None,
        })

    print(f"{'id':<8} {'cat':<18} {'F1_v4':>7} {'F1_v5':>7} {'Δ F1':>7} {'lat_v4':>7} {'lat_v5':>7}")
    print("-" * 80)
    for r in rows:
        f4 = f"{r['f1_v4']:.3f}" if r['f1_v4'] is not None else "—"
        f5 = f"{r['f1_v5']:.3f}"
        d = f"{r['delta_f1']:+.3f}" if r['delta_f1'] is not None else "NEW"
        l4 = f"{r['lat_v4']:.1f}" if r['lat_v4'] is not None else "—"
        l5 = f"{r['lat_v5']:.1f}"
        print(f"{r['id']:<8} {r['category']:<18} {f4:>7} {f5:>7} {d:>7} {l4:>7} {l5:>7}")

    # Per-category summary
    print()
    print("=== Per-category summary ===")
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    for cat, rs in sorted(by_cat.items()):
        f5_mean = sum(r["f1_v5"] for r in rs) / len(rs)
        f4_vals = [r["f1_v4"] for r in rs if r["f1_v4"] is not None]
        f4_mean = sum(f4_vals) / len(f4_vals) if f4_vals else None
        lat5 = sum(r["lat_v5"] for r in rs) / len(rs)
        f4s = f"{f4_mean:.3f}" if f4_mean is not None else "—"
        delta = f"{f5_mean - f4_mean:+.3f}" if f4_mean is not None else "NEW"
        print(f"  {cat:<18} (n={len(rs)}): F1 v4={f4s} → v5={f5_mean:.3f} ({delta}) | lat v5={lat5:.1f}s")

    # Drop/gain counts
    delta_vals = [r["delta_f1"] for r in rows if r["delta_f1"] is not None]
    gain = sum(1 for d in delta_vals if d > 0.05)
    drop = sum(1 for d in delta_vals if d < -0.05)
    eq = len(delta_vals) - gain - drop
    new_only = sum(1 for r in rows if r["delta_f1"] is None)
    print()
    print(f"Of {len(delta_vals)} questions with v4 baseline:")
    print(f"  Gains (Δ ≥ +0.05): {gain}")
    print(f"  Drops (Δ ≤ -0.05): {drop}")
    print(f"  Equal (|Δ| < 0.05): {eq}")
    print(f"  New questions (no v4 baseline): {new_only}")

    # Save CSV
    import csv
    out_csv = ROOT / "research" / "results" / "metrics" / "rq1_v4_vs_v5_diff.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
