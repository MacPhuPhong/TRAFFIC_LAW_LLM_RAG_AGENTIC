# -*- coding: utf-8 -*-
"""Generate RQ10 reranker comparison figure."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
F = ROOT / "research" / "results" / "figures"
F.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(ROOT / "research" / "results" / "metrics" / "rq10_reranker_summary.csv")
    df = df.set_index("mode")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Quality metrics
    cols = ["recall@10", "mrr", "ndcg@10"]
    df[cols].plot.bar(ax=axes[0], color=["#4C72B0", "#DD8452", "#55A467"])
    axes[0].set_title("RQ10 — Reranker quality (35-câu legal-only)")
    axes[0].set_ylabel("score")
    axes[0].set_xticklabels(df.index, rotation=0)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)
    for c_i, c in enumerate(cols):
        for i, v in enumerate(df[c]):
            axes[0].annotate(f"{v:.3f}", xy=(i + (c_i - 1) * 0.27, v + 0.005),
                             ha="center", fontsize=8)

    # Latency
    axes[1].bar(df.index, df["latency_ms_mean"], color=["#4C72B0", "#C44E52"])
    axes[1].set_title("RQ10 — Latency (ms)")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("ms (log scale)")
    for i, v in enumerate(df["latency_ms_mean"]):
        axes[1].annotate(f"{v:.0f}", xy=(i, v * 1.1), ha="center", fontsize=9)
    axes[1].grid(axis="y", linestyle=":", alpha=0.5, which="both")

    fig.tight_layout()
    path = F / "rq10_reranker_comparison.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
