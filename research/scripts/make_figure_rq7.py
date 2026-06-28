# -*- coding: utf-8 -*-
"""Generate RQ7 RAGAS figure."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
F = ROOT / "research" / "results" / "figures"
F.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(ROOT / "research" / "results" / "metrics" / "rq7_ragas_summary.csv")
    # Drop NaN metrics (e.g. answer_relevancy lib-failed)
    df_clean = df.dropna(subset=["mean"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(df_clean["metric"], df_clean["mean"],
                  color=["#4C72B0", "#55A467", "#DD8452"])
    for bar, val in zip(bars, df_clean["mean"]):
        ax.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width() / 2, val + 0.02),
                    ha="center", fontsize=10)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(f"RQ7 — RAGAS (gpt-4o-mini judge, n=35, pipeline=agentic)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    path = F / "rq7_ragas.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
