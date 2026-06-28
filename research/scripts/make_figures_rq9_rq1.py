# -*- coding: utf-8 -*-
"""Generate RQ1 and RQ9 figures (not covered by make_figures.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
M = ROOT / "research" / "results" / "metrics"
F = ROOT / "research" / "results" / "figures"
F.mkdir(parents=True, exist_ok=True)
PLOTS = ROOT / "research" / "results" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def rq9():
    df = pd.read_csv(M / "rq9_rewrite_per_question.csv")
    summary = df.groupby("variant").agg({
        "f1_token": "mean", "rouge_l": "mean",
        "cit_p": "mean", "cit_r": "mean", "cit_f1": "mean",
        "r_at_10": "mean", "mrr": "mean",
        "refusal": "mean", "latency_s": "mean",
    }).round(4)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cols = ["f1_token", "rouge_l", "r_at_10", "cit_f1"]
    summary[cols].plot.bar(ax=axes[0])
    axes[0].set_title("RQ9 — Rewrite vs No-Rewrite (answer + retrieval)")
    axes[0].set_ylabel("score")
    axes[0].set_xticklabels(summary.index, rotation=0)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)

    # Per-category recall
    per_cat = df.groupby(["variant", "category"])["r_at_10"].mean().unstack("variant")
    per_cat.plot.bar(ax=axes[1])
    axes[1].set_title("RQ9 — Recall@10 per category")
    axes[1].set_ylabel("recall@10")
    axes[1].set_xticklabels(per_cat.index, rotation=20, ha="right")
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(F / "rq9_rewrite_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote rq9_rewrite_comparison.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    per_cat.plot.bar(ax=ax)
    ax.set_title("RQ9 — Recall@10 by category (NO_REWRITE vs REWRITE)")
    ax.set_xticklabels(per_cat.index, rotation=20, ha="right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(F / "rq9_rewrite_per_category.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote rq9_rewrite_per_category.png")


def rq1():
    df = pd.read_csv(M / "rq1_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax2 = ax.twinx()

    x = range(len(df))
    ax.bar([i - 0.2 for i in x], df["f1"], width=0.2, label="F1 (token)", color="#4f8cf5")
    ax.bar([i for i in x], df["rouge_l"], width=0.2, label="ROUGE-L", color="#65c19a")
    ax2.bar([i + 0.2 for i in x], df["latency_s"], width=0.2,
            label="latency (s)", color="#f0b04a", alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["pipeline"])
    ax.set_ylabel("Answer quality (F1 / ROUGE-L)")
    ax2.set_ylabel("Latency (s)")
    ax.set_title("RQ1 — Gemini vs Vanilla RAG vs Agentic RAG (25 gold)")
    ax.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(PLOTS / "rq1_pipeline_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote rq1_pipeline_comparison.png")


def rq6():
    df = pd.read_csv(M / "rq6_retrieval_per_question.csv")
    means = df[[c for c in df.columns if c.startswith("recall@") or c in ("mrr", "ndcg@10")]].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    means.plot.bar(ax=ax, color="#4f8cf5")
    ax.set_title("RQ6 — Hybrid retrieval (Qdrant e5-base + BM25 + RRF) on 25 gold")
    ax.set_ylabel("mean score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(PLOTS / "rq6_retrieval_metrics.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote rq6_retrieval_metrics.png")


if __name__ == "__main__":
    rq1()
    rq6()
    rq9()
