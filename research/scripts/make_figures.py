# -*- coding: utf-8 -*-
"""Generate all PNG figures for RQ2, RQ3, RQ4, RQ5, RQ8 from CSV metrics.

Reads CSVs from research/results/metrics/ and writes PNGs to
research/results/figures/. Every figure is dpi=120, bbox_inches='tight'.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("figs")

M = ROOT / "research" / "results" / "metrics"
F = ROOT / "research" / "results" / "figures"
F.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "savefig.dpi": 120, "savefig.bbox": "tight"})


def save(fig, name):
    path = F / name
    fig.savefig(path)
    plt.close(fig)
    log.info("wrote %s (%.1f KB)", path.name, path.stat().st_size / 1024)


# -----------------------------------------------------------------------------
# RQ2 — chunking
# -----------------------------------------------------------------------------

def rq2_figures():
    ret = pd.read_csv(M / "rq2_retrieval_summary.csv")
    ret = ret.set_index("chunker")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    chunkers = ret.index.tolist()
    metrics = ["recall@5_khoan", "recall@10_khoan", "recall@20_khoan",
               "mrr_khoan", "ndcg@10_khoan"]
    labels = ["R@5", "R@10", "R@20", "MRR", "nDCG@10"]
    x = np.arange(len(metrics))
    w = 0.38
    for i, c in enumerate(chunkers):
        ax.bar(x + (i - 0.5) * w, [ret.loc[c, m] for m in metrics], w, label=c)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("RQ2 — Retrieval metrics (khoan granularity)\nHierarchical vs Fixed-512")
    ax.legend()
    save(fig, "rq2_chunking_retrieval.png")

    # answer-level
    if (M / "rq2_answer_summary.csv").exists():
        ans = pd.read_csv(M / "rq2_answer_summary.csv").set_index("chunker")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        metrics = ["f1_token", "rouge_l", "cit_r_khoan", "cit_r_doc", "refusal"]
        labels = ["F1 token", "ROUGE-L", "Cit-R\n(khoan)", "Cit-R\n(doc_id)", "Refusal rate"]
        x = np.arange(len(metrics))
        for i, c in enumerate(ans.index.tolist()):
            ax.bar(x + (i - 0.5) * w, [ans.loc[c, m] for m in metrics], w, label=c)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Score")
        ax.set_title("RQ2 — End-to-end vanilla RAG quality\nHierarchical vs Fixed-512")
        ax.legend()
        save(fig, "rq2_chunking_answer.png")

    # per-category F1
    if (M / "rq2_answer_per_question.csv").exists():
        df = pd.read_csv(M / "rq2_answer_per_question.csv")
        cat_df = df.groupby(["category", "chunker"])["f1_token"].mean().unstack()
        fig, ax = plt.subplots(figsize=(9, 4.5))
        cat_df.plot(kind="bar", ax=ax)
        ax.set_ylabel("F1 token (mean)")
        ax.set_title("RQ2 — F1 per category × chunker")
        ax.set_xlabel("")
        plt.xticks(rotation=25, ha="right")
        save(fig, "rq2_chunking_per_category.png")


# -----------------------------------------------------------------------------
# RQ3 — embedding
# -----------------------------------------------------------------------------

def rq3_figures():
    df = pd.read_csv(M / "rq3_embedding.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    metrics = ["recall@5", "recall@10", "mrr", "ndcg@10"]
    x = np.arange(len(metrics))
    w = 0.26
    for i, row in df.iterrows():
        ax.bar(x + (i - 1) * w, [row[m] for m in metrics], w, label=row["model"])
    ax.set_xticks(x)
    ax.set_xticklabels(["R@5", "R@10", "MRR", "nDCG@10"])
    ax.set_ylabel("Score")
    ax.set_title("RQ3 — Embedding comparison (khoan-level retrieval)")
    ax.legend()
    save(fig, "rq3_embedding_recall.png")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.bar(df["model"], df["encode_corpus_s"], color="tab:blue")
    a1.set_ylabel("Encode corpus (s)")
    a1.set_title("Corpus encode time (2705 chunks)")
    for i, v in enumerate(df["encode_corpus_s"]):
        a1.text(i, v, f" {v:.0f}", ha="center", va="bottom")
    plt.setp(a1.get_xticklabels(), rotation=20, ha="right")

    a2.bar(df["model"], df["query_encode_ms"], color="tab:orange")
    a2.set_ylabel("Query encode (ms / query)")
    a2.set_title("Per-query encode latency")
    for i, v in enumerate(df["query_encode_ms"]):
        a2.text(i, v, f" {v:.0f}", ha="center", va="bottom")
    plt.setp(a2.get_xticklabels(), rotation=20, ha="right")
    save(fig, "rq3_embedding_speed.png")


# -----------------------------------------------------------------------------
# RQ4 — vector DB
# -----------------------------------------------------------------------------

def rq4_figures():
    df = pd.read_csv(M / "rq4_vectordb_real.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(df))
    w = 0.32
    ax.bar(x - w/2, df["p50_ms"], w, label="p50 (ms)", color="tab:blue")
    ax.bar(x + w/2, df["p95_ms"], w, label="p95 (ms)", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(df["backend"], rotation=15, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"RQ4 — Vector-DB latency  (n={int(df['n_points'].iloc[0])} pts, "
                 f"dim={int(df['dim'].iloc[0])}, top-10, 100 repeats)")
    ax.legend()
    save(fig, "rq4_latency_comparison.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sizes = (df["p50_ms"] * 80).clip(lower=40)
    ax.scatter(df["peak_rss_mb"], df["recall@10"], s=sizes, alpha=0.6)
    for _, r in df.iterrows():
        ax.annotate(r["backend"].split(" ")[0],
                    (r["peak_rss_mb"], r["recall@10"]),
                    textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("Peak RSS (MB)  — in-process backends only")
    ax.set_ylabel("Recall@10 (khoan)")
    ax.set_title("RQ4 — Memory vs quality (marker size ∝ p50 latency)")
    save(fig, "rq4_memory_recall_tradeoff.png")


# -----------------------------------------------------------------------------
# RQ5 — prompt
# -----------------------------------------------------------------------------

def rq5_figures():
    df = pd.read_csv(M / "rq5_prompts_summary.csv").set_index("variant")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    metrics = ["f1_token", "rouge_l", "cit_p", "cit_r", "cit_f1", "refusal"]
    labels = ["F1", "ROUGE-L", "Cit-P", "Cit-R", "Cit-F1", "Refusal"]
    x = np.arange(len(metrics))
    w = 0.26
    for i, v in enumerate(df.index):
        ax.bar(x + (i - 1) * w, [df.loc[v, m] for m in metrics], w, label=v)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("RQ5 — Prompt ablation (P0 zero-shot · P1 role · P2 full rules)")
    ax.legend()
    save(fig, "rq5_prompts_comparison.png")

    per_q = pd.read_csv(M / "rq5_prompts_per_question.csv")
    heat = per_q.groupby(["category", "variant"])["f1_token"].mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(heat.values, aspect="auto", cmap="viridis",
                   vmin=0, vmax=max(0.5, heat.values.max()))
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_title("RQ5 — F1 per category × prompt variant")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat.values[i, j]:.2f}", ha="center", va="center",
                    color="white" if heat.values[i, j] < 0.25 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="F1")
    save(fig, "rq5_prompts_per_category.png")


# -----------------------------------------------------------------------------
# RQ8 — LangSmith
# -----------------------------------------------------------------------------

def rq8_figures():
    df = pd.read_csv(M / "rq8_langsmith_nodes.csv")
    df = df.head(10).copy()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(df))
    ax.bar(x, df["mean_s"], 0.4, label="mean", color="tab:blue")
    ax.bar(x, df["p95_s"] - df["mean_s"], 0.4, bottom=df["mean_s"],
           label="mean→p95", color="tab:orange", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(df["run_name"], rotation=20, ha="right")
    ax.set_ylabel("Latency (s)")
    src = df["source"].iloc[0] if "source" in df.columns else ""
    ax.set_title(f"RQ8 — Latency breakdown per run node (source: {src})")
    ax.legend()
    save(fig, "rq8_node_breakdown.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["run_name"], df["error_rate"], color="tab:red")
    ax.set_ylabel("Error rate")
    ax.set_title("RQ8 — Error rate per node")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    save(fig, "rq8_error_rate.png")

    # per-pipeline latency histogram from rq1 JSONL
    import json
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, color in [("gemini_only", "tab:green"),
                        ("vanilla_rag", "tab:blue"),
                        ("agentic_rag", "tab:red")]:
        fp = M / f"rq1_{name}.jsonl"
        if not fp.exists():
            continue
        lats = [json.loads(l)["latency_s"]
                for l in open(fp, encoding="utf-8") if l.strip()]
        ax.hist(lats, bins=12, alpha=0.55, label=name, color=color)
    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Count")
    ax.set_title("RQ8 — Per-pipeline latency distribution (25-question eval)")
    ax.legend()
    save(fig, "rq8_latency_hist.png")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    for fn, flag in [
        (rq2_figures, M / "rq2_retrieval_summary.csv"),
        (rq3_figures, M / "rq3_embedding.csv"),
        (rq4_figures, M / "rq4_vectordb_real.csv"),
        (rq5_figures, M / "rq5_prompts_summary.csv"),
        (rq8_figures, M / "rq8_langsmith_nodes.csv"),
    ]:
        if not flag.exists():
            log.warning("skip %s — %s not found", fn.__name__, flag.name)
            continue
        try:
            fn()
        except Exception as e:
            log.error("%s failed: %s", fn.__name__, e)


if __name__ == "__main__":
    main()
