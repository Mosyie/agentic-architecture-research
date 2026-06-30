"""Generate the paper figures from the aggregated results CSV.

Reads the tidy long-format CSV produced by
``aggregate_results.py --csv <path>`` and writes PNG figures into ``figures/``.

Figures:
  fig1_performance.png  -- grouped bars, accuracy per architecture x difficulty
                           (HotpotQA F1 | MiniWoB reward)
  fig3_cost.png         -- cost bars: avg tokens/sample and avg API calls, per domain
                           (C1 bars carry an amortised memory-training cap)

Usage:
  python scripts/plot_results.py [--csv results/aggregate_summary.csv]
                                 [--out figures]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, file output only
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

# The one-time C1 memory-training cost is stored next to the frozen rules; it is
# used to draw the amortised training cap on the C1 cost bars.
TRAIN_DIRS = {"hotpotQA": "memory/c1_hotpotqa", "miniWOB": "memory/c1_miniwob"}
TRAIN_COLOR = "#BBBBBB"

# Display order and labels. baseline first (HotpotQA only), then A1->C1.
ARCH_ORDER = ["baseline", "a1", "b1", "b2", "c1"]
ARCH_LABELS = {
    "baseline": "Baseline",
    "a1": "A1\nSingle",
    "b1": "B1\nPlanner",
    "b2": "B2\nCritic",
    "c1": "C1\nMemory",
}
DIFF_ORDER = ["easy", "medium", "hard"]
DIFF_COLORS = {"easy": "#4C9F70", "medium": "#E0A030", "hard": "#C0504D"}


def archs_present(df):
    """Architectures actually present in df, in canonical order."""
    return [a for a in ARCH_ORDER if a in set(df["arch"])]


def grouped_bars(ax, df, value_col, title, ylabel):
    """Grouped bar chart: x = architecture, bars grouped by difficulty."""
    archs = archs_present(df)
    x = range(len(archs))
    n = len(DIFF_ORDER)
    width = 0.8 / n
    for i, diff in enumerate(DIFF_ORDER):
        vals = []
        for arch in archs:
            row = df[(df["arch"] == arch) & (df["difficulty"] == diff)]
            vals.append(row[value_col].iloc[0] if len(row) else 0.0)
        offsets = [xi + (i - (n - 1) / 2) * width for xi in x]
        bars = ax.bar(offsets, vals, width, label=diff.capitalize(),
                      color=DIFF_COLORS[diff], edgecolor="black", linewidth=0.4)
        ax.bar_label(bars, fmt="%.2f", fontsize=6, padding=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels([ARCH_LABELS[a] for a in archs], fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)


def fig_performance(df, out):
    hp = df[df["benchmark"] == "hotpotQA"]
    mw = df[df["benchmark"] == "miniWOB"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    grouped_bars(axes[0], hp, "avg_f1",
                 "HotpotQA (static multi-hop QA)", "Average F1")
    grouped_bars(axes[1], mw, "avg_reward",
                 "MiniWoB++ (interactive UI)", "Average reward")
    axes[1].legend(title="Difficulty", fontsize=8, title_fontsize=8)
    fig.suptitle("Task performance by agent architecture and difficulty",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(out / "fig1_performance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def cost_bars(ax, df, value_col, title, ylabel, train_amort=None):
    archs = archs_present(df)
    x = range(len(archs))
    n = len(DIFF_ORDER)
    width = 0.8 / n
    for i, diff in enumerate(DIFF_ORDER):
        vals = []
        for arch in archs:
            row = df[(df["arch"] == arch) & (df["difficulty"] == diff)]
            vals.append(row[value_col].iloc[0] if len(row) else 0.0)
        offsets = [xi + (i - (n - 1) / 2) * width for xi in x]
        ax.bar(offsets, vals, width, color=DIFF_COLORS[diff],
               edgecolor="black", linewidth=0.4)
        if train_amort:  # one-time C1 training, amortised, stacked on C1 only
            extra = [train_amort if arch == "c1" else 0.0 for arch in archs]
            ax.bar(offsets, extra, width, bottom=vals, color=TRAIN_COLOR,
                   hatch="////", edgecolor="black", linewidth=0.4)
    ax.set_xticks(list(x))
    ax.set_xticklabels([ARCH_LABELS[a] for a in archs], fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)


def training_cost(benchmark, df):
    """Amortised C1 training cost per evaluation sample: (tokens, api_calls).

    Reads the one-time training totals from the frozen-rules file and divides by
    the number of C1 evaluation samples for this benchmark. Returns (None, None)
    if the file or the rows are missing."""
    folder = TRAIN_DIRS.get(benchmark)
    if not folder:
        return None, None
    path = Path(folder) / "frozen_rules.json"
    if not path.exists():
        return None, None
    info = json.loads(path.read_text(encoding="utf-8"))
    c1 = df[(df["benchmark"] == benchmark) & (df["arch"] == "c1")]
    test_n = c1["num_samples"].sum()
    if not test_n:
        return None, None
    return (info.get("total_tokens", 0) / test_n,
            info.get("total_api_calls", 0) / test_n)


def fig_cost(df, out):
    """Cost across both domains: rows = domain, cols = tokens / API calls.

    The C1 bars carry an extra hatched cap: the one-time memory-training cost
    amortised over the evaluation set."""
    hp = df[df["benchmark"] == "hotpotQA"]
    mw = df[df["benchmark"] == "miniWOB"]
    hp_tok, hp_api = training_cost("hotpotQA", df)
    mw_tok, mw_api = training_cost("miniWOB", df)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    cost_bars(axes[0, 0], hp, "avg_tokens_per_sample",
              "HotpotQA: token cost", "Avg. tokens per sample", train_amort=hp_tok)
    cost_bars(axes[0, 1], hp, "avg_api_calls",
              "HotpotQA: API-call cost", "Avg. API calls per sample", train_amort=hp_api)
    cost_bars(axes[1, 0], mw, "avg_tokens_per_sample",
              "MiniWoB++: token cost", "Avg. tokens per sample", train_amort=mw_tok)
    cost_bars(axes[1, 1], mw, "avg_api_calls",
              "MiniWoB++: API-call cost", "Avg. API calls per sample", train_amort=mw_api)
    handles = [Patch(facecolor=DIFF_COLORS[d], edgecolor="black",
                     label=d.capitalize()) for d in DIFF_ORDER]
    handles.append(Patch(facecolor=TRAIN_COLOR, hatch="////", edgecolor="black",
                         label="C1 training (amortised)"))
    axes[0, 1].legend(handles=handles, fontsize=8)
    fig.suptitle("Computational cost of each architecture, by domain",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(out / "fig3_cost.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/aggregate_summary.csv",
                        help="Aggregated summary CSV (from aggregate_results.py --csv).")
    parser.add_argument("--out", default="figures",
                        help="Directory to write PNG figures into.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig_performance(df, out)
    fig_cost(df, out)
    print(f"Wrote 2 figures to {out}/")


if __name__ == "__main__":
    main()
