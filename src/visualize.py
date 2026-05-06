"""
Result visualization: tables, bar charts, routing distribution pie chart.
Saves figures to figures/ directory.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


METHOD_ORDER  = ["baseline", "opera", "vcd", "combined", "adaptive", "rper", "maver"]
METHOD_LABELS = {
    "baseline": "Baseline",
    "opera":    "OPERA-lite",
    "vcd":      "VCD-lite",
    "combined": "Naive Combined",
    "adaptive": "Adaptive Routing\n(paper)",
    "rper":     "RPER\n(Stage 3)",
    "maver":    "MAVER\n(ours, best)",
}
COLORS = {
    "baseline": "#4C72B0",
    "opera":    "#55A868",
    "vcd":      "#C44E52",
    "combined": "#8172B2",
    "adaptive": "#CCB974",
    "rper":     "#64B5CD",
    "maver":    "#E76F51",
}


def load_results(results_path: str = "results/all_results.json") -> list[dict]:
    with open(results_path) as f:
        return json.load(f)


def f1_bar_chart(results: list[dict], output_dir: str = "figures") -> None:
    Path(output_dir).mkdir(exist_ok=True)

    splits = ["random", "popular", "adversarial"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, split in zip(axes, splits):
        split_data = {r["method"]: r for r in results if r["split"] == split}
        methods = [m for m in METHOD_ORDER if m in split_data]
        f1s     = [split_data[m]["f1"] for m in methods]
        colors  = [COLORS[m] for m in methods]

        bars = ax.bar(range(len(methods)), f1s, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(f"POPE-{split.capitalize()}", fontsize=13, fontweight="bold")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(
            [METHOD_LABELS[m] for m in methods], rotation=30, ha="right", fontsize=8
        )
        ax.set_ylabel("F1 Score (%)", fontsize=11)
        ax.set_ylim(60, 95)
        ax.axhline(
            y=split_data.get("baseline", {}).get("f1", 85),
            color="gray", linestyle="--", linewidth=1.0, alpha=0.7,
            label="Baseline"
        )
        for bar, val in zip(bars, f1s):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7.5
            )

    fig.suptitle("F1 Score by Method and POPE Split", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_bar_chart.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved figures/f1_bar_chart.png")


def precision_recall_scatter(results: list[dict], output_dir: str = "figures") -> None:
    Path(output_dir).mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    splits = ["random", "popular", "adversarial"]
    for ax, split in zip(axes, splits):
        split_data = {r["method"]: r for r in results if r["split"] == split}
        for method in METHOD_ORDER:
            if method not in split_data:
                continue
            d = split_data[method]
            ax.scatter(
                d["recall"], d["precision"],
                color=COLORS[method], s=120, zorder=5,
                label=METHOD_LABELS[method], edgecolors="black", linewidths=0.5,
            )
            ax.annotate(
                METHOD_LABELS[method].split("\n")[0],
                (d["recall"], d["precision"]),
                textcoords="offset points", xytext=(5, 3), fontsize=7,
            )

        ax.set_xlabel("Recall (%)", fontsize=11)
        ax.set_ylabel("Precision (%)", fontsize=11)
        ax.set_title(f"POPE-{split.capitalize()}", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)

    handles = [
        mpatches.Patch(color=COLORS[m], label=METHOD_LABELS[m].split("\n")[0])
        for m in METHOD_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Precision–Recall Trade-off", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "precision_recall_scatter.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    print("Saved figures/precision_recall_scatter.png")


def print_latex_table(results: list[dict]) -> None:
    """Print LaTeX-formatted results table for the report."""
    splits = ["random", "popular", "adversarial"]
    print("\n% === LaTeX Table ===")
    print(r"\begin{tabular}{llccccc}")
    print(r"\hline")
    print(r"Split & Method & Acc & Prec & Rec & F1 & Yes Ratio \\")
    print(r"\hline")

    for split in splits:
        split_data = [r for r in results if r["split"] == split]
        split_data.sort(key=lambda r: METHOD_ORDER.index(r["method"])
                        if r["method"] in METHOD_ORDER else 99)
        for i, r in enumerate(split_data):
            prefix = r"\multirow{" + str(len(split_data)) + r"}{*}{" + split.capitalize() + "} & " if i == 0 else "& "
            print(
                prefix +
                f"{METHOD_LABELS.get(r['method'], r['method'])} & "
                f"{r['accuracy']:.2f} & {r['precision']:.2f} & "
                f"{r['recall']:.2f} & {r['f1']:.2f} & {r['yes_ratio']:.2f} \\\\"
            )
        print(r"\hline")

    print(r"\end{tabular}")


def generate_all(results_path: str = "results/all_results.json", output_dir: str = "figures") -> None:
    results = load_results(results_path)
    f1_bar_chart(results, output_dir)
    precision_recall_scatter(results, output_dir)
    print_latex_table(results)
