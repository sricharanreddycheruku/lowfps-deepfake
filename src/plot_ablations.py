"""Figures for the hyperparameter ablations.

For every group in ablations.csv (learning rate, EMA, label smoothing, ...)
writes a grouped bar chart of clean / mean / worst variant AUC
(ablation_<group>.png) and the clean val AUC curves of its runs
(curves_<group>.png). backbones.csv gets the same two figures plus a
throughput / memory panel (ablation_backbones.png). The chosen setting of a
group is outlined and marked with an asterisk.

Usage:
    python src/plot_ablations.py --ablations outputs/ablations/ablations.csv \
        --backbones outputs/ablations/backbones.csv --out_dir outputs/figures
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = [("clean_auc", "clean"), ("mean_auc", "mean over variants"), ("worst_auc", "worst variant")]
COLORS = ["#2a6f97", "#61a5c2", "#a9d6e5"]


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_curve(runs_dir, run):
    path = os.path.join(runs_dir, run, "log.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        log = list(csv.DictReader(f))
    return [float(r["clean_val_auc"]) for r in log]


def group_rows(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    return groups


def label_of(row):
    return row["setting"] + (" *" if row.get("chosen") == "yes" else "")


def bar_panel(ax, rows, title):
    x = np.arange(len(rows))
    width = 0.8 / len(METRICS)
    for i, (key, name) in enumerate(METRICS):
        vals = [float(r[key]) for r in rows]
        ax.bar(x + (i - 1) * width, vals, width, label=name, color=COLORS[i])
        for xi, v in zip(x + (i - 1) * width, vals):
            ax.text(xi, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)
    for xi, r in zip(x, rows):
        if r.get("chosen") == "yes":
            ax.axvspan(xi - 0.45, xi + 0.45, fill=False, edgecolor="black", linewidth=1.2)
    lows = [float(r["worst_auc"]) for r in rows]
    highs = [float(r["clean_auc"]) for r in rows]
    ax.set_ylim(max(0.5, min(lows) - 0.05), min(1.0, max(highs) + 0.06))
    ax.set_xticks(x)
    ax.set_xticklabels([label_of(r) for r in rows], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("val AUC")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")


def bars(rows, title, path):
    fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(rows) + 2), 4.2))
    bar_panel(ax, rows, title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def curves(rows, runs_dir, title, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    drawn = 0
    for r in rows:
        c = read_curve(runs_dir, r["run"])
        if c is None:
            continue
        ax.plot(np.arange(1, len(c) + 1), c, marker="o", markersize=3,
                linewidth=2 if r.get("chosen") == "yes" else 1.2, label=label_of(r))
        drawn += 1
    if drawn == 0:
        plt.close(fig)
        return
    ax.set_xlabel("epoch")
    ax.set_ylabel("clean val AUC (EMA weights if enabled)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def backbone_figure(rows, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    bar_panel(axes[0], rows, "Backbones: val AUC")
    x = np.arange(len(rows))
    names = [label_of(r) for r in rows]
    for ax, key, title, color in [(axes[1], "img_per_s", "Inference throughput (fp16, batch 128)", "#2a6f97"),
                                  (axes[2], "peak_mb", "Peak GPU memory (MB)", "#61a5c2")]:
        vals = [float(r[key]) for r in rows]
        ax.bar(x, vals, color=color)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablations", default="outputs/ablations/ablations.csv")
    parser.add_argument("--backbones", default="outputs/ablations/backbones.csv")
    parser.add_argument("--runs_dir", default="outputs/runs")
    parser.add_argument("--out_dir", default="outputs/figures")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for group, rows in group_rows(read_rows(args.ablations)).items():
        bars(rows, f"Ablation: {group}", os.path.join(args.out_dir, f"ablation_{group}.png"))
        curves(rows, args.runs_dir, f"Training curves: {group}",
               os.path.join(args.out_dir, f"curves_{group}.png"))

    if os.path.exists(args.backbones):
        rows = read_rows(args.backbones)
        backbone_figure(rows, os.path.join(args.out_dir, "ablation_backbones.png"))
        curves(rows, args.runs_dir, "Training curves: backbones",
               os.path.join(args.out_dir, "curves_backbones.png"))
    print(f"figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
