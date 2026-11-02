"""Bar chart comparing per-variant AUC for the two Chunk 4.2 baseline runs
(aug_level none vs strong on EfficientNet-B0).

Usage:
    python src/plot_robust_baseline.py \
        --eval_json outputs/eval/baseline_none.json outputs/eval/baseline_strong.json \
        --labels "aug: none" "aug: strong" \
        --out outputs/figures/robust_baseline.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_json", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out", default="outputs/figures/robust_baseline.png")
    args = parser.parse_args()
    assert len(args.eval_json) == len(args.labels)

    runs = []
    for path in args.eval_json:
        with open(path, encoding="utf-8") as f:
            runs.append(json.load(f))

    variant_names = [r["variant"] for r in runs[0]["variants"]]
    x = np.arange(len(variant_names))
    width = 0.8 / len(runs)

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (run, label) in enumerate(zip(runs, args.labels)):
        aucs = [v["auc"] for v in run["variants"]]
        ax.bar(x + i * width, aucs, width, label=label)

    ax.set_xticks(x + width * (len(runs) - 1) / 2)
    ax.set_xticklabels(variant_names, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Per-variant AUC, baseline aug_level comparison (EfficientNet-B0)")
    ax.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
