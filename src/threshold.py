"""Picks a decision threshold for label/F1/accuracy reporting.

AUC (the main scored metric) does not depend on the threshold, but the
robustness table's F1/accuracy columns and the final predictions.csv
label column do. This script evaluates three threshold candidates on the
val split:
  - best_f1: the threshold that maximises F1 on clean val
  - youden_j: the threshold that maximises TPR - FPR (Youden's J) on
    clean val, the usual "balanced" choice
  - fixed_0.5: the default BCE decision boundary

Each candidate is then scored for stability by running the full
src/robust_eval.py variant suite at that fixed threshold and looking at
the spread (std) of per-variant F1 across all variants: the candidate
with the lowest spread is picked as the one least likely to break under
distribution shift. Ties are broken by higher mean F1.

Usage:
    python src/threshold.py --ckpt outputs/runs/smoke_efficientnet_b0/best.pt \
        --split_csv data/splits/split_v1.csv --out outputs/eval/threshold.json
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_curve

from robust_eval import VARIANTS, compute_metrics, load_models, load_rows, predict_variant


def best_f1_threshold(targets, probs):
    thresholds = np.unique(probs)
    thresholds = np.clip(thresholds, 0.0, 1.0)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(targets, (probs >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def youden_j_threshold(targets, probs):
    fpr, tpr, thresholds = roc_curve(targets, probs)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", nargs="+", required=True)
    parser.add_argument("--split_csv", default="data/splits/split_v1.csv")
    parser.add_argument("--out", default="outputs/eval/threshold.json")
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows(args.split_csv, split="val")
    models_list, size = load_models(args.ckpt, device)

    clean_targets, clean_probs = predict_variant(
        models_list, rows, VARIANTS["clean"], device, args.bs, args.num_workers, size)

    candidates = {
        "best_f1": best_f1_threshold(clean_targets, clean_probs),
        "youden_j": youden_j_threshold(clean_targets, clean_probs),
        "fixed_0.5": 0.5,
    }

    per_candidate = {}
    for name, t in candidates.items():
        variant_f1s = []
        variant_report = {}
        for variant_name, variant_fn in VARIANTS.items():
            targets, probs = predict_variant(
                models_list, rows, variant_fn, device, args.bs, args.num_workers, size)
            metrics = compute_metrics(targets, probs, t)
            variant_f1s.append(metrics["f1"])
            variant_report[variant_name] = {"f1": metrics["f1"], "accuracy": metrics["accuracy"]}
        per_candidate[name] = {
            "threshold": t,
            "mean_f1": float(np.mean(variant_f1s)),
            "std_f1": float(np.std(variant_f1s)),
            "per_variant": variant_report,
        }
        print(f"{name}: threshold {t:.4f} mean_f1 {np.mean(variant_f1s):.4f} "
              f"std_f1 {np.std(variant_f1s):.4f}")

    # most stable = lowest spread of F1 across variants; break ties with mean F1
    chosen = min(per_candidate.items(), key=lambda kv: (kv[1]["std_f1"], -kv[1]["mean_f1"]))[0]

    result = {
        "candidates": per_candidate,
        "chosen": chosen,
        "chosen_threshold": per_candidate[chosen]["threshold"],
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"chosen threshold policy: {chosen} ({per_candidate[chosen]['threshold']:.4f})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
