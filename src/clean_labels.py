"""Flags likely-mislabelled training images with confident learning
(Northcutt, Jiang and Chuang, JAIR 2021) and writes a cleaned split.

Input: runs trained on inner folds (e.g. split_v1_h0_v{1..4}.csv). Each run
predicts only its own val rows, so every probability is out-of-fold: the
model never trained on that image. Each image's P(fake) is the sigmoid of the
mean logit over the original and the flipped view.

Rule (binary confident learning): per-class thresholds t_k = mean P(class k)
over images labelled k. An image's confident class is the argmax of P(k)
over the classes with P(k) >= t_k; it is flagged when that class differs
from its label. Conflicting-label groups are left as they are.

Output: a copy of --base_split in which flagged train rows have split
"drop" (train.py ignores them); val rows are never changed. Also writes the
out-of-fold probabilities and a summary next to --out_prefix.

Usage:
    python src/clean_labels.py --runs outputs/runs/v6_s0_cnxt_s_v1 ... \
        --base_split data/splits/split_v1.csv --out_split data/splits/split_v1_clean.csv \
        --out_prefix outputs/ablations/v6_clean
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from augment import build_eval_transform
from ensemble import forward_logits, load_member
from robust_eval import VariantDataset, _clean, load_rows


def oof_probs(run_dir, device, bs, num_workers):
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        split_csv = json.load(f)["split_csv"]
    rows = load_rows(split_csv, split="val")
    model, ckpt = load_member(os.path.join(run_dir, "best.pt"), device)
    size = int(ckpt.get("img_size", 224))
    loader = DataLoader(VariantDataset(rows, _clean, build_eval_transform(size)),
                        batch_size=bs, shuffle=False, num_workers=num_workers)
    logits = []
    with torch.no_grad():
        for batch in loader:
            images = batch[0]
            z = (forward_logits(model, images, device, False)
                 + forward_logits(model, images, device, True)) / 2
            logits.append(z)
    p = 1.0 / (1.0 + np.exp(-np.concatenate(logits)))
    return {r["image_id"]: (float(pi), os.path.basename(run_dir)) for r, pi in zip(rows, p)}


def confident_flags(labels, p_fake):
    probs = np.stack([1 - p_fake, p_fake], axis=1)
    thresholds = np.array([probs[labels == k, k].mean() for k in (0, 1)])
    above = probs >= thresholds
    masked = np.where(above, probs, -1.0)
    confident = masked.argmax(axis=1)
    has_conf = above.any(axis=1)
    flags = has_conf & (confident != labels)
    return flags, thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--base_split", default="data/splits/split_v1.csv")
    parser.add_argument("--out_split", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oof = {}
    for run in args.runs:
        part = oof_probs(run, device, args.bs, args.num_workers)
        overlap = set(part) & set(oof)
        assert not overlap, f"{run}: {len(overlap)} images already predicted by another run"
        oof.update(part)

    with open(args.base_split, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        base = list(reader)
    cand = [r for r in base if r["split"] == "train" and r["is_conflict"] != "1"]
    missing = [r["image_id"] for r in cand if r["image_id"] not in oof]
    assert not missing, f"{len(missing)} train rows have no out-of-fold prediction"

    labels = np.array([int(r["label"]) for r in cand])
    p_fake = np.array([oof[r["image_id"]][0] for r in cand])
    flags, thresholds = confident_flags(labels, p_fake)
    flagged = {r["image_id"] for r, f in zip(cand, flags) if f}

    out_rows = [dict(r, split="drop") if r["image_id"] in flagged else r for r in base]
    os.makedirs(os.path.dirname(args.out_split) or ".", exist_ok=True)
    with open(args.out_split, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    with open(args.out_prefix + "_oof.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "label", "p_fake", "run", "flagged"])
        for r, p, fl in zip(cand, p_fake, flags):
            writer.writerow([r["image_id"], r["label"], f"{p:.6f}", oof[r["image_id"]][1], int(fl)])

    from sklearn.metrics import roc_auc_score
    summary = {
        "runs": [os.path.basename(r) for r in args.runs],
        "base_split": args.base_split,
        "n_candidates": len(cand),
        "oof_auc": float(roc_auc_score(labels, p_fake)),
        "thresholds": {"real": float(thresholds[0]), "fake": float(thresholds[1])},
        "n_flagged": int(flags.sum()),
        "n_flagged_real": int((flags & (labels == 0)).sum()),
        "n_flagged_fake": int((flags & (labels == 1)).sum()),
        "n_train_after": sum(1 for r in out_rows if r["split"] == "train"),
    }
    with open(args.out_prefix + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out_split}")


if __name__ == "__main__":
    main()
