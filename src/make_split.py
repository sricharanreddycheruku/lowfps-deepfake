"""Build a leak-free train/val split.

Uses StratifiedGroupKFold (5 folds, fold 0 = val) so near-duplicate images
(same group_id from find_duplicates.py) never straddle train and val, and
class balance is preserved in both splits. Groups with conflicting labels
(a real and a fake image judged near-identical) are kept in train with a
soft label of 0.5, since the plan's default is to keep rather than drop
them; the choice is ablated later in Chunk 5.

Usage:
    python src/make_split.py --dup_csv outputs/data/dup_groups.csv \
        --data_root data/raw/hackathon_dataset --out data/splits/split_v1.csv \
        --seed 42 --val_frac 0.2
"""

import argparse
import csv
import os

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def load_groups(dup_csv):
    rows = []
    with open(dup_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "image_id": row["image_id"],
                "label": int(row["label"]),
                "group_id": int(row["group_id"]),
            })
    return rows


def resolve_conflicts(rows):
    """Groups with more than one distinct label get a soft label of 0.5,
    recorded in a separate column; the hard label column keeps the
    original per-image value so the split csv stays self-describing."""
    group_labels = {}
    for r in rows:
        group_labels.setdefault(r["group_id"], set()).add(r["label"])
    conflicting = {g for g, labels in group_labels.items() if len(labels) > 1}
    for r in rows:
        r["soft_label"] = 0.5 if r["group_id"] in conflicting else float(r["label"])
        r["is_conflict"] = r["group_id"] in conflicting
    return rows, conflicting


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dup_csv", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.2)
    args = parser.parse_args()

    rows = load_groups(args.dup_csv)
    rows, conflicting_groups = resolve_conflicts(rows)

    n_folds = max(2, round(1 / args.val_frac))

    image_ids = np.array([r["image_id"] for r in rows])
    labels = np.array([r["label"] for r in rows])
    groups = np.array([r["group_id"] for r in rows])

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
    fold_of_index = np.zeros(len(rows), dtype=int)
    for fold_idx, (_train_idx, val_idx) in enumerate(sgkf.split(image_ids, labels, groups)):
        fold_of_index[val_idx] = fold_idx

    val_fold = 0
    split_of_index = np.where(fold_of_index == val_fold, "val", "train")

    for r, split in zip(rows, split_of_index):
        r["split"] = split
        folder = "real" if r["label"] == 0 else "fake"
        # forward slashes always, so the split csv is portable between
        # the local machine and the Linux training host
        root = args.data_root.rstrip("/\\").replace("\\", "/")
        r["path"] = f"{root}/train/{folder}/{r['image_id']}.jpg"

    # leak check: no group_id shared between train and val
    train_groups = {r["group_id"] for r in rows if r["split"] == "train"}
    val_groups = {r["group_id"] for r in rows if r["split"] == "val"}
    overlap = train_groups & val_groups
    assert not overlap, f"group leakage between train and val: {sorted(overlap)[:10]}"

    n_total = len(rows)
    n_val = sum(1 for r in rows if r["split"] == "val")
    val_frac_actual = n_val / n_total

    def class_ratio(split_name):
        subset = [r for r in rows if r["split"] == split_name]
        fake = sum(1 for r in subset if r["label"] == 1)
        return fake / len(subset) if subset else 0.0

    train_ratio = class_ratio("train")
    val_ratio = class_ratio("val")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "path", "label", "soft_label", "group_id", "is_conflict", "split"])
        for r in rows:
            writer.writerow([r["image_id"], r["path"], r["label"], r["soft_label"],
                              r["group_id"], int(r["is_conflict"]), r["split"]])

    summary = {
        "n_total": n_total,
        "n_train": n_total - n_val,
        "n_val": n_val,
        "val_frac_actual": round(val_frac_actual, 4),
        "train_fake_ratio": round(train_ratio, 4),
        "val_fake_ratio": round(val_ratio, 4),
        "class_ratio_diff_pct_points": round(abs(train_ratio - val_ratio) * 100, 3),
        "conflicting_groups_total": len(conflicting_groups),
        "group_leakage": len(overlap),
    }
    print(summary)

    assert abs(val_frac_actual - args.val_frac) <= 0.01, "val fraction out of tolerance"
    assert abs(train_ratio - val_ratio) <= 0.01, "class ratio mismatch between train and val"


if __name__ == "__main__":
    main()
