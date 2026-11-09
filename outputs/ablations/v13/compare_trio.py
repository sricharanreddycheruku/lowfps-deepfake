"""Compare a fixed ResNet-50 substitution in the archived v8 trio."""

import csv
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, "src")
from ensemble_pool import fast_auc
from robust_eval import VARIANTS

root = Path("outputs/ablations/v13")
historical = Path(sys.argv[1])
with Path("data/splits/split_v1.csv").open(newline="", encoding="utf-8") as f:
    rows = [row for row in csv.DictReader(f) if row["split"] == "val"]
ids = np.array([r["image_id"] for r in rows])
labels = np.array([int(r["label"]) for r in rows])
groups = {}
for i, r in enumerate(rows):
    groups.setdefault(r["group_id"], []).append(i)

def read_old(arch):
    with np.load(historical / f"v8_p50_s42_{arch}.npz") as data:
        return np.stack([data[name].mean(axis=0) for name in VARIANTS])

def read_new(arm):
    with np.load(root / arm / "validation_logits.npz") as data:
        np.testing.assert_array_equal(data["image_ids"], ids)
        return np.stack([data[name] for name in VARIANTS])

common = read_old("cnxt_small") + read_old("swin_t")
all_scores = {"archived_v8": (common + read_old("resnet50")) / 3}
for arm in ("bce", "gce", "relabel"):
    all_scores[arm] = (common + read_new(arm)) / 3

def metrics(scores, idx):
    auc = [fast_auc(labels[idx], scores[i, idx]) for i in range(len(VARIANTS))]
    return np.array([auc[list(VARIANTS).index("clean")], np.mean(auc), min(auc)])

keys = ("clean", "mean_23", "worst_23")
full = np.arange(len(labels))
observed = {arm: metrics(scores, full) for arm, scores in all_scores.items()}
rng = np.random.default_rng(0)
group_indices = list(groups.values())
boots = {arm: [] for arm in ("gce", "relabel")}
for _ in range(1000):
    selected = rng.integers(len(groups), size=len(groups))
    idx = np.concatenate([group_indices[i] for i in selected])
    if np.unique(labels[idx]).size != 2:
        continue
    reference = metrics(all_scores["bce"], idx)
    for arm in boots:
        boots[arm].append(metrics(all_scores[arm], idx) - reference)

result = {
    "reference": "archived seed-42 ConvNeXt-Small and Swin-T fixed across arms; swap only ResNet-50",
    "auc": {k: dict(zip(keys, v.tolist())) for k, v in observed.items()},
    "gain_vs_matched_bce": {k: dict(zip(keys, (observed[k] - observed["bce"]).tolist()))
                            for k in boots},
    "gain_95pct_ci": {k: dict(zip(keys, np.percentile(v, [2.5, 97.5], axis=0).T.tolist()))
                      for k, v in boots.items()},
}
(root / "comparison_trio.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
