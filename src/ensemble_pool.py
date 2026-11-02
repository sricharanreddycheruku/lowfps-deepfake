"""Scores fixed member sets against a reference ensemble, with a paired bootstrap.

Members are the runs whose per-variant logits are cached by src/ensemble.py
(outputs/cache/<run>.npz, built here from best.pt when missing). A set is
fused with equal weights, so nothing is tuned on val except the single
choice between mean-logit and mean-rank, made once on the largest set and
then applied to every set.

For each set the script reports clean / mean / worst-of-suite AUC and, against
the reference set, the paired bootstrap (resampling val images, same indices
for both systems) of the clean-AUC gain and of the worst-variant-AUC gain.

Usage:
    python src/ensemble_pool.py --sets outputs/ablations/ens_v4_sets.json \
        --ref shipped --out outputs/ablations/ens_v4
    sets file: {"shipped": ["final_resnet50", "e_cnxt_small", "final_swin_t"], ...}
"""

import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from ensemble import fuse, member_cache, member_logits, summarise
from robust_eval import VARIANTS, load_rows


def set_scores(caches, members, tta, rule):
    """Fused score per variant, shape (n_variants, N)."""
    out = {}
    for variant in VARIANTS:
        z = np.stack([member_logits(caches[m], variant, tta) for m in members])
        out[variant] = fuse(z, rule)
    return out


def fast_auc(y, s):
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def paired_bootstrap(y, a, b, reps, seed):
    """a, b: dict variant -> score. Returns gain samples for clean and worst."""
    rng = np.random.RandomState(seed)
    n = len(y)
    clean, worst = [], []
    for _ in range(reps):
        idx = rng.randint(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        aa = [fast_auc(yy, a[v][idx]) for v in VARIANTS]
        bb = [fast_auc(yy, b[v][idx]) for v in VARIANTS]
        ci = list(VARIANTS).index("clean")
        clean.append(aa[ci] - bb[ci])
        worst.append(min(aa) - min(bb))
    return np.array(clean), np.array(worst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", required=True)
    parser.add_argument("--ref", default="shipped")
    parser.add_argument("--runs_dir", default="outputs/runs")
    parser.add_argument("--cache_dir", default="outputs/cache")
    parser.add_argument("--split_csv", default="data/splits/split_v1.csv")
    parser.add_argument("--out", default="outputs/ablations/ens_v4")
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.sets, encoding="utf-8") as f:
        sets = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows(args.split_csv, split="val")
    y = np.array([int(r["label"]) for r in rows])

    names = sorted({m for ms in sets.values() for m in ms})
    caches = {}
    for n in names:
        ckpt = os.path.join(args.runs_dir, n, "best.pt")
        caches[n] = member_cache(n, ckpt, rows, device, args.bs, args.num_workers, args.cache_dir)

    largest = max(sets, key=lambda k: len(sets[k]))
    rule_mean = {r: summarise({v: roc_auc_score(y, s) for v, s in set_scores(
        caches, sets[largest], True, r).items()})["mean_auc"] for r in ("mean_logit", "rank_avg")}
    rule = max(rule_mean, key=rule_mean.get)
    print("rule choice on", largest, rule_mean, "->", rule)

    scores = {k: set_scores(caches, ms, True, rule) for k, ms in sets.items()}
    report = {"rule": rule, "rule_choice_set": largest, "rule_mean_auc": rule_mean, "sets": {}}
    ref = scores[args.ref]
    ref_sum = summarise({v: roc_auc_score(y, s) for v, s in ref.items()})
    for k, sc in scores.items():
        s = summarise({v: roc_auc_score(y, x) for v, x in sc.items()})
        entry = {"members": sets[k], **{a: round(b, 4) for a, b in s.items()}}
        if k != args.ref:
            c, w = paired_bootstrap(y, sc, ref, args.reps, args.seed)
            entry.update({
                "gain_clean": round(s["clean_auc"] - ref_sum["clean_auc"], 4),
                "gain_worst": round(s["worst_auc"] - ref_sum["worst_auc"], 4),
                "clean_ci95": [round(float(np.percentile(c, 2.5)), 4), round(float(np.percentile(c, 97.5)), 4)],
                "worst_ci95": [round(float(np.percentile(w, 2.5)), 4), round(float(np.percentile(w, 97.5)), 4)],
            })
        report["sets"][k] = entry
        print(k, json.dumps(entry))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(f"{args.out}_pool.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
