"""Linear stacking of member logits (experiment E8 of the v5 round).

Features are the per-member logits (mean of the orig and flip views),
z-scored with constants from the fitting folds only. A 5-fold
StratifiedGroupKFold over val group_id gives out-of-fold predictions: each fold's
L2 logistic regression is fit on the clean features of the other folds and
applied to all robustness variants of its own fold. The OOF scores are compared
with the shipped mean_logit fusion by the paired bootstrap used elsewhere.

--final also fits one model on all of val and writes its coefficients, which is
what a bundle would carry if the stack passes the decision rule.

  python src/stack.py --members final_resnet50 e_cnxt_small final_swin_t <extra...> \
      --out outputs/ablations/stack_v5.json
"""

import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from ensemble import member_cache, summarise
from ensemble_pool import (bootstrap_indices, load_bench, member_arch,
                           paired_gain, set_scores, suite_boot, t4_estimate)
from robust_eval import VARIANTS, load_rows

SHIPPED = ["final_resnet50", "e_cnxt_small", "final_swin_t"]


def features(caches, members, variant):
    z = [np.mean([caches[m][variant][0], caches[m][variant][1]], axis=0) for m in members]
    return np.stack(z, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", nargs="+", default=SHIPPED)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs_dir", default="outputs/runs")
    parser.add_argument("--cache_dir", default="outputs/cache")
    parser.add_argument("--split_csv", default="data/splits/split_v1.csv")
    parser.add_argument("--bench", default="outputs/ablations/bench*.json")
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--boot_seed", type=int, default=0)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--out", default="outputs/ablations/stack_v5.json")
    args = parser.parse_args()

    rows = load_rows(args.split_csv, split="val")
    y = np.array([int(r["label"]) for r in rows])
    groups = np.array([int(r["group_id"]) for r in rows])
    caches = {m: member_cache(m, None, rows, None, 0, 0, args.cache_dir) for m in args.members}
    feats = {v: features(caches, args.members, v) for v in VARIANTS}

    oof = {v: np.zeros(len(y)) for v in VARIANTS}
    folds = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for tr, te in folds.split(y, y, groups):
        x_tr = feats["clean"][tr]
        mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
        clf = LogisticRegression(C=args.c, max_iter=1000).fit((x_tr - mu) / sd, y[tr])
        for v in VARIANTS:
            oof[v][te] = clf.decision_function((feats[v][te] - mu) / sd)

    per_variant = {v: roc_auc_score(y, s) for v, s in oof.items()}
    summ = summarise(per_variant)

    ship_scores = set_scores(caches, {}, {"members": SHIPPED, "weights": None,
                                          "views": ["orig", "flip"]}, "mean_logit")
    ship_pv = {v: roc_auc_score(y, s) for v, s in ship_scores.items()}
    ship_sum = summarise(ship_pv)
    draws = bootstrap_indices(len(y), args.reps, args.boot_seed)
    c, w = paired_gain(suite_boot(y, oof, draws), suite_boot(y, ship_scores, draws))

    speed = load_bench(args.bench)
    archs = {m: member_arch(args.runs_dir, m) for m in args.members}
    t4 = t4_estimate({"members": args.members, "views": ["orig", "flip"]}, archs, speed)

    report = {"members": args.members, "C": args.c, "folds": args.folds, "seed": args.seed,
              "oof": {k: round(v, 4) for k, v in summ.items()},
              "shipped": {k: round(v, 4) for k, v in ship_sum.items()},
              "gain_clean": round(summ["clean_auc"] - ship_sum["clean_auc"], 4),
              "gain_worst": round(summ["worst_auc"] - ship_sum["worst_auc"], 4),
              "clean_ci95": [round(float(np.percentile(c, 2.5)), 4), round(float(np.percentile(c, 97.5)), 4)],
              "worst_ci95": [round(float(np.percentile(w, 2.5)), 4), round(float(np.percentile(w, 97.5)), 4)],
              "t4_est_s": t4, "per_variant_oof": {k: round(v, 4) for k, v in per_variant.items()}}

    if args.final:
        mu, sd = feats["clean"].mean(0), feats["clean"].std(0) + 1e-6
        clf = LogisticRegression(C=args.c, max_iter=1000).fit((feats["clean"] - mu) / sd, y)
        report["final_fit"] = {"mean": mu.tolist(), "std": sd.tolist(),
                               "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0])}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "per_variant_oof"}, indent=1))


if __name__ == "__main__":
    main()
