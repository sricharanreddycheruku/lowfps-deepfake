"""Scores fixed member sets against a reference ensemble, with a paired bootstrap.

Members are the runs whose per-variant logits are cached by src/ensemble.py
(outputs/cache/<run>.npz, built here from best.pt when missing; the centre-crop
view is cached next to it as <run>_crop90.npz). A set is fused with the fixed
weights written in the sets file (equal when absent), so nothing is tuned on
val. The fusion rule is fixed by --fusion; "auto" keeps the v4 behaviour of
choosing between mean-logit and mean-rank once on the largest set.

Views per member: orig, flip, crop90 (centre crop 90% then resize back to
224, applied after each robustness variant). The view logits are averaged
before fusion. --views sets the default, a set can override it.

For each set the script reports clean / mean / worst-of-suite AUC, a T4 time
estimate for 3,000 images (img/s from outputs/ablations/bench*.json, times
views, times 3.5) and, against the reference set, the paired bootstrap
(resampling val images, same indices for both systems) of the clean-AUC gain
and of the worst-variant-AUC gain. Per-variant AUCs of every set are written
to <out>_per_variant.csv and, one file per set, <out>_<set>_per_variant.csv.

Usage:
    python src/ensemble_pool.py --sets outputs/ablations/ens_v5_sets.json \
        --ref shipped --out outputs/ablations/ens_v5
    sets file: {"shipped": ["final_resnet50", "e_cnxt_small", "final_swin_t"],
                "plus_x": {"members": [...], "weights": [...], "views": ["orig", "flip"]}, ...}
"""

import argparse
import csv
import glob
import json
import os

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from augment import build_eval_transform
from ensemble import forward_logits, fuse, load_member, member_cache, summarise
from robust_eval import VARIANTS, VariantDataset, _centercrop90, load_rows

ALL_VIEWS = ("orig", "flip", "crop90")
T4_FACTOR = 3.5
N_TEST = 3000


def crop90_cache(name, path, rows, device, bs, num_workers, cache_dir):
    """Per-variant logits of one member on the centre-crop view, shape (N,)."""
    cache_path = os.path.join(cache_dir, f"{name}_crop90.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        if all(v in data.files for v in VARIANTS) and data["clean"].shape[0] == len(rows):
            return {v: data[v] for v in VARIANTS}
    model, ckpt = load_member(path, device)
    size = int(ckpt.get("img_size", 224))
    out = {}
    for variant, fn in VARIANTS.items():
        def view_fn(img, idx, fn=fn):
            return _centercrop90(fn(img, idx), idx)
        ds = VariantDataset(rows, view_fn, build_eval_transform(size))
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers)
        out[variant] = np.concatenate([forward_logits(model, images, device, flip=False)
                                       for images, _ in loader])
        print(f"{name} crop90 {variant} done")
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache_path, **out)
    return out


def view_logits(cache, crop, variant, views):
    """Member logits of one variant, averaged over the chosen views."""
    z = []
    if "orig" in views:
        z.append(cache[variant][0])
    if "flip" in views:
        z.append(cache[variant][1])
    if "crop90" in views:
        z.append(crop[variant])
    return np.mean(z, axis=0)


def parse_sets(raw, default_views):
    """Sets file entry -> {"members", "weights" or None, "views"}."""
    sets = {}
    for key, val in raw.items():
        spec = {"members": val} if isinstance(val, list) else dict(val)
        spec.setdefault("weights", None)
        spec["views"] = list(spec.get("views") or default_views)
        bad = [v for v in spec["views"] if v not in ALL_VIEWS]
        if bad:
            raise SystemExit(f"set {key}: unknown views {bad}")
        if spec["weights"] is not None and len(spec["weights"]) != len(spec["members"]):
            raise SystemExit(f"set {key}: weights and members differ in length")
        sets[key] = spec
    return sets


def set_scores(caches, crops, spec, rule):
    """Fused score per variant (dict variant -> (N,))."""
    weights = spec["weights"] if rule == "mean_logit" else None
    out = {}
    for variant in VARIANTS:
        z = np.stack([view_logits(caches[m], crops.get(m), variant, spec["views"])
                      for m in spec["members"]])
        if weights is not None:
            out[variant] = fuse(z, "weighted_logit", weights)
        else:
            out[variant] = fuse(z, rule)
    return out


def load_bench(pattern):
    """arch -> img/s at 224 input, later files (sorted by name) win."""
    speed = {}
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for r in json.load(f).get("results", []):
                if int(r.get("size", 224)) == 224:
                    speed[r["arch"]] = float(r["img_per_s"])
    return speed


def member_arch(runs_dir, name):
    cfg = os.path.join(runs_dir, name, "config.json")
    if os.path.exists(cfg):
        with open(cfg, encoding="utf-8") as f:
            return json.load(f).get("arch")
    return None


def t4_estimate(spec, archs, speed):
    """Seconds for N_TEST images on a T4: forward passes / bench img/s x 3.5.
    None when a member has no benchmark row."""
    total = 0.0
    for m in spec["members"]:
        ips = speed.get(archs.get(m))
        if ips is None:
            return None
        total += len(spec["views"]) * N_TEST / ips
    return round(total * T4_FACTOR, 1)


def fast_auc(y, s):
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bootstrap_indices(n, reps, seed):
    rng = np.random.RandomState(seed)
    return [rng.randint(0, n, n) for _ in range(reps)]


def suite_boot(y, scores, draws):
    """(clean, worst) AUC per bootstrap draw for one system; None for a draw
    that holds a single class."""
    ci = list(VARIANTS).index("clean")
    out = []
    for idx in draws:
        yy = y[idx]
        if yy.min() == yy.max():
            out.append(None)
            continue
        aa = [fast_auc(yy, scores[v][idx]) for v in VARIANTS]
        out.append((aa[ci], min(aa)))
    return out


def paired_gain(a_boot, b_boot):
    """Gain samples (clean, worst) of system a over b on the same draws."""
    pairs = [(a, b) for a, b in zip(a_boot, b_boot) if a is not None and b is not None]
    clean = np.array([a[0] - b[0] for a, b in pairs])
    worst = np.array([a[1] - b[1] for a, b in pairs])
    return clean, worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", required=True)
    parser.add_argument("--ref", default="shipped")
    parser.add_argument("--fusion", default="mean_logit", choices=["mean_logit", "rank_avg", "auto"])
    parser.add_argument("--views", default="orig,flip",
                        help="comma list of orig, flip, crop90 (default for sets without their own)")
    parser.add_argument("--runs_dir", default="outputs/runs")
    parser.add_argument("--cache_dir", default="outputs/cache")
    parser.add_argument("--split_csv", default="data/splits/split_v1.csv")
    parser.add_argument("--bench", default="outputs/ablations/bench*.json")
    parser.add_argument("--out", default="outputs/ablations/ens_v5")
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.sets, encoding="utf-8") as f:
        sets = parse_sets(json.load(f), args.views.split(","))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows(args.split_csv, split="val")
    y = np.array([int(r["label"]) for r in rows])

    names = sorted({m for sp in sets.values() for m in sp["members"]})
    crop_names = sorted({m for sp in sets.values() if "crop90" in sp["views"] for m in sp["members"]})
    caches, crops = {}, {}
    for n in names:
        ckpt = os.path.join(args.runs_dir, n, "best.pt")
        caches[n] = member_cache(n, ckpt, rows, device, args.bs, args.num_workers, args.cache_dir)
    for n in crop_names:
        ckpt = os.path.join(args.runs_dir, n, "best.pt")
        crops[n] = crop90_cache(n, ckpt, rows, device, args.bs, args.num_workers, args.cache_dir)

    rule_mean, rule_set = None, None
    if args.fusion == "auto":
        rule_set = max(sets, key=lambda k: len(sets[k]["members"]))
        rule_mean = {r: summarise({v: roc_auc_score(y, s) for v, s in set_scores(
            caches, crops, {**sets[rule_set], "weights": None}, r).items()})["mean_auc"]
            for r in ("mean_logit", "rank_avg")}
        rule = max(rule_mean, key=rule_mean.get)
        print("rule choice on", rule_set, rule_mean, "->", rule)
    else:
        rule = args.fusion

    speed = load_bench(args.bench)
    archs = {n: member_arch(args.runs_dir, n) for n in names}
    scores = {k: set_scores(caches, crops, sp, rule) for k, sp in sets.items()}
    per_variant = {k: {v: roc_auc_score(y, x) for v, x in sc.items()} for k, sc in scores.items()}

    draws = bootstrap_indices(len(y), args.reps, args.seed)
    ref_boot = suite_boot(y, scores[args.ref], draws)
    ref_sum = summarise(per_variant[args.ref])
    ref_t4 = t4_estimate(sets[args.ref], archs, speed)
    report = {"rule": rule, "fusion_arg": args.fusion, "views_default": args.views.split(","),
              "rule_choice_set": rule_set, "rule_mean_auc": rule_mean, "reps": args.reps,
              "seed": args.seed, "sets": {}}
    for k, sc in scores.items():
        s = summarise(per_variant[k])
        sp = sets[k]
        t4 = t4_estimate(sp, archs, speed)
        entry = {"members": sp["members"], "weights": sp["weights"], "views": sp["views"],
                 **{a: round(b, 4) for a, b in s.items()}, "t4_est_s": t4}
        if k != args.ref:
            c, w = paired_gain(suite_boot(y, sc, draws), ref_boot)
            entry.update({
                "gain_clean": round(s["clean_auc"] - ref_sum["clean_auc"], 4),
                "gain_worst": round(s["worst_auc"] - ref_sum["worst_auc"], 4),
                "clean_ci95": [round(float(np.percentile(c, 2.5)), 4), round(float(np.percentile(c, 97.5)), 4)],
                "worst_ci95": [round(float(np.percentile(w, 2.5)), 4), round(float(np.percentile(w, 97.5)), 4)],
                "t4_est_ratio_vs_ref": round(t4 / ref_t4, 2) if t4 and ref_t4 else None,
            })
        report["sets"][k] = entry
        print(k, json.dumps(entry))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(f"{args.out}_pool.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(f"{args.out}_per_variant.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant"] + list(sets))
        for v in VARIANTS:
            w.writerow([v] + [f"{per_variant[k][v]:.6f}" for k in sets])
    for k in sets:
        with open(f"{args.out}_{k}_per_variant.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["variant", "auc"])
            for v in VARIANTS:
                w.writerow([v, f"{per_variant[k][v]:.6f}"])


if __name__ == "__main__":
    main()
