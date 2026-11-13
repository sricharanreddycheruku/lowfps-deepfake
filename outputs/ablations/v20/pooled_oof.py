"""Pooled out-of-fold scores for candidate ensembles (round v20).

Reads cached per-image validation logits of the five fold models of each
member, checks that the folds partition the labelled set, and reports pooled
AUC over all 7,000 images plus a per-fold comparison with the v11-recipe trio.

Run from the project root:
    python outputs/ablations/v20/pooled_oof.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DEFAULT_VIEWS = [
    "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30", "blur_sigma1", "blur_sigma2",
    "noise_std5", "noise_std10", "noise_std20", "downscale_0.5", "downscale_0.25",
    "brightness_plus30", "brightness_minus30", "contrast_0.6", "contrast_1.4",
    "grayscale", "centercrop90", "rotation5", "hflip", "resize160", "resize299",
    "combo_jpeg50_blur1_noise5",
]
EXTENDED_VIEWS = ["snow", "rain", "black_boxes", "block_dropout", "pixelate8",
                  "colour_tint", "jpeg_q20"]
STRESS_VIEWS = ["pixelate5", "pixelate12", "pixelate24", "pixelate8_patch",
                "pixelate8_jpeg50", "downscale_0.125", "jpeg_q10", "blur_sigma4",
                "noise_std40"]

MEMBER_RUNS = {
    "cnxt22k": {k: "cnxt_22k_sbi" for k in range(5)},
    "swin22k": {k: "swin_22k_sbi" for k in range(5)},
    "maxvit": {0: "maxvit_t_consistency", 1: "maxvit_t", 2: "maxvit_t",
               3: "maxvit_t", 4: "maxvit_t_consistency"},
    "cnxt_base": {k: "cnxt_base" for k in range(5)},
    "dino": {k: "dino_sbi" for k in range(5)},
}
SETS = {
    "A": ["cnxt22k", "swin22k", "maxvit", "cnxt_base"],
    "B": ["cnxt22k", "swin22k", "maxvit"],
    "C": ["cnxt22k", "swin22k", "maxvit", "cnxt_base", "dino"],
}
V11_TRIO = ["control", "cnxt_small", "swin_t"]
TEAMMATE_CLEAN = 0.9052
MARGIN = 0.003


def split_path(splits, k):
    return splits / ("split_v1.csv" if k == 0 else f"split_v1_fold{k}.csv")


def load_fold(root, splits, k, runs):
    """Return val labels, group ids and {name: {view: logits}} aligned to the split order."""
    df = pd.read_csv(split_path(splits, k))
    val = df[df["split"] == "val"].reset_index(drop=True)
    out = {}
    for name, run in runs.items():
        d = np.load(root / f"fold{k}" / run / "validation_logits.npz")
        ids = [str(x) for x in d["image_ids"]]
        if ids != val["image_id"].astype(str).tolist():
            pos = {i: n for n, i in enumerate(ids)}
            order = np.array([pos[i] for i in val["image_id"].astype(str)])
        else:
            order = np.arange(len(ids))
        if "targets" in d.files:
            assert np.array_equal(d["targets"][order].astype(int), val["label"].values), (k, run)
        out[name] = {v: d[v][order].astype(np.float64) for v in d.files
                     if v not in ("image_ids", "targets")}
    return val, out


def fuse(logits, names, view):
    return np.mean([logits[n][view] for n in names], axis=0)


def fused_views(logits, names):
    views = {v: fuse(logits, names, v) for v in ["clean"] + DEFAULT_VIEWS + EXTENDED_VIEWS + STRESS_VIEWS}
    views["clean_tta"] = 0.5 * (views["clean"] + views["hflip"])
    return views


def summary(y, views):
    auc = {v: roc_auc_score(y, s) for v, s in views.items()}
    default = [auc[v] for v in DEFAULT_VIEWS]
    return {
        "clean_tta": auc["clean_tta"],
        "clean_no_tta": auc["clean"],
        "accuracy_tta": float(np.mean((views["clean_tta"] > 0) == y)),
        "default_mean": float(np.mean(default)),
        "default_worst": float(np.min(default)),
        "default_worst_view": DEFAULT_VIEWS[int(np.argmin(default))],
        "pixelate8": auc["pixelate8"],
        "extended_mean": float(np.mean([auc[v] for v in EXTENDED_VIEWS])),
        "stress_mean": float(np.mean([auc[v] for v in STRESS_VIEWS])),
        "pixelate24": auc["pixelate24"],
    }


def bootstrap_ci(y, groups, score, draws, seed):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    vals = []
    for _ in range(draws):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        if 0 < y[idx].sum() < len(idx):
            vals.append(roc_auc_score(y[idx], score[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="outputs/ablations/v14")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--out", default="outputs/ablations/v20/pooled_oof.json")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root, splits = Path(args.root), Path(args.splits)

    ys, groups, ids, fold_logits = [], [], [], []
    for k in range(5):
        val, logits = load_fold(root, splits, k, {m: MEMBER_RUNS[m][k] for m in MEMBER_RUNS})
        ys.append(val["label"].values)
        groups.append(val["group_id"].values)
        ids += val["image_id"].astype(str).tolist()
        fold_logits.append(logits)
    full = pd.read_csv(split_path(splits, 0))
    assert len(ids) == len(set(ids)) == len(full) == 7000, "folds must partition the labelled set"
    y, g = np.concatenate(ys), np.concatenate(groups)

    result = {"n_images": len(ids), "members": {}, "sets": {}, "fold_check": {}}
    for m in MEMBER_RUNS:
        views = {}
        for k in range(5):
            fv = fused_views(fold_logits[k], [m])
            for v, s in fv.items():
                views.setdefault(v, []).append(s)
        result["members"][m] = summary(y, {v: np.concatenate(s) for v, s in views.items()})

    for name, members in SETS.items():
        views = {}
        for k in range(5):
            for v, s in fused_views(fold_logits[k], members).items():
                views.setdefault(v, []).append(s)
        views = {v: np.concatenate(s) for v, s in views.items()}
        s = summary(y, views)
        s["clean_tta_ci95"] = bootstrap_ci(y, g, views["clean_tta"], args.draws, args.seed)
        s["pixelate8_ci95"] = bootstrap_ci(y, g, views["pixelate8"], args.draws, args.seed)
        s["selection_score"] = 0.5 * (s["clean_tta"] + s["default_mean"])
        s["per_fold_clean_tta"] = [
            roc_auc_score(ys[k], fused_views(fold_logits[k], members)["clean_tta"]) for k in range(5)]
        result["sets"][name] = s

    for k in (0, 4):
        val, trio = load_fold(root, splits, k, {r: r for r in V11_TRIO})
        yk = val["label"].values
        base = summary(yk, fused_views(trio, V11_TRIO))
        result["fold_check"][f"fold{k}"] = {"v11_trio": base}
        for name, members in SETS.items():
            cand = summary(yk, fused_views(fold_logits[k], members))
            result["fold_check"][f"fold{k}"][name] = cand

    chosen = max(result["sets"], key=lambda n: result["sets"][n]["selection_score"])
    s = result["sets"][chosen]
    checks = {"pooled_clean_vs_teammate": bool(s["clean_tta"] >= TEAMMATE_CLEAN)}
    for k in (0, 4):
        fc = result["fold_check"][f"fold{k}"]
        for metric in ("clean_tta", "pixelate8"):
            checks[f"fold{k}_{metric}_non_inferior"] = bool(fc[chosen][metric] >= fc["v11_trio"][metric] - MARGIN)
    result["decision"] = {"selected": chosen, "checks": checks, "ship": all(checks.values())}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    keys = ["clean_tta", "accuracy_tta", "default_mean", "default_worst", "pixelate8", "extended_mean", "stress_mean"]
    print(f"{'':12}" + "".join(f"{k:>14}" for k in keys))
    for n, s in list(result["members"].items()) + [(f"set {n}", v) for n, v in result["sets"].items()]:
        print(f"{n:12}" + "".join(f"{s[k]:14.5f}" for k in keys))
    for k in (0, 4):
        for n, s in result["fold_check"][f"fold{k}"].items():
            print(f"fold{k} {n:9}" + "".join(f"{s[x]:14.5f}" for x in keys))
    print(json.dumps(result["decision"], indent=1))
    for n, s in result["sets"].items():
        print(n, "clean CI", s["clean_tta_ci95"], "per fold", [round(x, 4) for x in s["per_fold_clean_tta"]])


if __name__ == "__main__":
    main()
