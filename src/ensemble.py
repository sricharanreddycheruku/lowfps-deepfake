"""Compares fusion rules and flip TTA for a set of trained members.

Every member is run once over the whole robustness suite (the 23 variants
of src/robust_eval.py on the val split), with and without a horizontal
flip, and the per-image logits are cached per member. All fusion rules are
then evaluated from the cache, so adding a member or a rule is cheap.

Fusion rules (z = per-member logit, after flip TTA if enabled):
    mean_logit      sigmoid(mean z)                     usable at inference
    mean_prob       mean sigmoid(z)                     usable at inference
    weighted_logit  sigmoid(sum w z), w from a grid     usable at inference
    rank_avg        mean of per-batch ranks             reference only, needs
                                                        the whole test set

The weight grid has step 0.1 and sums to 1; equal weights are kept unless
the weighted fit beats them by more than --min_gain on mean-variant AUC
(avoids overfitting the val split); mean_logit is kept over mean_prob unless
mean_prob wins by the same margin. Flip TTA is kept when it lowers neither
the mean nor the worst variant AUC. The script also scores every subset of
the given members with equal-weight mean_logit, so the member choice is
backed by numbers, and picks a decision threshold for the chosen setup.

Outputs (prefix given by --out):
    <out>.csv                  one row per single model / subset / rule / TTA
    <out>_chosen.json          chosen members, rule, weights, tta, threshold
    <out>_val_preds.csv        fused clean-val probabilities of the chosen setup

Usage:
    python src/ensemble.py --ckpt outputs/runs/final_resnet50/best.pt \
        outputs/runs/final_convnext_tiny/best.pt outputs/runs/final_swin_t/best.pt \
        --out outputs/ablations/ensemble
"""

import argparse
import csv
import itertools
import json
import os

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

from augment import build_eval_transform
from models import build_model
from robust_eval import VARIANTS, VariantDataset, load_rows


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@torch.inference_mode()
def forward_logits(model, images, device, flip):
    """Logits for one batch; the same fp16-on-CUDA path the inference script uses."""
    images = images.to(device, memory_format=torch.channels_last)
    if flip:
        images = torch.flip(images, dims=[3])
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        out = model(images)
    return out.squeeze(1).float().cpu().numpy()


def load_member(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device, memory_format=torch.channels_last).eval()
    return model, ckpt


def member_cache(name, path, rows, device, bs, num_workers, cache_dir):
    """Per-variant logits of one member, shape (2, N): plain and flipped."""
    cache_path = os.path.join(cache_dir, f"{name}.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        if all(v in data.files for v in VARIANTS) and data["clean"].shape[1] == len(rows):
            return {v: data[v] for v in VARIANTS}
    model, ckpt = load_member(path, device)
    size = int(ckpt.get("img_size", 224))
    out = {}
    for variant, fn in VARIANTS.items():
        ds = VariantDataset(rows, fn, build_eval_transform(size))
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers)
        plain, flipped = [], []
        for images, _ in loader:
            plain.append(forward_logits(model, images, device, flip=False))
            flipped.append(forward_logits(model, images, device, flip=True))
        out[variant] = np.stack([np.concatenate(plain), np.concatenate(flipped)])
        print(f"{name} {variant} done")
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache_path, **out)
    return out


def member_logits(cache, variant, tta):
    z = cache[variant]
    return z.mean(axis=0) if tta else z[0]


def fuse(logits, rule, weights=None):
    """logits: (M, N). Returns the fused score in [0, 1] (rank_avg is rescaled)."""
    if rule == "mean_logit":
        return sigmoid(logits.mean(axis=0))
    if rule == "mean_prob":
        return sigmoid(logits).mean(axis=0)
    if rule == "weighted_logit":
        return sigmoid((np.asarray(weights)[:, None] * logits).sum(axis=0))
    if rule == "rank_avg":
        ranks = np.stack([rankdata(z) for z in logits])
        return ranks.mean(axis=0) / logits.shape[1]
    raise ValueError(f"unknown fusion rule: {rule}")


def suite_scores(caches, targets, tta, rule, weights=None):
    """AUC per variant of a fused set of members."""
    aucs = {}
    for variant in VARIANTS:
        z = np.stack([member_logits(c, variant, tta) for c in caches])
        aucs[variant] = roc_auc_score(targets, fuse(z, rule, weights))
    return aucs


def summarise(aucs):
    vals = list(aucs.values())
    return {"clean_auc": aucs["clean"], "mean_auc": float(np.mean(vals)),
            "worst_auc": float(np.min(vals)), "std_auc": float(np.std(vals))}


def weight_grid(n, step=0.1):
    ticks = int(round(1 / step))
    for combo in itertools.product(range(ticks + 1), repeat=n):
        if sum(combo) == ticks:
            yield tuple(c * step for c in combo)


def pick_threshold(targets, caches, tta, rule, weights):
    """Youden-J, best-F1 and 0.5 on the clean fused scores, scored by how
    stable F1 stays across the variant suite (lowest std wins)."""
    from sklearn.metrics import roc_curve

    z = np.stack([member_logits(c, "clean", tta) for c in caches])
    p = fuse(z, rule, weights)
    fpr, tpr, thr = roc_curve(targets, p)
    youden = float(thr[np.argmax(tpr - fpr)])
    grid = np.unique(np.round(p, 4))
    f1s = [f1_score(targets, (p >= t).astype(int)) for t in grid]
    best_f1 = float(grid[int(np.argmax(f1s))])
    candidates = {"youden_j": youden, "best_f1": best_f1, "fixed_0.5": 0.5}
    report = {}
    for name, t in candidates.items():
        per_variant = []
        for variant in VARIANTS:
            zv = np.stack([member_logits(c, variant, tta) for c in caches])
            per_variant.append(f1_score(targets, (fuse(zv, rule, weights) >= t).astype(int)))
        report[name] = {"threshold": t, "mean_f1": float(np.mean(per_variant)),
                        "std_f1": float(np.std(per_variant))}
    chosen = min(report, key=lambda k: report[k]["std_f1"])
    return chosen, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", nargs="+", required=True)
    parser.add_argument("--split_csv", default="data/splits/split_v1.csv")
    parser.add_argument("--out", default="outputs/ablations/ensemble")
    parser.add_argument("--cache_dir", default="outputs/cache")
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--min_gain", type=float, default=0.002,
                        help="weighted fusion must beat equal weights by this much on mean AUC")
    parser.add_argument("--max_members", type=int, default=4,
                        help="largest member subset scored in the subset table")
    parser.add_argument("--members", nargs="+", default=None,
                        help="names (run directory names) of the members to ship; "
                             "default is the best subset by mean-variant AUC")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows(args.split_csv, split="val")
    targets = np.array([int(r["label"]) for r in rows])

    names = [os.path.basename(os.path.dirname(os.path.abspath(p))) for p in args.ckpt]
    paths = dict(zip(names, args.ckpt))
    caches = {n: member_cache(n, paths[n], rows, device, args.bs, args.num_workers, args.cache_dir)
              for n in names}

    table = []

    def add(kind, label, members, tta, rule, weights=None):
        s = summarise(suite_scores([caches[m] for m in members], targets, tta, rule, weights))
        table.append({"kind": kind, "name": label, "members": "+".join(members),
                      "rule": rule, "tta": int(tta), "weights": "" if weights is None else
                      "/".join(f"{w:.1f}" for w in weights), **s})
        return s

    for n in names:
        for tta in (False, True):
            add("single", n, [n], tta, "mean_logit")

    subset_mean = {}
    for k in range(2, min(len(names), args.max_members) + 1):
        for sub in itertools.combinations(names, k):
            for tta in (False, True):
                s = add("subset", "+".join(sub), list(sub), tta, "mean_logit")
                subset_mean[(sub, tta)] = s["mean_auc"]

    if args.members:
        members = args.members
    else:
        best = max((k for k in subset_mean if k[1]), key=lambda k: subset_mean[k])
        members = list(best[0])

    rules = {}
    for tta in (False, True):
        for rule in ("mean_logit", "mean_prob", "rank_avg"):
            rules[(rule, tta)] = add("rule", "ensemble", members, tta, rule)["mean_auc"]

    best_w, best_w_auc = None, -1.0
    for w in weight_grid(len(members)):
        s = summarise(suite_scores([caches[m] for m in members], targets, True,
                                   "weighted_logit", w))
        if s["mean_auc"] > best_w_auc:
            best_w, best_w_auc = w, s["mean_auc"]
    add("rule", "ensemble", members, True, "weighted_logit", best_w)

    equal_auc = rules[("mean_logit", True)]
    use_weights = best_w_auc - equal_auc > args.min_gain

    # implementable rules only (rank_avg is a reference)
    fusion = "weighted_logit" if use_weights else (
        "mean_prob" if rules[("mean_prob", True)] - rules[("mean_logit", True)] > args.min_gain
        else "mean_logit")
    weights = list(best_w) if use_weights else [1.0 / len(members)] * len(members)
    # flip TTA costs one extra forward pass per member, which the time budget
    # easily allows, so it is kept whenever it hurts neither mean nor worst AUC
    if fusion == "weighted_logit":
        tta_on = True
    else:
        off = summarise(suite_scores([caches[m] for m in members], targets, False, fusion, weights))
        on = summarise(suite_scores([caches[m] for m in members], targets, True, fusion, weights))
        tta_on = on["mean_auc"] >= off["mean_auc"] and on["worst_auc"] >= off["worst_auc"]

    final = summarise(suite_scores([caches[m] for m in members], targets, tta_on,
                                   fusion, weights))
    table.append({"kind": "chosen", "name": "ensemble", "members": "+".join(members),
                  "rule": fusion, "tta": int(tta_on),
                  "weights": "/".join(f"{w:.2f}" for w in weights), **final})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = ["kind", "name", "members", "rule", "tta", "weights",
              "clean_auc", "mean_auc", "worst_auc", "std_auc"]
    with open(f"{args.out}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in table:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    chosen_caches = [caches[m] for m in members]
    thr_name, thr_report = pick_threshold(targets, chosen_caches, tta_on, fusion, weights)
    chosen = {
        "members": members,
        "ckpts": [paths[m] for m in members],
        "fusion": fusion,
        "weights": weights,
        "tta": bool(tta_on),
        "threshold_policy": thr_name,
        "threshold": thr_report[thr_name]["threshold"],
        "threshold_candidates": thr_report,
        "val": final,
    }
    with open(f"{args.out}_chosen.json", "w", encoding="utf-8") as f:
        json.dump(chosen, f, indent=2)

    z = np.stack([member_logits(c, "clean", tta_on) for c in chosen_caches])
    probs = fuse(z, fusion, weights)
    with open(f"{args.out}_val_preds.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["image_id", "label", "confidence"])
        for r, p in zip(rows, probs):
            wr.writerow([r["image_id"], r["label"], f"{p:.8f}"])

    print(json.dumps({k: chosen[k] for k in ("members", "fusion", "weights", "tta",
                                             "threshold_policy", "threshold", "val")}, indent=2))
    print(f"wrote {args.out}.csv, {args.out}_chosen.json, {args.out}_val_preds.csv")


if __name__ == "__main__":
    main()
