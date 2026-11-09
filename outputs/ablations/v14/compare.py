"""Compare paired-view arms using the same bootstrap draws over held-out groups."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path.cwd() / "src"))
from ensemble_pool import fast_auc
from robust_eval import EXTENDED_VARIANTS, STRESS_VARIANTS, VARIANTS, load_rows


def summaries(auc, mixed_auc):
    standard = [auc[name] for name in VARIANTS]
    stress = [auc[name] for name in STRESS_VARIANTS if name != "clean"]
    extended = [auc[name] for name in EXTENDED_VARIANTS if name != "clean"]
    return {
        "clean": auc["clean"],
        "mean_23": float(np.mean(standard)),
        "worst_23": min(standard),
        "extended_mean": float(np.mean(extended)),
        "stress_mean": float(np.mean(stress)),
        "stress_worst": min(stress),
        "pixelate8": auc["pixelate8"],
        "mixed_auc": mixed_auc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="data/splits/split_v1.csv")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=["control", "aug", "consistency"])
    parser.add_argument("--trio", action="store_true",
                        help="compare fixed replacements and a low-resolution add-on")
    parser.add_argument("--include-dino", action="store_true",
                        help="include an additional fixed-weight DINO blend in trio mode")
    args = parser.parse_args()
    if args.include_dino and not args.trio:
        parser.error("--include-dino requires --trio")
    if args.trio:
        args.arms = ["control", "aug", "consistency", "lowres", "cnxt_small", "swin_t"]
    if args.include_dino:
        args.arms.append("dino")
    rows = load_rows(args.split, split="val")
    ids = np.array([row["image_id"] for row in rows])
    y = np.array([int(row["label"]) for row in rows])
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row["group_id"], []).append(i)
    group_indices = list(groups.values())

    names = list({**VARIANTS, **EXTENDED_VARIANTS, **STRESS_VARIANTS})
    logits = {}
    for arm in args.arms:
        with np.load(args.results / arm / "validation_logits.npz") as data:
            np.testing.assert_array_equal(data["image_ids"], ids)
            logits[arm] = np.stack([data[name] for name in names])
    if args.trio:
        others = logits["cnxt_small"] + logits["swin_t"]
        base = (others + logits["control"]) / 3
        blends = {
            "control": base,
            "aug": (others + logits["aug"]) / 3,
            "consistency": (others + logits["consistency"]) / 3,
            "lowres_addon": 0.8 * base + 0.2 * logits["lowres"],
        }
        if args.include_dino:
            blends["dino_addon"] = 0.8 * base + 0.2 * logits["dino"]
        logits = blends
        args.arms = list(logits)
    comparisons = [(arm, args.arms[0]) for arm in args.arms[1:]]
    if "aug" in args.arms and "consistency" in args.arms:
        comparisons.append(("consistency", "aug"))

    def aucs(arm, idx):
        return {name: fast_auc(y[idx], values[idx]) for name, values in zip(names, logits[arm])}

    # A fixed simulated test mixture measures cross-quality ranking. Each draw
    # assigns a variant independently of the image label and shares it across arms.
    clean = names.index("clean")
    standard = [names.index(name) for name in VARIANTS if name != "clean"]
    stress = [names.index(name) for name in STRESS_VARIANTS if name != "clean"]
    assignments = []
    for seed in range(5):
        generator = np.random.default_rng(seed)
        family = generator.choice(3, size=len(y), p=[0.5, 0.25, 0.25])
        assignments.append(np.select(
            [family == 1, family == 2],
            [generator.choice(standard, size=len(y)), generator.choice(stress, size=len(y))],
            default=clean))
    all_idx = np.arange(len(y))
    mixed = {arm: [logits[arm][assigned, all_idx] for assigned in assignments]
             for arm in args.arms}

    def summarize_arm(arm, idx, auc):
        mixed_auc = float(np.mean([fast_auc(y[idx], pred[idx]) for pred in mixed[arm]]))
        return summaries(auc, mixed_auc)

    full = {arm: aucs(arm, all_idx) for arm in args.arms}
    observed = {arm: summarize_arm(arm, all_idx, auc) for arm, auc in full.items()}
    metrics = list(observed[args.arms[0]])
    boot = {f"{a}_minus_{b}": [] for a, b in comparisons}
    rng = np.random.default_rng(0)
    for _ in range(1000):
        sampled = rng.integers(len(group_indices), size=len(group_indices))
        idx = np.concatenate([group_indices[i] for i in sampled])
        if np.unique(y[idx]).size != 2:
            continue
        result = {arm: summarize_arm(arm, idx, aucs(arm, idx)) for arm in args.arms}
        for a, b in comparisons:
            boot[f"{a}_minus_{b}"].append([result[a][name] - result[b][name] for name in metrics])

    outcome = {
        "split": args.split,
        "images": len(rows),
        "groups": len(groups),
        "summary": observed,
        "auc_by_variant": full,
        "comparisons": {},
    }
    for a, b in comparisons:
        key = f"{a}_minus_{b}"
        ci = np.percentile(boot[key], [2.5, 97.5], axis=0).T.tolist()
        outcome["comparisons"][key] = {
            "delta": {name: observed[a][name] - observed[b][name] for name in metrics},
            "95pct_ci": dict(zip(metrics, ci)),
        }
    filename = "comparison_trio.json" if args.trio else "comparison.json"
    (args.results / filename).write_text(json.dumps(outcome, indent=2) + "\n",
                                                  encoding="utf-8")
    print(json.dumps({k: v for k, v in outcome.items() if k != "auc_by_variant"}, indent=2))


if __name__ == "__main__":
    main()
