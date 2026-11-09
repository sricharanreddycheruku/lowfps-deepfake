"""Compare Fold 4 baseline control trio with all-consistent ensemble across held-out groups."""

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
    parser.add_argument("--split", default="data/splits/split_v1_fold4.csv")
    parser.add_argument("--results", type=Path, default=Path("outputs/ablations/v14/fold4"))
    args = parser.parse_args()

    rows = load_rows(args.split, split="val")
    ids = np.array([row["image_id"] for row in rows])
    y = np.array([int(row["label"]) for row in rows])
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row["group_id"], []).append(i)
    group_indices = list(groups.values())

    names = list({**VARIANTS, **EXTENDED_VARIANTS, **STRESS_VARIANTS})

    required_runs = [
        "control", "consistency", "cnxt_small", "swin_t",
        "cnxt_small_consistency", "swin_t_consistency"
    ]
    raw_logits = {}
    for run in required_runs:
        run_file = args.results / run / "validation_logits.npz"
        if not run_file.exists():
            print(f"Waiting: {run_file} does not exist yet.")
            return
        with np.load(run_file) as data:
            np.testing.assert_array_equal(data["image_ids"], ids)
            raw_logits[run] = np.stack([data[name] for name in names])

    # Construct the ensembles to compare on Fold 4
    blends = {
        # 1. Baseline control trio (v11 equivalent on Fold 4)
        "baseline_trio": (raw_logits["control"] + raw_logits["cnxt_small"] + raw_logits["swin_t"]) / 3.0,
        # 2. ResNet consistency only
        "resnet_consistency_trio": (raw_logits["consistency"] + raw_logits["cnxt_small"] + raw_logits["swin_t"]) / 3.0,
        # 3. ConvNeXt consistency only
        "cnxt_consistency_trio": (raw_logits["control"] + raw_logits["cnxt_small_consistency"] + raw_logits["swin_t"]) / 3.0,
        # 4. Swin-T consistency only
        "swin_consistency_trio": (raw_logits["control"] + raw_logits["cnxt_small"] + raw_logits["swin_t_consistency"]) / 3.0,
        # 5. Full all-consistent ensemble (v15 trio)
        "all_consistent_trio": (raw_logits["consistency"] + raw_logits["cnxt_small_consistency"] + raw_logits["swin_t_consistency"]) / 3.0,
    }

    arm_names = list(blends.keys())
    comparisons = [
        ("resnet_consistency_trio", "baseline_trio"),
        ("cnxt_consistency_trio", "baseline_trio"),
        ("swin_consistency_trio", "baseline_trio"),
        ("all_consistent_trio", "baseline_trio"),
        ("all_consistent_trio", "resnet_consistency_trio"),
    ]

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
    mixed = {arm: [blends[arm][assigned, all_idx] for assigned in assignments]
             for arm in arm_names}

    def aucs(arm, idx):
        return {name: fast_auc(y[idx], values[idx]) for name, values in zip(names, blends[arm])}

    def summarize_arm(arm, idx, auc):
        mixed_auc = float(np.mean([fast_auc(y[idx], pred[idx]) for pred in mixed[arm]]))
        return summaries(auc, mixed_auc)

    full = {arm: aucs(arm, all_idx) for arm in arm_names}
    observed = {arm: summarize_arm(arm, all_idx, auc) for arm, auc in full.items()}
    metrics = list(observed[arm_names[0]])
    boot = {f"{a}_minus_{b}": [] for a, b in comparisons}
    rng = np.random.default_rng(0)
    for _ in range(1000):
        sampled = rng.integers(len(group_indices), size=len(group_indices))
        idx = np.concatenate([group_indices[i] for i in sampled])
        if np.unique(y[idx]).size != 2:
            continue
        result = {arm: summarize_arm(arm, idx, aucs(arm, idx)) for arm in arm_names}
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
    output_path = args.results / "comparison_all_consistent.json"
    output_path.write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
    print(f"Saved evaluation to {output_path}")
    print(json.dumps({k: v for k, v in outcome.items() if k != "auc_by_variant"}, indent=2))


if __name__ == "__main__":
    main()
