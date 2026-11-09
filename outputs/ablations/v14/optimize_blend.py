"""Evaluate ensembling strategies (weights, rank averaging, calibration) on validation logits."""

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

sys.path.insert(0, str(Path.cwd() / "src"))
from ensemble_pool import fast_auc
from robust_eval import EXTENDED_VARIANTS, STRESS_VARIANTS, VARIANTS, load_rows


def evaluate_blend(y, preds_by_variant):
    names = list(preds_by_variant.keys())
    standard = [names.index(n) for n in VARIANTS]
    stress = [names.index(n) for n in STRESS_VARIANTS if n != "clean"]
    
    aucs = {n: fast_auc(y, preds_by_variant[n]) for n in names}
    return {
        "clean": aucs["clean"],
        "mean_23": float(np.mean([aucs[names[i]] for i in standard])),
        "worst_23": min([aucs[names[i]] for i in standard]),
        "stress_mean": float(np.mean([aucs[names[i]] for i in stress])),
        "stress_worst": min([aucs[names[i]] for i in stress]),
        "pixelate8": aucs["pixelate8"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="data/splits/split_v1.csv")
    parser.add_argument("--results", type=Path, default=Path("outputs/ablations/v14/fold0"))
    args = parser.parse_args()

    rows = load_rows(args.split, split="val")
    ids = np.array([row["image_id"] for row in rows])
    y = np.array([int(row["label"]) for row in rows])

    names = list({**VARIANTS, **EXTENDED_VARIANTS, **STRESS_VARIANTS})

    runs = ["consistency", "cnxt_small_consistency", "swin_t_consistency"]
    logits = {}
    for r in runs:
        with np.load(args.results / r / "validation_logits.npz") as data:
            np.testing.assert_array_equal(data["image_ids"], ids)
            logits[r] = np.stack([data[n] for n in names])

    m_res = logits["consistency"]
    m_cnxt = logits["cnxt_small_consistency"]
    m_swin = logits["swin_t_consistency"]

    # 1. Equal-weight Logit Mean
    mean_preds = {name: (m_res[i] + m_cnxt[i] + m_swin[i]) / 3.0 for i, name in enumerate(names)}
    res_mean = evaluate_blend(y, mean_preds)

    # 2. Rank Averaging
    rank_preds = {}
    for i, name in enumerate(names):
        r_res = rankdata(m_res[i]) / len(y)
        r_cnxt = rankdata(m_cnxt[i]) / len(y)
        r_swin = rankdata(m_swin[i]) / len(y)
        rank_preds[name] = (r_res + r_cnxt + r_swin) / 3.0
    res_rank = evaluate_blend(y, rank_preds)

    # 3. Optimize Weights on Clean + Worst-23 standard
    def loss(w):
        w = np.array(w)
        w = w / np.sum(w)
        combined = w[0] * m_res[names.index("clean")] + w[1] * m_cnxt[names.index("clean")] + w[2] * m_swin[names.index("clean")]
        return -fast_auc(y, combined)

    opt = minimize(loss, [1/3, 1/3, 1/3], bounds=[(0.05, 0.9), (0.05, 0.9), (0.05, 0.9)])
    opt_w = (opt.x / np.sum(opt.x)).round(4)
    opt_preds = {name: opt_w[0] * m_res[i] + opt_w[1] * m_cnxt[i] + opt_w[2] * m_swin[i] for i, name in enumerate(names)}
    res_opt = evaluate_blend(y, opt_preds)

    print("=== Strategy Comparison on Fold 0 Consistent Trio ===")
    print(f"1. Equal Mean (1/3, 1/3, 1/3):")
    print(f"   Clean: {res_mean['clean']:.5f} | Worst-23: {res_mean['worst_23']:.5f} | Stress Mean: {res_mean['stress_mean']:.5f} | Stress Worst: {res_mean['stress_worst']:.5f}")
    print(f"2. Rank Averaging:")
    print(f"   Clean: {res_rank['clean']:.5f} | Worst-23: {res_rank['worst_23']:.5f} | Stress Mean: {res_rank['stress_mean']:.5f} | Stress Worst: {res_rank['stress_worst']:.5f}")
    print(f"3. Optimized Weights {opt_w.tolist()}:")
    print(f"   Clean: {res_opt['clean']:.5f} | Worst-23: {res_opt['worst_23']:.5f} | Stress Mean: {res_opt['stress_mean']:.5f} | Stress Worst: {res_opt['stress_worst']:.5f}")


if __name__ == "__main__":
    main()
