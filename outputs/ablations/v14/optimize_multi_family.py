import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

def load_fold(fold_name, split_csv):
    base_dir = Path("outputs/ablations/v14") / fold_name
    split_df = pd.read_csv(split_csv)
    val_df = split_df[split_df["split"] == "val"].reset_index(drop=True)
    y_true = val_df["label"].astype(int).values

    models = {
        "resnet50": np.load(base_dir / "consistency" / "validation_logits.npz"),
        "cnxt_22k": np.load(base_dir / "cnxt_22k_consistency" / "validation_logits.npz"),
        "swin_22k": np.load(base_dir / "swin_22k_consistency" / "validation_logits.npz"),
        "maxvit_t": np.load(base_dir / "maxvit_t_consistency" / "validation_logits.npz"),
        "dino": np.load(base_dir / "dino_consistency" / "validation_logits.npz"),
        "srm": np.load(base_dir / "srm_consistency" / "validation_logits.npz"),
    }
    views = [k for k in models["cnxt_22k"].keys() if k != "image_ids" and isinstance(models["cnxt_22k"][k], np.ndarray) and models["cnxt_22k"][k].dtype.kind in "fc"]
    return models, y_true, views

def main():
    m0, y0, views0 = load_fold("fold0", "data/splits/split_v1.csv")
    m4, y4, views4 = load_fold("fold4", "data/splits/split_v1_fold4.csv")

    model_names = ["cnxt_22k", "swin_22k", "maxvit_t", "resnet50", "dino", "srm"]

    # Objective: optimize weights on Fold 0 to maximize mean AUC across clean and all stress views
    stress_views = [v for v in views0 if v != "clean"]

    # Pre-extract logits matrix for fold 0 and fold 4
    # Shape: (num_models, num_samples) for each view
    views_mat0 = {v: np.stack([m0[m][v].astype(float) for m in model_names], axis=0) for v in views0}
    views_mat4 = {v: np.stack([m4[m][v].astype(float) for m in model_names], axis=0) for v in views4}

    def loss_func(w):
        w = np.exp(w)
        w = w / np.sum(w)
        # We want to maximize: 0.5 * clean_auc + 0.5 * mean(stress_aucs)
        clean_logits = np.sum(views_mat0["clean"] * w[:, None], axis=0)
        p_clean = 1.0 / (1.0 + np.exp(-clean_logits))
        auc_clean = roc_auc_score(y0, p_clean)

        s_aucs = []
        for v in stress_views:
            logits = np.sum(views_mat0[v] * w[:, None], axis=0)
            p = 1.0 / (1.0 + np.exp(-logits))
            s_aucs.append(roc_auc_score(y0, p))

        score = 0.5 * auc_clean + 0.5 * np.mean(s_aucs)
        return -score

    # Optimize
    init_w = np.zeros(len(model_names))
    res = minimize(loss_func, init_w, method="Nelder-Mead", options={"maxiter": 2000})

    opt_w = np.exp(res.x)
    opt_w = opt_w / np.sum(opt_w)

    print("Optimal Weights (Optimized on Fold 0):")
    for name, w in zip(model_names, opt_w):
        print(f"  {name:<15}: {w:.4f} ({w*100:.1f}%)")

    # Evaluate on Fold 0 and Fold 4 with optimal weights vs equal weights
    def eval_suite(views_mat, y_true):
        def calc(w):
            w = np.array(w).reshape(-1, 1)
            clean_logits = np.sum(views_mat["clean"] * w, axis=0)
            p_clean = 1.0 / (1.0 + np.exp(-clean_logits))
            auc_clean = roc_auc_score(y_true, p_clean)

            s_aucs = {}
            for v in stress_views:
                logits = np.sum(views_mat[v] * w, axis=0)
                p = 1.0 / (1.0 + np.exp(-logits))
                s_aucs[v] = roc_auc_score(y_true, p)

            return {
                "clean": auc_clean,
                "pixelate8": s_aucs.get("pixelate8", 0.0),
                "stress_mean": np.mean(list(s_aucs.values())),
                "stress_worst": np.min(list(s_aucs.values())),
            }

        eq_w = [1.0 / len(model_names)] * len(model_names)
        return calc(eq_w), calc(opt_w)

    eq0, opt0 = eval_suite(views_mat0, y0)
    eq4, opt4 = eval_suite(views_mat4, y4)

    print("\n--- FOLD 0 COMPARISON ---")
    print(f"Equal Weights  : Clean={eq0['clean']:.5f}, Pix8={eq0['pixelate8']:.5f}, StressMean={eq0['stress_mean']:.5f}, Worst={eq0['stress_worst']:.5f}")
    print(f"Optimal Weights: Clean={opt0['clean']:.5f}, Pix8={opt0['pixelate8']:.5f}, StressMean={opt0['stress_mean']:.5f}, Worst={opt0['stress_worst']:.5f}")

    print("\n--- FOLD 4 CROSS-VALIDATION (Held-Out Test of Weights) ---")
    print(f"Equal Weights  : Clean={eq4['clean']:.5f}, Pix8={eq4['pixelate8']:.5f}, StressMean={eq4['stress_mean']:.5f}, Worst={eq4['stress_worst']:.5f}")
    print(f"Optimal Weights: Clean={opt4['clean']:.5f}, Pix8={opt4['pixelate8']:.5f}, StressMean={opt4['stress_mean']:.5f}, Worst={opt4['stress_worst']:.5f}")

if __name__ == "__main__":
    main()
