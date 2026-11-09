import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

split_df = pd.read_csv("data/splits/split_v1_fold4.csv")
val_df = split_df[split_df["split"] == "val"].reset_index(drop=True)
y_true = val_df["label"].astype(int).values

npz_1k_base = np.load("outputs/ablations/v14/fold4/cnxt_small/validation_logits.npz")
npz_1k_cons = np.load("outputs/ablations/v14/fold4/cnxt_small_consistency/validation_logits.npz")
npz_22k_cons = np.load("outputs/ablations/v14/fold4/cnxt_22k_consistency/validation_logits.npz")

views = [k for k in npz_22k_cons.keys() if k != "image_ids" and isinstance(npz_22k_cons[k], np.ndarray) and npz_22k_cons[k].dtype.kind in "fc"]

def get_stats(npz):
    aucs = {}
    for v in views:
        if v not in npz:
            continue
        logits = npz[v].astype(float)
        probs = 1.0 / (1.0 + np.exp(-logits))
        aucs[v] = float(roc_auc_score(y_true, probs))
    stress_views = [v for v in views if v != "clean" and v in aucs]
    stress_aucs = [aucs[v] for v in stress_views]
    return {
        "clean": aucs["clean"],
        "pixelate8": aucs.get("pixelate8", None),
        "stress_mean": float(np.mean(stress_aucs)),
        "stress_worst": float(np.min(stress_aucs)),
        "worst_view": stress_views[int(np.argmin(stress_aucs))]
    }

s_1k_base = get_stats(npz_1k_base)
s_1k_cons = get_stats(npz_1k_cons)
s_22k_cons = get_stats(npz_22k_cons)

print("=== CONVNEXT FOLD 4 COMPARISON ===")
print(f"1k Baseline:    clean={s_1k_base['clean']:.5f}, pixelate8={s_1k_base['pixelate8']:.5f}, stress_mean={s_1k_base['stress_mean']:.5f}, worst={s_1k_base['stress_worst']:.5f} ({s_1k_base['worst_view']})")
print(f"1k Consistency: clean={s_1k_cons['clean']:.5f}, pixelate8={s_1k_cons['pixelate8']:.5f}, stress_mean={s_1k_cons['stress_mean']:.5f}, worst={s_1k_cons['stress_worst']:.5f} ({s_1k_cons['worst_view']})")
print(f"22k Consistency:clean={s_22k_cons['clean']:.5f}, pixelate8={s_22k_cons['pixelate8']:.5f}, stress_mean={s_22k_cons['stress_mean']:.5f}, worst={s_22k_cons['stress_worst']:.5f} ({s_22k_cons['worst_view']})")
