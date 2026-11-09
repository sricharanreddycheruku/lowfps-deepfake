import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# 1. Load fold 0 metadata
split_df = pd.read_csv("data/splits/split_v1.csv")
val_df = split_df[split_df["split"] == "val"].reset_index(drop=True)
y_true = val_df["label"].astype(int).values
groups = val_df["group_id"].astype(str).values

# 2. Load model predictions
models_npz = {
    "resnet50": np.load("outputs/ablations/v14/fold0/consistency/validation_logits.npz"),
    "cnxt_1k": np.load("outputs/ablations/v14/fold0/cnxt_small_consistency/validation_logits.npz"),
    "swin_1k": np.load("outputs/ablations/v14/fold0/swin_t_consistency/validation_logits.npz"),
    "cnxt_22k": np.load("outputs/ablations/v14/fold0/cnxt_22k_consistency/validation_logits.npz"),
    "swin_22k": np.load("outputs/ablations/v14/fold0/swin_22k_consistency/validation_logits.npz"),
    "effv2s_21k": np.load("outputs/ablations/v14/fold0/effv2s_21k_consistency/validation_logits.npz"),
}

# Determine common views
views = [k for k in models_npz["cnxt_22k"].keys() if k != "image_ids" and isinstance(models_npz["cnxt_22k"][k], np.ndarray) and models_npz["cnxt_22k"][k].dtype.kind in "fc"]
stress_views = [v for v in views if v != "clean"]

def ensemble_logits(model_keys, npz_dict, weights=None):
    if weights is None:
        weights = [1.0 / len(model_keys)] * len(model_keys)
    out = {}
    for v in views:
        stacked = np.stack([npz_dict[m][v].astype(float) for m in model_keys], axis=0)
        w = np.array(weights).reshape(-1, 1)
        out[v] = np.sum(stacked * w, axis=0)
    return out

ens_v14_trio = ensemble_logits(["resnet50", "cnxt_1k", "swin_1k"], models_npz)
ens_found_trio = ensemble_logits(["resnet50", "cnxt_22k", "swin_22k"], models_npz)
ens_found_4way = ensemble_logits(["resnet50", "cnxt_22k", "swin_22k", "effv2s_21k"], models_npz)
ens_all_6way = ensemble_logits(["resnet50", "cnxt_1k", "swin_1k", "cnxt_22k", "swin_22k", "effv2s_21k"], models_npz)

configurations = {
    "v14_consistent_trio (1k)": ens_v14_trio,
    "foundation_trio (22k)": ens_found_trio,
    "foundation_4way (22k+21k)": ens_found_4way,
    "mega_6way (1k+22k+21k)": ens_all_6way,
    "single_swin_22k": {v: models_npz["swin_22k"][v].astype(float) for v in views},
    "single_cnxt_22k": {v: models_npz["cnxt_22k"][v].astype(float) for v in views},
}

def evaluate_suite(logits_dict):
    aucs = {}
    for v in views:
        probs = 1.0 / (1.0 + np.exp(-logits_dict[v]))
        aucs[v] = float(roc_auc_score(y_true, probs))
    s_aucs = [aucs[v] for v in stress_views]
    return {
        "clean": aucs["clean"],
        "pixelate8": aucs.get("pixelate8", 0.0),
        "stress_mean": float(np.mean(s_aucs)),
        "stress_worst": float(np.min(s_aucs)),
        "worst_view": stress_views[int(np.argmin(s_aucs))]
    }

print("=== FOLD 0 ENSEMBLE SUITE EVALUATION ===")
summary = {}
for name, log_dict in configurations.items():
    stats = evaluate_suite(log_dict)
    summary[name] = stats
    print(f"{name:26s} | Clean: {stats['clean']:.5f} | Pixelate8: {stats['pixelate8']:.5f} | Stress Mean: {stats['stress_mean']:.5f} | Stress Worst: {stats['stress_worst']:.5f} ({stats['worst_view']})")

# Paired Group Bootstrap between v14_consistent_trio and mega_6way & foundation_trio
unique_groups = np.unique(groups)
n_groups = len(unique_groups)
rng = np.random.RandomState(42)
B = 1000

diffs_found_trio = {"clean": [], "stress_mean": [], "stress_worst": [], "pixelate8": []}
diffs_6way = {"clean": [], "stress_mean": [], "stress_worst": [], "pixelate8": []}

for b in range(B):
    sample_g = rng.choice(unique_groups, size=n_groups, replace=True)
    idx_map = {g: np.where(groups == g)[0] for g in np.unique(sample_g)}
    sample_idx = np.concatenate([idx_map[g] for g in sample_g])
    
    y_b = y_true[sample_idx]
    if len(np.unique(y_b)) < 2:
        continue
    
    # helper for one config
    def get_b_metrics(log_dict):
        p_clean = 1.0 / (1.0 + np.exp(-log_dict["clean"][sample_idx]))
        c_auc = roc_auc_score(y_b, p_clean)
        
        p_pix8 = 1.0 / (1.0 + np.exp(-log_dict["pixelate8"][sample_idx]))
        pix_auc = roc_auc_score(y_b, p_pix8)
        
        s_vals = []
        for sv in stress_views:
            p_s = 1.0 / (1.0 + np.exp(-log_dict[sv][sample_idx]))
            s_vals.append(roc_auc_score(y_b, p_s))
        return c_auc, pix_auc, np.mean(s_vals), np.min(s_vals)

    c_base, pix_base, m_base, w_base = get_b_metrics(ens_v14_trio)
    c_f, pix_f, m_f, w_f = get_b_metrics(ens_found_trio)
    c_6, pix_6, m_6, w_6 = get_b_metrics(ens_all_6way)
    
    diffs_found_trio["clean"].append(c_f - c_base)
    diffs_found_trio["pixelate8"].append(pix_f - pix_base)
    diffs_found_trio["stress_mean"].append(m_f - m_base)
    diffs_found_trio["stress_worst"].append(w_f - w_base)
    
    diffs_6way["clean"].append(c_6 - c_base)
    diffs_6way["pixelate8"].append(pix_6 - pix_base)
    diffs_6way["stress_mean"].append(m_6 - m_base)
    diffs_6way["stress_worst"].append(w_6 - w_base)

def ci(arr):
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

print("\n=== PAIRED BOOTSTRAP DELTAS (vs v14 Consistent Trio) ===")
print("Foundation Trio (22k) vs v14 Consistent Trio:")
for k in ["clean", "pixelate8", "stress_mean", "stress_worst"]:
    m, lo, hi = ci(diffs_found_trio[k])
    print(f"  {k:12s}: mean {m:+.5f} (95% CI [{lo:+.5f}, {hi:+.5f}])")

print("\nMega 6-Way Blend vs v14 Consistent Trio:")
for k in ["clean", "pixelate8", "stress_mean", "stress_worst"]:
    m, lo, hi = ci(diffs_6way[k])
    print(f"  {k:12s}: mean {m:+.5f} (95% CI [{lo:+.5f}, {hi:+.5f}])")

with open("outputs/ablations/v14/fold0/comparison_foundation.json", "w") as f:
    json.dump({
        "summary": summary,
        "deltas_found_trio": {k: ci(diffs_found_trio[k]) for k in diffs_found_trio},
        "deltas_6way": {k: ci(diffs_6way[k]) for k in diffs_6way}
    }, f, indent=2)
