import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# 1. Load fold 4 metadata
split_df = pd.read_csv("data/splits/split_v1_fold4.csv")
val_df = split_df[split_df["split"] == "val"].reset_index(drop=True)
y_true = val_df["label"].astype(int).values
groups = val_df["group_id"].astype(str).values

# 2. Load model predictions
models_npz = {
    "resnet50": np.load("outputs/ablations/v14/fold4/consistency/validation_logits.npz"),
    "cnxt_1k": np.load("outputs/ablations/v14/fold4/cnxt_small_consistency/validation_logits.npz"),
    "swin_1k": np.load("outputs/ablations/v14/fold4/swin_t_consistency/validation_logits.npz"),
    "cnxt_22k": np.load("outputs/ablations/v14/fold4/cnxt_22k_consistency/validation_logits.npz"),
    "swin_22k": np.load("outputs/ablations/v14/fold4/swin_22k_consistency/validation_logits.npz"),
    "effv2s_21k": np.load("outputs/ablations/v14/fold4/effv2s_21k_consistency/validation_logits.npz"),
    "maxvit": np.load("outputs/ablations/v14/fold4/maxvit_t_consistency/validation_logits.npz"),
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
ens_mega_5way = ensemble_logits(["resnet50", "cnxt_1k", "swin_1k", "cnxt_22k", "swin_22k"], models_npz)
ens_quad_22k = ensemble_logits(["resnet50", "cnxt_22k", "swin_22k", "maxvit"], models_npz)
ens_mega_6way_max = ensemble_logits(["resnet50", "cnxt_1k", "swin_1k", "cnxt_22k", "swin_22k", "maxvit"], models_npz)

configurations = {
    "v14_consistent_trio (1k)": ens_v14_trio,
    "foundation_trio (22k)": ens_found_trio,
    "foundation_4way (22k+21k)": ens_found_4way,
    "mega_5way (1k+22k)": ens_mega_5way,
    "quad_foundation (22k+max)": ens_quad_22k,
    "mega_6way (1k+22k+max)": ens_mega_6way_max,
    "single_maxvit_t": {v: models_npz["maxvit"][v].astype(float) for v in views},
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

print("=== FOLD 4 ENSEMBLE SUITE EVALUATION (WITH MAXVIT) ===")
print(f"{'Configuration':26s} | {'Clean':8s} | {'Pix8':8s} | {'StrMean':8s} | {'StrWorst':8s} | Worst View")
print("-" * 75)
summary = {}
for name, log_dict in configurations.items():
    stats = evaluate_suite(log_dict)
    summary[name] = stats
    print(f"{name:26s} | {stats['clean']:.5f} | {stats['pixelate8']:.5f} | {stats['stress_mean']:.5f} | {stats['stress_worst']:.5f} | {stats['worst_view']}")
