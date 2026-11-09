import os
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def load_fold_data(fold_num, split_csv):
    split_df = pd.read_csv(split_csv)
    val_df = split_df[split_df["split"] == "val"].reset_index(drop=True)
    y_true = val_df["label"].values.astype(int)

    fold_dir = f"outputs/ablations/v14/fold{fold_num}"
    candidates = {
        "cnxt_22k": f"{fold_dir}/cnxt_22k_consistency",
        "swin_22k": f"{fold_dir}/swin_22k_consistency",
        "maxvit_t": f"{fold_dir}/maxvit_t_consistency",
        "dino": f"{fold_dir}/dino_consistency",
        "srm": f"{fold_dir}/srm_consistency",
        "cnxt_base": f"{fold_dir}/cnxt_base",
        "cnxt_sbi": f"{fold_dir}/cnxt_22k_sbi",
        "cnxt_base_sbi": f"{fold_dir}/cnxt_base_sbi",
        "swin_sbi": f"{fold_dir}/swin_22k_sbi",
        "maxvit_sbi": f"{fold_dir}/maxvit_t_sbi",
        "dino_sbi": f"{fold_dir}/dino_sbi",
        "srm_sbi": f"{fold_dir}/srm_sbi",
    }
    
    loaded = {}
    for name, p in candidates.items():
        npz_p = f"{p}/validation_logits.npz"
        if os.path.exists(npz_p):
            loaded[name] = np.load(npz_p)

    views = [k for k in loaded["cnxt_22k"].keys() if loaded["cnxt_22k"][k].dtype.kind in ('f', 'i') and k != "targets"]
    
    probs = {}
    for name, d in loaded.items():
        cp = 1.0 / (1.0 + np.exp(-d["clean"].astype(np.float64)))
        probs[name] = {
            "clean": cp,
            "stress": {v: 1.0 / (1.0 + np.exp(-d[v].astype(np.float64))) for v in views if v != "clean" and v in d}
        }
    return y_true, views, probs, list(loaded.keys())

y0, views0, p0, cand0 = load_fold_data(0, "data/splits/split_v1.csv")
y4, views4, p4, cand4 = load_fold_data(4, "data/splits/split_v1_fold4.csv")

def score_fold(y_true, views, probs, ens_members):
    cp = np.mean([probs[m]["clean"] for m in ens_members], axis=0)
    c_auc = roc_auc_score(y_true, cp)
    s_aucs = []
    for v in views:
        if v == "clean": continue
        vp = np.mean([probs[m]["stress"][v] for m in ens_members if v in probs[m]["stress"]], axis=0)
        s_aucs.append(roc_auc_score(y_true, vp))
    sm = float(np.mean(s_aucs))
    score = 0.5 * (c_auc + sm)
    return c_auc, sm, score

def joint_eval(ens_members):
    c0, sm0, sc0 = score_fold(y0, views0, p0, ens_members)
    c4, sm4, sc4 = score_fold(y4, views4, p4, ens_members)
    joint = 0.5 * (sc0 + sc4)
    return joint, sc0, sc4, c0, sm0, c4, sm4

print("=" * 85)
print("JOINT CROSS-FOLD OPTIMIZATION (FOLD 0 + FOLD 4 MEAN)")
print("=" * 85)

# Benchmark v17 baseline
v17_members = ["cnxt_22k", "swin_22k", "maxvit_t", "dino", "srm"]
j17, sc0_17, sc4_17, _, _, _, _ = joint_eval(v17_members)
print(f"v17 Baseline (5 families) -> Joint Mean: {j17:.5f} | Fold 0: {sc0_17:.5f} | Fold 4: {sc4_17:.5f}")

# Joint forward selection
remaining = list(cand0)
selected = []
best_joint = 0.0

print("\n--- Joint Forward Selection ---")
while remaining:
    best_candidate = None
    best_cand_score = -1.0
    best_details = None
    for cand in remaining:
        trial = selected + [cand]
        j_trial, sc0_t, sc4_t, c0, sm0, c4, sm4 = joint_eval(trial)
        if j_trial > best_cand_score:
            best_cand_score = j_trial
            best_candidate = cand
            best_details = (j_trial, sc0_t, sc4_t)
            
    if best_cand_score > best_joint or len(selected) < 4:
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_joint = best_cand_score
        j, sc0, sc4 = best_details
        print(f"Step {len(selected)}: Added {best_candidate:<15} -> Joint: {j:.5f} (delta: {j - j17:+.5f}) | Fold 0: {sc0:.5f} | Fold 4: {sc4:.5f}")
    else:
        break

print("-" * 85)
print(f"OPTIMAL JOINT ENSEMBLE ({len(selected)} members): {selected}")
print(f"Optimal Joint Score: {best_joint:.5f} (+{best_joint - j17:.5f} over v17 {j17:.5f})")
print("=" * 85)

# Also test candidate architectures
suites = {
    "v17 Foundation (5)": ["cnxt_22k", "swin_22k", "maxvit_t", "dino", "srm"],
    "Dual 10 (Clean+SBI 5 fam)": ["cnxt_22k", "cnxt_sbi", "swin_22k", "swin_sbi", "maxvit_t", "maxvit_sbi", "dino", "dino_sbi", "srm", "srm_sbi"],
    "High-Cap Hybrid (Base+Base_SBI+Swin+MaxViT+DINO)": ["cnxt_base", "cnxt_base_sbi", "swin_22k", "swin_sbi", "maxvit_t", "dino"],
    "Super Mega 12": cand0,
    "Optimal Joint": selected
}

print(f"\n{'Suite Name':<35} | {'Joint Score':<12} | {'Fold 0 Score':<12} | {'Fold 4 Score':<12} | {'Delta vs v17':<12}")
print("-" * 85)
for sname, smembers in suites.items():
    j, s0, s4, _, _, _, _ = joint_eval(smembers)
    print(f"{sname:<35} | {j:<12.5f} | {s0:<12.5f} | {s4:<12.5f} | {j - j17:<+12.5f}")
