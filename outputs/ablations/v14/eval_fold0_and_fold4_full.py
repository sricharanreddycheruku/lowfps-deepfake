import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def evaluate_fold(fold_num, split_csv):
    print("=" * 80)
    print(f"COMPREHENSIVE BENCHMARK: FOLD {fold_num}")
    print("=" * 80)
    split_df = pd.read_csv(split_csv)
    val_df = split_df[split_df["split"] == "val"].reset_index(drop=True)
    y_true = val_df["label"].values.astype(int)

    fold_dir = f"outputs/ablations/v14/fold{fold_num}"
    
    # All candidates
    members = {
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
    for name, p in members.items():
        npz_p = f"{p}/validation_logits.npz"
        if os.path.exists(npz_p):
            loaded[name] = np.load(npz_p)

    views = [k for k in loaded["cnxt_22k"].keys() if loaded["cnxt_22k"][k].dtype.kind in ('f', 'i') and k != "targets"]
    
    print(f"{'Model Architecture':<30} | {'Clean AUC':<10} | {'Stress Mean':<12} | {'Stress Min':<11} | {'Score':<8}")
    print("-" * 80)
    
    probs = {}
    for name, d in loaded.items():
        cp = 1.0 / (1.0 + np.exp(-d["clean"].astype(np.float64)))
        c_auc = roc_auc_score(y_true, cp)
        s_aucs = [roc_auc_score(y_true, 1.0 / (1.0 + np.exp(-d[v].astype(np.float64)))) for v in views if v != "clean" and v in d]
        sm = float(np.mean(s_aucs))
        smin = float(np.min(s_aucs))
        score = 0.5 * (c_auc + sm)
        tag = " [NEW]" if "base" in name or "sbi" in name else ""
        print(f"{(name + tag):<30} | {c_auc:<10.5f} | {sm:<12.5f} | {smin:<11.5f} | {score:<8.5f}")
        probs[name] = {
            "clean": cp,
            "stress": {v: 1.0 / (1.0 + np.exp(-d[v].astype(np.float64))) for v in views if v != "clean" and v in d}
        }

    print("-" * 80)
    
    def score_ensemble(ens_members):
        cp = np.mean([probs[m]["clean"] for m in ens_members], axis=0)
        c_auc = roc_auc_score(y_true, cp)
        s_aucs = []
        for v in views:
            if v == "clean": continue
            vp = np.mean([probs[m]["stress"][v] for m in ens_members if v in probs[m]["stress"]], axis=0)
            s_aucs.append(roc_auc_score(y_true, vp))
        sm = float(np.mean(s_aucs))
        smin = float(np.min(s_aucs))
        score = 0.5 * (c_auc + sm)
        return c_auc, sm, smin, score

    # 1. Incumbent 5-family foundation suite (v17 architecture: cnxt_22k, swin_22k, maxvit_t, dino, srm)
    c17, sm17, smin17, sc17 = score_ensemble(["cnxt_22k", "swin_22k", "maxvit_t", "dino", "srm"])
    print(f"{'v17 Foundation Suite (5 fam)':<30} | {c17:<10.5f} | {sm17:<12.5f} | {smin17:<11.5f} | {sc17:<8.5f}")

    # 2. SBI-enhanced suite
    sbi_models = ["cnxt_22k", "swin_22k", "maxvit_t", "dino", "srm"]
    if "swin_sbi" in loaded:
        # replace swin_22k with swin_sbi or blend
        c_sbi, sm_sbi, smin_sbi, sc_sbi = score_ensemble(["cnxt_22k", "swin_sbi", "maxvit_t", "dino", "srm"])
        print(f"{'SBI Suite (swin_sbi)':<30} | {c_sbi:<10.5f} | {sm_sbi:<12.5f} | {smin_sbi:<11.5f} | {sc_sbi:<8.5f} (delta: {sc_sbi - sc17:+.5f})")

    # 3. Hybrid Super Ensemble: cnxt_base_sbi + swin_sbi (or swin_22k) + maxvit_t + dino + srm
    swin_pick = "swin_sbi" if "swin_sbi" in loaded else "swin_22k"
    c_hyb, sm_hyb, smin_hyb, sc_hyb = score_ensemble(["cnxt_base_sbi", swin_pick, "maxvit_t", "dino", "srm"])
    print(f"{'Hybrid Suite (Base_SBI+Swin)':<30} | {c_hyb:<10.5f} | {sm_hyb:<12.5f} | {smin_hyb:<11.5f} | {sc_hyb:<8.5f} (delta: {sc_hyb - sc17:+.5f})")

    # 4. Multi-backbone Mega Suite: blend all top members
    all_active = [m for m in ["cnxt_22k", "cnxt_base", "cnxt_base_sbi", "swin_22k", "swin_sbi", "maxvit_t", "dino", "srm"] if m in loaded]
    c_mega, sm_mega, smin_mega, sc_mega = score_ensemble(all_active)
    print(f"{f'Mega Ensemble ({len(all_active)} members)':<30} | {c_mega:<10.5f} | {sm_mega:<12.5f} | {smin_mega:<11.5f} | {sc_mega:<8.5f} (delta: {sc_mega - sc17:+.5f})")
    print("=" * 80)
    return sc17, sc_hyb, sc_mega

sc0_17, sc0_hyb, sc0_mega = evaluate_fold(0, "data/splits/split_v1.csv")
print("\n")
sc4_17, sc4_hyb, sc4_mega = evaluate_fold(4, "data/splits/split_v1_fold4.csv")
