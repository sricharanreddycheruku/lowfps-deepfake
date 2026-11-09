import os
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def analyze_fold(fold_num, split_csv):
    print("=" * 85)
    print(f"COMPLETE ABLATION & ENSEMBLE SUITE EVALUATION: FOLD {fold_num}")
    print("=" * 85)
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
    
    print(f"{'Model Architecture':<30} | {'Clean AUC':<10} | {'Stress Mean':<12} | {'Stress Min':<11} | {'Score':<8}")
    print("-" * 85)
    
    probs = {}
    for name, d in loaded.items():
        cp = 1.0 / (1.0 + np.exp(-d["clean"].astype(np.float64)))
        c_auc = roc_auc_score(y_true, cp)
        s_aucs = [roc_auc_score(y_true, 1.0 / (1.0 + np.exp(-d[v].astype(np.float64)))) for v in views if v != "clean" and v in d]
        sm = float(np.mean(s_aucs))
        smin = float(np.min(s_aucs))
        score = 0.5 * (c_auc + sm)
        tag = " [SBI]" if "sbi" in name else (" [BASE]" if "base" in name else "")
        print(f"{(name + tag):<30} | {c_auc:<10.5f} | {sm:<12.5f} | {smin:<11.5f} | {score:<8.5f}")
        probs[name] = {
            "clean": cp,
            "stress": {v: 1.0 / (1.0 + np.exp(-d[v].astype(np.float64))) for v in views if v != "clean" and v in d}
        }

    print("-" * 85)
    
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

    # 1. Incumbent 5-family foundation suite (v17 architecture)
    c17, sm17, smin17, sc17 = score_ensemble(["cnxt_22k", "swin_22k", "maxvit_t", "dino", "srm"])
    print(f"{'v17 Foundation Suite (5 fam)':<30} | {c17:<10.5f} | {sm17:<12.5f} | {smin17:<11.5f} | {sc17:<8.5f} [BASELINE]")

    # 2. Pure SBI 5-family suite
    c_sbi, sm_sbi, smin_sbi, sc_sbi = score_ensemble(["cnxt_sbi", "swin_sbi", "maxvit_sbi", "dino_sbi", "srm_sbi"])
    print(f"{'Pure SBI Suite (5 fam)':<30} | {c_sbi:<10.5f} | {sm_sbi:<12.5f} | {smin_sbi:<11.5f} | {sc_sbi:<8.5f} (delta: {sc_sbi - sc17:+.5f})")

    # 3. Dual-Stream Pair Hybrid (Clean + SBI for all 5 families: 10 models)
    dual_10 = ["cnxt_22k", "cnxt_sbi", "swin_22k", "swin_sbi", "maxvit_t", "maxvit_sbi", "dino", "dino_sbi", "srm", "srm_sbi"]
    c_d10, sm_d10, smin_d10, sc_d10 = score_ensemble(dual_10)
    print(f"{'Dual 10 (Clean+SBI 5 fam)':<30} | {c_d10:<10.5f} | {sm_d10:<12.5f} | {smin_d10:<11.5f} | {sc_d10:<8.5f} (delta: {sc_d10 - sc17:+.5f})")

    # 4. Super Mega Hybrid: Dual 10 + Base + Base_SBI (12 models)
    mega_12 = list(loaded.keys())
    c_m12, sm_m12, smin_m12, sc_m12 = score_ensemble(mega_12)
    print(f"{'Super Mega Hybrid (12 models)':<30} | {c_m12:<10.5f} | {sm_m12:<12.5f} | {smin_m12:<11.5f} | {sc_m12:<8.5f} (delta: {sc_m12 - sc17:+.5f})")

    # 5. Greedy forward selection to find mathematically optimal ensemble
    remaining = list(loaded.keys())
    selected = []
    best_score = 0.0
    print("\n--- Greedy Forward Ensemble Optimization ---")
    while remaining:
        best_candidate = None
        best_cand_score = -1.0
        best_metrics = None
        for cand in remaining:
            trial = selected + [cand]
            c_trial, sm_trial, smin_trial, sc_trial = score_ensemble(trial)
            if sc_trial > best_cand_score:
                best_cand_score = sc_trial
                best_candidate = cand
                best_metrics = (c_trial, sm_trial, smin_trial, sc_trial)
        
        if best_cand_score > best_score or len(selected) < 4:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            best_score = best_cand_score
            c, sm, smin, sc = best_metrics
            print(f"Step {len(selected)}: Added {best_candidate:<15} -> Clean {c:.5f} | Stress {sm:.5f} | Min {smin:.5f} | Score {sc:.5f} (delta vs v17: {sc - sc17:+.5f})")
        else:
            break
            
    print(f"Optimal Ensemble ({len(selected)} models): {selected}")
    return sc17, sc_d10, sc_m12, best_score, selected

print("\n")
sc0_17, sc0_d10, sc0_m12, sc0_opt, sel0 = analyze_fold(0, "data/splits/split_v1.csv")
print("\n" + "="*85 + "\n")
sc4_17, sc4_d10, sc4_m12, sc4_opt, sel4 = analyze_fold(4, "data/splits/split_v1_fold4.csv")
