import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def eval_fold(fold_num, split_csv):
    print("=" * 70)
    print(f"EVALUATING FOLD {fold_num}")
    print("=" * 70)
    split_df = pd.read_csv(split_csv)
    val_df = split_df[split_df["split"] == "val"].reset_index(drop=True)
    y_true = val_df["label"].values.astype(int)

    fold_dir = f"outputs/ablations/v14/fold{fold_num}"
    
    # 6 incumbent families
    fams = ["cnxt_22k", "swin_22k", "maxvit_t", "eva02_t", "dino_vits16", "srm_b0"]
    fam_data = {}
    for f in fams:
        p = f"{fold_dir}/{f}/validation_logits.npz"
        if os.path.exists(p):
            fam_data[f] = np.load(p)

    views = [k for k in fam_data["cnxt_22k"].keys() if fam_data["cnxt_22k"][k].dtype.kind in ('f', 'i') and k != "targets"]
    
    # Check standalone performance of each model
    print(f"{'Model':<25} | {'Clean AUC':<10} | {'Stress Mean':<12} | {'Stress Min':<11} | {'Score':<8}")
    print("-" * 70)
    for f in fams:
        d = fam_data[f]
        cp = 1.0 / (1.0 + np.exp(-d["clean"].astype(np.float64)))
        c_auc = roc_auc_score(y_true, cp)
        s_aucs = [roc_auc_score(y_true, 1.0 / (1.0 + np.exp(-d[v].astype(np.float64)))) for v in views if v != "clean" and v in d]
        sm = float(np.mean(s_aucs))
        smin = float(np.min(s_aucs))
        score = 0.5 * (c_auc + sm)
        print(f"{f:<25} | {c_auc:<10.5f} | {sm:<12.5f} | {smin:<11.5f} | {score:<8.5f}")

    # Check new members: cnxt_base and cnxt_22k_sbi
    for new_m in ["cnxt_base", "cnxt_22k_sbi"]:
        p = f"{fold_dir}/{new_m}/validation_logits.npz"
        if os.path.exists(p):
            d = np.load(p)
            cp = 1.0 / (1.0 + np.exp(-d["clean"].astype(np.float64)))
            c_auc = roc_auc_score(y_true, cp)
            s_aucs = [roc_auc_score(y_true, 1.0 / (1.0 + np.exp(-d[v].astype(np.float64)))) for v in views if v != "clean" and v in d]
            sm = float(np.mean(s_aucs))
            smin = float(np.min(s_aucs))
            score = 0.5 * (c_auc + sm)
            print(f"{new_m + ' [NEW]':<25} | {c_auc:<10.5f} | {sm:<12.5f} | {smin:<11.5f} | {score:<8.5f}")

    print("-" * 70)
    
    # 1. Incumbent 6-family ensemble
    clean_preds = [1.0 / (1.0 + np.exp(-fam_data[f]["clean"].astype(np.float64))) for f in fams]
    ens6_clean_auc = roc_auc_score(y_true, np.mean(clean_preds, axis=0))
    ens6_stress_aucs = []
    for v in views:
        if v == "clean": continue
        vp = [1.0 / (1.0 + np.exp(-fam_data[f][v].astype(np.float64))) for f in fams if v in fam_data[f]]
        ens6_stress_aucs.append(roc_auc_score(y_true, np.mean(vp, axis=0)))
    ens6_sm = float(np.mean(ens6_stress_aucs))
    ens6_smin = float(np.min(ens6_stress_aucs))
    ens6_score = 0.5 * (ens6_clean_auc + ens6_sm)
    print(f"{'Incumbent 6-Family Suite':<25} | {ens6_clean_auc:<10.5f} | {ens6_sm:<12.5f} | {ens6_smin:<11.5f} | {ens6_score:<8.5f}")

    # 2. Upgraded Suite: Replace cnxt_22k with cnxt_base
    p_base = f"{fold_dir}/cnxt_base/validation_logits.npz"
    if os.path.exists(p_base):
        d_base = np.load(p_base)
        up_clean_preds = clean_preds.copy()
        up_clean_preds[0] = 1.0 / (1.0 + np.exp(-d_base["clean"].astype(np.float64)))
        up_clean_auc = roc_auc_score(y_true, np.mean(up_clean_preds, axis=0))
        up_stress_aucs = []
        for v in views:
            if v == "clean": continue
            vp = []
            for idx, f in enumerate(fams):
                if idx == 0:
                    vp.append(1.0 / (1.0 + np.exp(-d_base[v].astype(np.float64))))
                elif v in fam_data[f]:
                    vp.append(1.0 / (1.0 + np.exp(-fam_data[f][v].astype(np.float64))))
            up_stress_aucs.append(roc_auc_score(y_true, np.mean(vp, axis=0)))
        up_sm = float(np.mean(up_stress_aucs))
        up_smin = float(np.min(up_stress_aucs))
        up_score = 0.5 * (up_clean_auc + up_sm)
        print(f"{'Upgraded (CNXT-Base)':<25} | {up_clean_auc:<10.5f} | {up_sm:<12.5f} | {up_smin:<11.5f} | {up_score:<8.5f} (delta: {up_score - ens6_score:+.5f})")

    # 3. Super Suite: CNXT-Base + CNXT-SBI + 5 Families (7 members)
    p_sbi = f"{fold_dir}/cnxt_22k_sbi/validation_logits.npz"
    if os.path.exists(p_base) and os.path.exists(p_sbi):
        d_sbi = np.load(p_sbi)
        super_clean_preds = [
            1.0 / (1.0 + np.exp(-d_base["clean"].astype(np.float64))),
            1.0 / (1.0 + np.exp(-d_sbi["clean"].astype(np.float64))),
            1.0 / (1.0 + np.exp(-fam_data["swin_22k"]["clean"].astype(np.float64))),
            1.0 / (1.0 + np.exp(-fam_data["maxvit_t"]["clean"].astype(np.float64))),
            1.0 / (1.0 + np.exp(-fam_data["eva02_t"]["clean"].astype(np.float64))),
            1.0 / (1.0 + np.exp(-fam_data["dino_vits16"]["clean"].astype(np.float64))),
            1.0 / (1.0 + np.exp(-fam_data["srm_b0"]["clean"].astype(np.float64))),
        ]
        super_clean_auc = roc_auc_score(y_true, np.mean(super_clean_preds, axis=0))
        super_stress_aucs = []
        for v in views:
            if v == "clean": continue
            vp = [
                1.0 / (1.0 + np.exp(-d_base[v].astype(np.float64))),
                1.0 / (1.0 + np.exp(-d_sbi[v].astype(np.float64))),
                1.0 / (1.0 + np.exp(-fam_data["swin_22k"][v].astype(np.float64))),
                1.0 / (1.0 + np.exp(-fam_data["maxvit_t"][v].astype(np.float64))),
                1.0 / (1.0 + np.exp(-fam_data["eva02_t"][v].astype(np.float64))),
                1.0 / (1.0 + np.exp(-fam_data["dino_vits16"][v].astype(np.float64))),
                1.0 / (1.0 + np.exp(-fam_data["srm_b0"][v].astype(np.float64))),
            ]
            super_stress_aucs.append(roc_auc_score(y_true, np.mean(vp, axis=0)))
        super_sm = float(np.mean(super_stress_aucs))
        super_smin = float(np.min(super_stress_aucs))
        super_score = 0.5 * (super_clean_auc + super_sm)
        print(f"{'Super Suite (7-Member)':<25} | {super_clean_auc:<10.5f} | {super_sm:<12.5f} | {super_smin:<11.5f} | {super_score:<8.5f} (delta: {super_score - ens6_score:+.5f})")

eval_fold(0, "data/splits/split_v1.csv")
print("\n")
eval_fold(4, "data/splits/split_v1_fold4.csv")
