import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

split_df = pd.read_csv("data/splits/split_v1.csv")
val_df = split_df[split_df["split"] == "val"].reset_index(drop=True)
y_true = val_df["label"].values.astype(int)

# Load base model logits
base_data = np.load("results/fold0/cnxt_base_consistency/validation_logits.npz")
all_keys = list(base_data.keys())
views = [k for k in all_keys if base_data[k].dtype.kind in ('f', 'i') and k != "targets"]
print(f"Loaded {len(views)} numeric evaluation views")

clean_probs = 1.0 / (1.0 + np.exp(-base_data["clean"].astype(np.float64)))
clean_auc = roc_auc_score(y_true, clean_probs)

stress_aucs = {}
for v in views:
    if v == "clean":
        continue
    probs = 1.0 / (1.0 + np.exp(-base_data[v].astype(np.float64)))
    stress_aucs[v] = roc_auc_score(y_true, probs)

stress_mean = float(np.mean(list(stress_aucs.values())))
stress_min = float(np.min(list(stress_aucs.values())))
worst_view = min(stress_aucs, key=stress_aucs.get)

print("=" * 60)
print(f"ConvNeXt-Base Standalone Fold 0:")
print(f"  Clean AUC:       {clean_auc:.5f}")
print(f"  Stress Mean AUC: {stress_mean:.5f}")
print(f"  Stress Min AUC:  {stress_min:.5f} ({worst_view})")
print(f"  Combined Score:  {0.5 * (clean_auc + stress_mean):.5f}")
print("=" * 60)

# Incumbent ConvNeXt-Small 22k
small_data = np.load("outputs/ablations/v14/fold0/cnxt_22k/validation_logits.npz")
sm_clean_probs = 1.0 / (1.0 + np.exp(-small_data["clean"].astype(np.float64)))
sm_clean_auc = roc_auc_score(y_true, sm_clean_probs)
sm_stress_aucs = {v: roc_auc_score(y_true, 1.0 / (1.0 + np.exp(-small_data[v].astype(np.float64)))) for v in views if v != "clean" and v in small_data}
sm_stress_mean = float(np.mean(list(sm_stress_aucs.values())))
sm_stress_min = float(np.min(list(sm_stress_aucs.values())))

print(f"Incumbent ConvNeXt-Small 22k Standalone Fold 0:")
print(f"  Clean AUC:       {sm_clean_auc:.5f}")
print(f"  Stress Mean AUC: {sm_stress_mean:.5f}")
print(f"  Stress Min AUC:  {sm_stress_min:.5f}")
print(f"  Delta (Base - Small): Clean {clean_auc - sm_clean_auc:+.5f}, Mean {stress_mean - sm_stress_mean:+.5f}, Min {stress_min - sm_stress_min:+.5f}")
print("=" * 60)

# Also test SBI
sbi_data = np.load("outputs/ablations/v14/fold0/cnxt_22k_sbi/validation_logits.npz")
sbi_clean_probs = 1.0 / (1.0 + np.exp(-sbi_data["clean"].astype(np.float64)))
sbi_clean_auc = roc_auc_score(y_true, sbi_clean_probs)
sbi_stress_aucs = {v: roc_auc_score(y_true, 1.0 / (1.0 + np.exp(-sbi_data[v].astype(np.float64)))) for v in views if v != "clean" and v in sbi_data}
sbi_stress_mean = float(np.mean(list(sbi_stress_aucs.values())))
sbi_stress_min = float(np.min(list(sbi_stress_aucs.values())))
print(f"ConvNeXt-Small + SBI Standalone Fold 0:")
print(f"  Clean AUC:       {sbi_clean_auc:.5f}")
print(f"  Stress Mean AUC: {sbi_stress_mean:.5f}")
print(f"  Stress Min AUC:  {sbi_stress_min:.5f}")
print("=" * 60)

# 6-Family Blend
families = ["cnxt_22k", "swin_22k", "maxvit_t", "eva02_t", "dino_vits16", "srm_b0"]
fam_clean_preds = []
fam_stress_preds = {v: [] for v in views if v != "clean"}
for fam in families:
    p = f"outputs/ablations/v14/fold0/{fam}/validation_logits.npz"
    d = np.load(p)
    fam_clean_preds.append(1.0 / (1.0 + np.exp(-d["clean"].astype(np.float64))))
    for v in fam_stress_preds:
        if v in d:
            fam_stress_preds[v].append(1.0 / (1.0 + np.exp(-d[v].astype(np.float64))))

ens6_clean = np.mean(fam_clean_preds, axis=0)
ens6_clean_auc = roc_auc_score(y_true, ens6_clean)
ens6_stress_aucs = {v: roc_auc_score(y_true, np.mean(fam_stress_preds[v], axis=0)) for v in fam_stress_preds}
ens6_stress_mean = float(np.mean(list(ens6_stress_aucs.values())))
ens6_stress_min = float(np.min(list(ens6_stress_aucs.values())))

print(f"Incumbent 6-Family Ensemble Fold 0:")
print(f"  Clean AUC:       {ens6_clean_auc:.5f}")
print(f"  Stress Mean AUC: {ens6_stress_mean:.5f}")
print(f"  Stress Min AUC:  {ens6_stress_min:.5f}")

# Upgraded Ensemble: Replace Small with Base (or Base + SBI blend)
fam_base_clean = fam_clean_preds.copy()
# CNXT family is index 0: replace with 0.5 * cnxt_base + 0.5 * cnxt_sbi
fam_base_clean[0] = 0.5 * clean_probs + 0.5 * sbi_clean_probs
ens_up_clean = np.mean(fam_base_clean, axis=0)
ens_up_clean_auc = roc_auc_score(y_true, ens_up_clean)

fam_base_stress = {}
for v in fam_stress_preds:
    stress_p = fam_stress_preds[v].copy()
    base_v = 1.0 / (1.0 + np.exp(-base_data[v].astype(np.float64)))
    sbi_v = 1.0 / (1.0 + np.exp(-sbi_data[v].astype(np.float64)))
    stress_p[0] = 0.5 * base_v + 0.5 * sbi_v
    fam_base_stress[v] = roc_auc_score(y_true, np.mean(stress_p, axis=0))

ens_up_stress_mean = float(np.mean(list(fam_base_stress.values())))
ens_up_stress_min = float(np.min(list(fam_base_stress.values())))

print("=" * 60)
print(f"UPGRADED 6-Family Ensemble (with ConvNeXt-Base + SBI) Fold 0:")
print(f"  Clean AUC:       {ens_up_clean_auc:.5f} (delta: {ens_up_clean_auc - ens6_clean_auc:+.5f})")
print(f"  Stress Mean AUC: {ens_up_stress_mean:.5f} (delta: {ens_up_stress_mean - ens6_stress_mean:+.5f})")
print(f"  Stress Min AUC:  {ens_up_stress_min:.5f} (delta: {ens_up_stress_min - ens6_stress_min:+.5f})")
print(f"  Combined Score:  {0.5 * (ens_up_clean_auc + ens_up_stress_mean):.5f}")
print("=" * 60)
