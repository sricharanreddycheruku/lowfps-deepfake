import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

split_df = pd.read_csv("data/splits/split_v1.csv")
val_df = split_df[split_df["split"] == "val"].reset_index(drop=True)
y_true = val_df["label"].values

sbi_data = np.load("outputs/ablations/v14/fold0/cnxt_22k_sbi/validation_logits.npz")
views = list(sbi_data.keys())
print(f"Total views evaluated: {len(views)}")

clean_logits = sbi_data["clean"]
clean_probs = 1.0 / (1.0 + np.exp(-clean_logits))
clean_auc = roc_auc_score(y_true, clean_probs)

stress_aucs = {}
for v in views:
    if v == "clean":
        continue
    probs = 1.0 / (1.0 + np.exp(-sbi_data[v]))
    stress_aucs[v] = roc_auc_score(y_true, probs)

stress_mean = float(np.mean(list(stress_aucs.values())))
stress_min = float(np.min(list(stress_aucs.values())))
worst_view = min(stress_aucs, key=stress_aucs.get)

print(f"SBI convnext_small_22k Standalone Fold 0:")
print(f"  Clean AUC: {clean_auc:.5f}")
print(f"  Stress Mean AUC: {stress_mean:.5f}")
print(f"  Stress Min AUC: {stress_min:.5f} ({worst_view})")

# Baseline convnext_small_22k without SBI
base_data = np.load("outputs/ablations/v14/fold0/cnxt_22k/validation_logits.npz")
base_clean_probs = 1.0 / (1.0 + np.exp(-base_data["clean"]))
base_clean_auc = roc_auc_score(y_true, base_clean_probs)
base_stress_aucs = {v: roc_auc_score(y_true, 1.0 / (1.0 + np.exp(-base_data[v]))) for v in views if v != "clean"}
base_stress_mean = float(np.mean(list(base_stress_aucs.values())))
base_stress_min = float(np.min(list(base_stress_aucs.values())))

print(f"\nIncumbent Baseline convnext_small_22k (without SBI) Fold 0:")
print(f"  Clean AUC: {base_clean_auc:.5f}")
print(f"  Stress Mean AUC: {base_stress_mean:.5f}")
print(f"  Stress Min AUC: {base_stress_min:.5f}")

# Now test blend of 6 families + SBI:
# 6 families: cnxt_22k, swin_22k, maxvit_t, eva02_t, dino_vits16, srm_b0
families = ["cnxt_22k", "swin_22k", "maxvit_t", "eva02_t", "dino_vits16", "srm_b0"]
fam_clean_preds = []
fam_stress_preds = {v: [] for v in views if v != "clean"}

for fam in families:
    p = f"outputs/ablations/v14/fold0/{fam}/validation_logits.npz"
    d = np.load(p)
    fam_clean_preds.append(1.0 / (1.0 + np.exp(-d["clean"])))
    for v in fam_stress_preds:
        fam_stress_preds[v].append(1.0 / (1.0 + np.exp(-d[v])))

ens6_clean = np.mean(fam_clean_preds, axis=0)
ens6_clean_auc = roc_auc_score(y_true, ens6_clean)
ens6_stress_aucs = {v: roc_auc_score(y_true, np.mean(fam_stress_preds[v], axis=0)) for v in fam_stress_preds}
ens6_stress_mean = float(np.mean(list(ens6_stress_aucs.values())))
ens6_stress_min = float(np.min(list(ens6_stress_aucs.values())))

print(f"\nIncumbent 6-Family Blend Fold 0:")
print(f"  Clean AUC: {ens6_clean_auc:.5f}")
print(f"  Stress Mean AUC: {ens6_stress_mean:.5f}")
print(f"  Stress Min AUC: {ens6_stress_min:.5f}")

# Blend with SBI replacing or added to cnxt
# Add SBI as an explicit 7th member (or replace cnxt with average of cnxt and cnxt_sbi)
fam_plus_sbi_clean = fam_clean_preds.copy()
# Replace cnxt with blend of cnxt and cnxt_sbi
fam_plus_sbi_clean[0] = 0.5 * fam_clean_preds[0] + 0.5 * clean_probs
ens7_clean = np.mean(fam_plus_sbi_clean, axis=0)
ens7_clean_auc = roc_auc_score(y_true, ens7_clean)

fam_plus_sbi_stress = {}
for v in fam_stress_preds:
    stress_p = fam_stress_preds[v].copy()
    stress_p[0] = 0.5 * stress_p[0] + 0.5 * (1.0 / (1.0 + np.exp(-sbi_data[v])))
    fam_plus_sbi_stress[v] = roc_auc_score(y_true, np.mean(stress_p, axis=0))

ens7_stress_mean = float(np.mean(list(fam_plus_sbi_stress.values())))
ens7_stress_min = float(np.min(list(fam_plus_sbi_stress.values())))

print(f"\n6-Family Suite with CNXT+SBI Blended Fold 0:")
print(f"  Clean AUC: {ens7_clean_auc:.5f} (delta: {ens7_clean_auc - ens6_clean_auc:+.5f})")
print(f"  Stress Mean AUC: {ens7_stress_mean:.5f} (delta: {ens7_stress_mean - ens6_stress_mean:+.5f})")
print(f"  Stress Min AUC: {ens7_stress_min:.5f} (delta: {ens7_stress_min - ens6_stress_min:+.5f})")
