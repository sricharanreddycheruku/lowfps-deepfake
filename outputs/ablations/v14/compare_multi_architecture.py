import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def evaluate_fold(fold_name, split_csv):
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
    stress_views = [v for v in views if v != "clean"]

    def ensemble(model_keys, weights=None):
        if weights is None:
            weights = [1.0 / len(model_keys)] * len(model_keys)
        out = {}
        for v in views:
            stacked = np.stack([models[m][v].astype(float) for m in model_keys], axis=0)
            w = np.array(weights).reshape(-1, 1)
            out[v] = np.sum(stacked * w, axis=0)
        return out

    combos = {
        "1. Incumbent (ResNet+Cnxt1k+Swin1k)": np.load(base_dir / "consistency" / "validation_logits.npz"), # placeholder
        "2. Foundation Trio (ResNet50+Cnxt22k+Swin22k)": ensemble(["resnet50", "cnxt_22k", "swin_22k"]),
        "3. Foundation Quad (+ MaxViT-T)": ensemble(["resnet50", "cnxt_22k", "swin_22k", "maxvit_t"]),
        "4. Trio Pure Modern (Cnxt22k+Swin22k+MaxViT-T)": ensemble(["cnxt_22k", "swin_22k", "maxvit_t"]),
        "5. Pure Modern + DINO": ensemble(["cnxt_22k", "swin_22k", "maxvit_t", "dino"]),
        "6. Pure Modern + SRM": ensemble(["cnxt_22k", "swin_22k", "maxvit_t", "srm"]),
        "7. Pure Modern + DINO + SRM": ensemble(["cnxt_22k", "swin_22k", "maxvit_t", "dino", "srm"]),
        "8. All 6-Family Blend": ensemble(["resnet50", "cnxt_22k", "swin_22k", "maxvit_t", "dino", "srm"]),
        "Standalone MaxViT-T": {v: models["maxvit_t"][v].astype(float) for v in views},
        "Standalone DINO": {v: models["dino"][v].astype(float) for v in views},
        "Standalone SRM": {v: models["srm"][v].astype(float) for v in views},
    }

    results = {}
    print(f"\n==================== {fold_name.upper()} RESULTS ====================")
    print(f"{'Configuration':<45} | {'Clean AUC':<10} | {'Pixel8 AUC':<10} | {'Stress Mean':<12} | {'Stress Worst':<12}")
    print("-" * 96)

    for name, logits_dict in combos.items():
        aucs = {}
        for v in views:
            probs = 1.0 / (1.0 + np.exp(-logits_dict[v]))
            aucs[v] = float(roc_auc_score(y_true, probs))
        s_aucs = [aucs[v] for v in stress_views]
        clean = aucs["clean"]
        pix8 = aucs.get("pixelate8", 0.0)
        s_mean = float(np.mean(s_aucs))
        s_worst = float(np.min(s_aucs))
        worst_name = [v for v in stress_views if aucs[v] == s_worst][0]
        results[name] = {
            "clean": clean,
            "pixelate8": pix8,
            "stress_mean": s_mean,
            "stress_worst": s_worst,
            "worst_name": worst_name,
        }
        print(f"{name:<45} | {clean:.5f}    | {pix8:.5f}    | {s_mean:.5f}      | {s_worst:.5f} ({worst_name})")

    return results

def main():
    res_f0 = evaluate_fold("fold0", "data/splits/split_v1.csv")
    res_f4 = evaluate_fold("fold4", "data/splits/split_v1_fold4.csv")

    out_file = Path("outputs/ablations/v14/multi_architecture_comparison.json")
    out_file.write_text(json.dumps({"fold0": res_f0, "fold4": res_f4}, indent=2), encoding="utf-8")
    print(f"\nWrote full comparison to {out_file}")

if __name__ == "__main__":
    main()
