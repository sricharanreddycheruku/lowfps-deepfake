"""Packs the 6-member v15 All-Consistent Full-Data Ensemble into a submission bundle."""

import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path.cwd() / "src"))
from augment import IMAGENET_MEAN, IMAGENET_STD


def main():
    base_dir = Path("results/v15_full")
    if not base_dir.exists():
        # Try finding it relative to repo
        for candidate in [Path.cwd() / "results" / "v15_full", Path(__file__).resolve().parents[3] / "results" / "v15_full"]:
            if candidate.exists():
                base_dir = candidate
                break

    models_info = [
        {"name": "v15_full_s42_resnet50", "arch": "resnet50", "ckpt": base_dir / "resnet50_s42" / "fixed.pt"},
        {"name": "v15_full_s7_resnet50", "arch": "resnet50", "ckpt": base_dir / "resnet50_s7" / "fixed.pt"},
        {"name": "v15_full_s42_cnxt_small", "arch": "convnext_small", "ckpt": base_dir / "cnxt_small_s42" / "fixed.pt"},
        {"name": "v15_full_s7_cnxt_small", "arch": "convnext_small", "ckpt": base_dir / "cnxt_small_s7" / "fixed.pt"},
        {"name": "v15_full_s42_swin_t", "arch": "swin_t", "ckpt": base_dir / "swin_t_s42" / "fixed.pt"},
        {"name": "v15_full_s7_swin_t", "arch": "swin_t", "ckpt": base_dir / "swin_t_s7" / "fixed.pt"},
    ]

    members = []
    weight = 1.0 / len(models_info)
    for m in models_info:
        if not m["ckpt"].exists():
            print(f"Error: checkpoint {m['ckpt']} does not exist.")
            return
        ckpt = torch.load(m["ckpt"], map_location="cpu", weights_only=False)
        state = {k: v.detach().cpu().contiguous() for k, v in ckpt["state_dict"].items()}
        members.append({
            "name": m["name"],
            "arch": ckpt["arch"],
            "img_size": int(ckpt.get("img_size", 224)),
            "weight": float(weight),
            "state_dict": state,
        })
        print(f"Loaded {m['name']} ({ckpt['arch']}) from {m['ckpt']}")

    bundle = {
        "members": members,
        "fusion": "mean_logit",
        "tta": True,
        "threshold": 0.5,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "image_size": 224,
    }

    out_path = Path("submission/v15_model_weights.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Successfully packaged bundle to {out_path} ({size_mb:.1f} MB, {len(members)} members).")


if __name__ == "__main__":
    main()
