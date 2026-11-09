"""Packs the 6-member v16 Foundation & Multi-Axis Vision Transformer Full-Data Ensemble."""

from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path.cwd() / "src"))
from augment import IMAGENET_MEAN, IMAGENET_STD


def main():
    base_dir = Path("results/v16_full")
    models_info = [
        {"name": "v16_full_s42_cnxt_22k", "arch": "convnext_small_22k", "ckpt": base_dir / "cnxt_22k_s42" / "fixed.pt"},
        {"name": "v16_full_s7_cnxt_22k", "arch": "convnext_small_22k", "ckpt": base_dir / "cnxt_22k_s7" / "fixed.pt"},
        {"name": "v16_full_s42_swin_22k", "arch": "swin_small_22k", "ckpt": base_dir / "swin_22k_s42" / "fixed.pt"},
        {"name": "v16_full_s7_swin_22k", "arch": "swin_small_22k", "ckpt": base_dir / "swin_22k_s7" / "fixed.pt"},
        {"name": "v16_full_s42_maxvit_t", "arch": "maxvit_t", "ckpt": base_dir / "maxvit_t_s42" / "fixed.pt"},
        {"name": "v16_full_s7_maxvit_t", "arch": "maxvit_t", "ckpt": base_dir / "maxvit_t_s7" / "fixed.pt"},
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

    out_path = Path("submission/v16_model_weights.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out_path)
    print(f"Wrote {out_path} ({out_path.stat().st_size / (1024 * 1024):.1f} MB)")


if __name__ == "__main__":
    main()
