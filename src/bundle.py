"""Packs the chosen ensemble into a single weights file for the submission.

Reads the *_chosen.json written by src/ensemble.py (members, fusion rule,
weights, TTA flag, threshold) and the members' best.pt checkpoints (EMA
weights), and writes one dict:

    members   list of {name, arch, img_size, weight, state_dict}
    fusion    "mean_logit" | "mean_prob" | "weighted_logit"
    tta       bool, horizontal-flip TTA (logits of both views are averaged)
    threshold decision threshold for the 0/1 label
    mean, std ImageNet normalisation constants
    image_size  input side of the first member (members carry their own)

Only tensors, lists, numbers and strings are stored, so the file also loads
with torch.load(..., weights_only=True). State dicts stay in float32: the
whole bundle is well under the size limit, and it keeps the bundled
predictions identical to the ones src/ensemble.py scored on val.

With --verify the bundle is reloaded from disk, run on the clean val split
through the same code path the inference script uses, and compared with the
fused probabilities that src/ensemble.py wrote (max abs diff must be < 1e-4).

Usage:
    python src/bundle.py --chosen outputs/ablations/ensemble_chosen.json \
        --out submission/model_weights.pt --verify outputs/ablations/ensemble_val_preds.csv
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from augment import IMAGENET_MEAN, IMAGENET_STD, build_eval_transform
from ensemble import forward_logits, fuse
from models import build_model
from robust_eval import VARIANTS, VariantDataset, load_rows


def build_bundle(chosen, fp16=False):
    members = []
    for name, path, weight in zip(chosen["members"], chosen["ckpts"], chosen["weights"]):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = {k: v.detach().cpu().contiguous() for k, v in ckpt["state_dict"].items()}
        if fp16:
            # half-precision storage; load_state_dict casts back to fp32
            state = {k: v.half() if v.is_floating_point() else v for k, v in state.items()}
        members.append({
            "name": name,
            "arch": ckpt["arch"],
            "img_size": int(ckpt.get("img_size", 224)),
            "weight": float(weight),
            "state_dict": state,
        })
    return {
        "members": members,
        "fusion": chosen["fusion"],
        "tta": bool(chosen["tta"]),
        "threshold": float(chosen["threshold"]),
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "image_size": members[0]["img_size"],
    }


def predict_bundle(bundle, rows, device, bs=64, num_workers=4):
    """Fused P(fake) for the given val rows, straight from a loaded bundle."""
    per_member = []
    for m in bundle["members"]:
        model = build_model(m["arch"], pretrained=False)
        model.load_state_dict(m["state_dict"])
        model = model.to(device, memory_format=torch.channels_last).eval()
        ds = VariantDataset(rows, VARIANTS["clean"], build_eval_transform(m["img_size"]))
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers)
        views = []
        for flip in ((False, True) if bundle["tta"] else (False,)):
            views.append(np.concatenate(
                [forward_logits(model, images, device, flip) for images, _ in loader]))
        per_member.append(np.mean(views, axis=0))
        del model
    weights = [m["weight"] for m in bundle["members"]]
    return fuse(np.stack(per_member), bundle["fusion"], weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chosen", default="outputs/ablations/ensemble_chosen.json")
    parser.add_argument("--out", default="submission/model_weights.pt")
    parser.add_argument("--split_csv", default="data/splits/split_v1.csv")
    parser.add_argument("--verify", default=None,
                        help="ensemble.py *_val_preds.csv to compare the bundle against")
    parser.add_argument("--bs", type=int, default=64,
                        help="verification batch size; use the one ensemble.py ran with, "
                             "fp16 kernels differ slightly between batch sizes")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--fp16", action="store_true",
                        help="store floating-point weights in half precision (halves the file)")
    parser.add_argument("--max_diff", type=float, default=1e-4,
                        help="largest allowed difference to the --verify predictions")
    args = parser.parse_args()

    with open(args.chosen, encoding="utf-8") as f:
        chosen = json.load(f)

    bundle = build_bundle(chosen, args.fp16)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(bundle, args.out)
    size_mb = os.path.getsize(args.out) / 2**20
    print(f"wrote {args.out}: {size_mb:.1f} MB, members "
          f"{[(m['name'], m['arch'], m['img_size']) for m in bundle['members']]}, "
          f"fusion {bundle['fusion']}, tta {bundle['tta']}, threshold {bundle['threshold']:.4f}")
    assert size_mb < 5000, "bundle is larger than the 5 GB limit"

    # both load modes must work
    torch.load(args.out, map_location="cpu", weights_only=True)
    reloaded = torch.load(args.out, map_location="cpu", weights_only=False)
    print("loads with weights_only=True and weights_only=False")

    if args.verify:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rows = load_rows(args.split_csv, split="val")
        probs = predict_bundle(reloaded, rows, device, bs=args.bs, num_workers=args.num_workers)
        with open(args.verify, newline="", encoding="utf-8") as f:
            ref = list(csv.DictReader(f))
        assert [r["image_id"] for r in ref] == [r["image_id"] for r in rows], "row order differs"
        ref_probs = np.array([float(r["confidence"]) for r in ref])
        diff = float(np.abs(probs - ref_probs).max())
        print(f"bundle vs ensemble.py on val: max abs diff {diff:.2e}")
        assert diff < args.max_diff, "bundle predictions differ from ensemble.py"


if __name__ == "__main__":
    main()
