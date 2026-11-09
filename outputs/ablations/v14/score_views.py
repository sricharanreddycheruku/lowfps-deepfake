"""Save image-aligned validation logits for the three established suites."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

root = Path.cwd()
sys.path.insert(0, str(root / "outputs/ablations/v13"))
from evaluate import predict
from augment import build_eval_transform
from ensemble import load_member
from robust_eval import EXTENDED_VARIANTS, STRESS_VARIANTS, VARIANTS, load_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="data/splits/split_v1.csv")
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.split, split="val")
    device = torch.device("cuda")
    model, checkpoint = load_member(args.run / "fixed.pt", device)
    variants = {**VARIANTS, **EXTENDED_VARIANTS, **STRESS_VARIANTS}
    logits = predict(model, rows, build_eval_transform(checkpoint["img_size"]),
                     variants, device)
    ids = np.array([row["image_id"] for row in rows])
    np.savez(args.run / "validation_logits.npz", image_ids=ids, **logits)


if __name__ == "__main__":
    main()
