"""Compare the two fixed checkpoints on fold 0 with paired group resampling."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path.cwd() / "src"))
from augment import build_eval_transform
from ensemble import forward_logits, load_member
from ensemble_pool import fast_auc
from robust_eval import EXTENDED_VARIANTS, VARIANTS, VariantDataset, load_rows


def predict(model, rows, transform, variants, device):
    scores = {}
    for name, degrade in variants.items():
        dataset = VariantDataset(rows, degrade, transform)
        loader = DataLoader(dataset, batch_size=64, num_workers=4)
        batches = [
            (forward_logits(model, images, device, False)
             + forward_logits(model, images, device, True)) / 2
            for images, _ in loader
        ]
        scores[name] = np.concatenate(batches)
        print(f"Scored {name}", flush=True)
    return scores


def metrics(labels, scores, indices):
    auc = {name: fast_auc(labels[indices], score[indices])
           for name, score in scores.items()}
    standard = [auc[name] for name in VARIANTS]
    return np.array([auc["clean"], np.mean(standard), min(standard), auc["pixelate8"]])


def compare(labels, scores, group_indices, candidate):
    names = ("clean", "mean_23", "worst_23", "pixelate8")
    full = np.arange(len(labels))
    observed = {arm: metrics(labels, pred, full) for arm, pred in scores.items()}
    rng = np.random.default_rng(0)
    gains = []
    for _ in range(1000):
        draw = rng.integers(len(group_indices), size=len(group_indices))
        indices = np.concatenate([group_indices[i] for i in draw])
        if np.unique(labels[indices]).size == 2:
            gains.append(metrics(labels, scores[candidate], indices)
                         - metrics(labels, scores["bce"], indices))
    return {
        **{arm: dict(zip(names, values.tolist())) for arm, values in observed.items()},
        "gain_vs_bce": dict(zip(names, (observed[candidate] - observed["bce"]).tolist())),
        "gain_95pct_ci": dict(zip(names, np.percentile(gains, [2.5, 97.5], axis=0).T.tolist())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("gce", "relabel"), default="gce")
    args = parser.parse_args()
    rows = load_rows("data/splits/split_v1.csv", split="val")
    labels = np.array([int(row["label"]) for row in rows])
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row["group_id"], []).append(i)

    variants = {**VARIANTS, "pixelate8": EXTENDED_VARIANTS["pixelate8"]}
    device = torch.device("cuda")
    scores = {}
    image_ids = np.array([row["image_id"] for row in rows])
    for arm in ("bce", args.candidate):
        cache = Path(f"results/{arm}/validation_logits.npz")
        if arm == "bce" and args.candidate == "relabel":
            with np.load(cache) as data:
                np.testing.assert_array_equal(data["image_ids"], image_ids)
                scores[arm] = {name: data[name] for name in variants}
            continue
        model, checkpoint = load_member(f"results/{arm}/fixed.pt", device)
        scores[arm] = predict(model, rows, build_eval_transform(checkpoint["img_size"]),
                              variants, device)
        np.savez(cache, image_ids=image_ids, **scores[arm])
        del model

    result = compare(labels, scores, list(groups.values()), args.candidate)
    result["validation_images"] = len(rows)
    result["validation_groups"] = len(groups)
    Path(f"results/comparison_{args.candidate}.json").write_text(json.dumps(result, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
