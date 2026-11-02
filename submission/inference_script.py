"""Inference script for the Deepfake ML Hackathon submission.

Loads the bundled ensemble (ResNet-50, ConvNeXt-Small, Swin-T), runs every
image in --test_dir through all members with a horizontal-flip TTA, fuses the
logits as stored in the bundle and writes image_id,label,confidence.

    python inference_script.py \
        --test_dir ./test/images/ \
        --weights ./model_weights.pt \
        --output predictions.csv

Optional: --batch_size (default 128), --threshold (default: the value stored
in the bundle), --num_workers (default min(4, cpu count)).

The weights file holds plain tensors, so no pretrained weights are downloaded
and nothing outside --weights is read. Preprocessing is Resize to the member's
input size, ToTensor and ImageNet normalisation, the same as src/augment.py.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def build_model(arch, dropout=0.3):
    """Same architectures as src/models.py, without pretrained weights.
    Every model maps (N, 3, H, W) to (N, 1) raw logits."""
    def head(in_features):
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))

    if arch in ("efficientnet_b0", "efficientnet_v2_s"):
        model = getattr(models, arch)(weights=None)
        model.classifier = head(model.classifier[-1].in_features)
    elif arch in ("convnext_tiny", "convnext_small"):
        model = getattr(models, arch)(weights=None)
        model.classifier[-1] = head(model.classifier[-1].in_features)
    elif arch == "swin_t":
        model = models.swin_t(weights=None)
        model.head = head(model.head.in_features)
    elif arch == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = head(model.fc.in_features)
    else:
        raise ValueError(f"unsupported arch in bundle: {arch}")
    return model


class TestDataset(Dataset):
    """Flat directory of images; yields one tensor per member input size."""

    def __init__(self, paths, sizes, mean, std):
        self.paths = paths
        self.transforms = {
            size: transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]) for size in sorted(set(sizes))
        }

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return {size: t(img) for size, t in self.transforms.items()}


def list_images(test_dir):
    names = [n for n in os.listdir(test_dir) if n.lower().endswith(IMAGE_EXTENSIONS)]
    return [os.path.join(test_dir, n) for n in sorted(names)]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fuse(logits, rule, weights):
    """logits: (members, N) float array. Returns P(fake)."""
    if rule == "mean_logit":
        return sigmoid(logits.mean(axis=0))
    if rule == "mean_prob":
        return sigmoid(logits).mean(axis=0)
    if rule == "weighted_logit":
        return sigmoid((np.asarray(weights)[:, None] * logits).sum(axis=0))
    raise ValueError(f"unknown fusion rule: {rule}")


def load_bundle(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def load_members(bundle, device):
    members = []
    for m in bundle["members"]:
        model = build_model(m["arch"])
        model.load_state_dict(m["state_dict"])
        model = model.to(device, memory_format=torch.channels_last).eval()
        members.append((model, int(m["img_size"])))
        print(f"[INFO] loaded {m['name']} ({m['arch']}, {m['img_size']}px)")
    return members


@torch.inference_mode()
def forward(model, images, device, flip):
    images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
    if flip:
        images = torch.flip(images, dims=[3])
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=device.type == "cuda"):
        out = model(images)
    return out.squeeze(1).float().cpu().numpy()


def predict(bundle, paths, device, batch_size, num_workers):
    members = load_members(bundle, device)
    sizes = [size for _, size in members]
    dataset = TestDataset(paths, sizes, bundle["mean"], bundle["std"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == "cuda")
    flips = (False, True) if bundle["tta"] else (False,)

    per_member = [[] for _ in members]
    print(f"[INFO] {len(paths)} images, batch {batch_size}, tta {bundle['tta']}, "
          f"fusion {bundle['fusion']}, device {device}")
    for batch in loader:
        for i, (model, size) in enumerate(members):
            views = [forward(model, batch[size], device, flip) for flip in flips]
            per_member[i].append(np.mean(views, axis=0))
    logits = np.stack([np.concatenate(chunks) for chunks in per_member])
    weights = [m["weight"] for m in bundle["members"]]
    return fuse(logits, bundle["fusion"], weights)


def main():
    parser = argparse.ArgumentParser(description="Deepfake ML Hackathon inference")
    parser.add_argument("--test_dir", required=True, help="directory with the test images")
    parser.add_argument("--weights", required=True, help="path to model_weights.pt")
    parser.add_argument("--output", default="predictions.csv", help="output CSV path")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=None,
                        help="label threshold; defaults to the value in the bundle")
    parser.add_argument("--num_workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()

    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    paths = list_images(args.test_dir)
    if not paths:
        sys.exit(f"no images found in {args.test_dir}")
    ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    assert len(set(ids)) == len(ids), "duplicate image ids (same stem, different extension)"

    bundle = load_bundle(args.weights)
    threshold = bundle["threshold"] if args.threshold is None else args.threshold
    probs = predict(bundle, paths, device, args.batch_size, args.num_workers)
    probs = np.round(np.clip(probs, 0.0, 1.0), 6)
    labels = (probs >= threshold).astype(int)

    assert len(probs) == len(ids), "row count differs from image count"
    assert not np.isnan(probs).any(), "NaN in confidence"

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "label", "confidence"])
        for image_id, label, p in zip(ids, labels, probs):
            writer.writerow([image_id, int(label), f"{p:.6f}"])

    elapsed = time.time() - start
    print(f"[DONE] wrote {args.output}")
    print(f"  images: {len(ids)}, predicted fake: {int(labels.sum())} "
          f"({labels.mean() * 100:.1f}%), mean confidence {probs.mean():.4f}")
    print(f"  elapsed: {elapsed:.1f} s")
    if device.type == "cuda":
        print(f"  peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
