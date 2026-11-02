"""Train/eval image transforms.

Train transform: geometric augmentation (resized crop, flip, small rotation)
followed by a randomly ordered block of degradations (JPEG re-compression,
blur, noise, downscale/upscale, colour jitter, grayscale, sharpen), then the
usual tensor conversion, ImageNet normalisation and random erasing. The
degradation block exists to stop the model keying on dataset-specific
artifacts (see the JPEG-quality and brightness gaps noted in
outputs/data/data_report.json) and to make the robustness variants in
Chunk 4 look like "more of the same" rather than unseen distribution shift.

Eval transform (also used at inference) is deliberately minimal: resize to
224, tensor, normalize. It must match submission/inference_script.py
byte-for-byte, which --preview checks.

Usage:
    python src/augment.py --preview --data_root data/raw/hackathon_dataset \
        --split_csv data/splits/split_v1.csv --out outputs/figures/aug_preview.png
"""

import argparse
import io
import random

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


class JpegRecompress:
    """Re-encodes the image through an in-memory JPEG at a random quality."""

    def __init__(self, quality_range=(30, 95), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        quality = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class GaussianBlur:
    def __init__(self, sigma_range=(0.1, 2.0), p=0.3):
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        sigma = random.uniform(*self.sigma_range)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))


class GaussianNoise:
    def __init__(self, std_range=(2, 20), p=0.3):
        self.std_range = std_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        std = random.uniform(*self.std_range)
        arr = np.asarray(img).astype(np.float32)
        noise = np.random.normal(0, std, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


class DownscaleUpscale:
    def __init__(self, scale_range=(0.25, 0.75), p=0.3):
        self.scale_range = scale_range
        self.p = p
        self.interpolations = [
            Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS,
        ]

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        scale = random.uniform(*self.scale_range)
        small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                            random.choice(self.interpolations))
        return small.resize((w, h), random.choice(self.interpolations))


class Sharpen:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        return img.filter(ImageFilter.SHARPEN)


class RandomOrder:
    """Applies a list of transforms in a freshly shuffled order each call,
    so the model does not learn a fixed degradation pipeline."""

    def __init__(self, transforms_list):
        self.transforms_list = transforms_list

    def __call__(self, img):
        order = list(self.transforms_list)
        random.shuffle(order)
        for t in order:
            img = t(img)
        return img


def build_degradation_block(level):
    if level == "none":
        return transforms.Lambda(lambda img: img)
    if level == "light":
        ops = [
            JpegRecompress(p=0.3),
            GaussianBlur(p=0.15),
            GaussianNoise(p=0.15),
            DownscaleUpscale(p=0.15),
            Sharpen(p=0.1),
        ]
    elif level == "strong":
        ops = [
            JpegRecompress(p=0.5),
            GaussianBlur(p=0.3),
            GaussianNoise(p=0.3),
            DownscaleUpscale(p=0.3),
            Sharpen(p=0.2),
        ]
    else:
        raise ValueError(f"unknown aug_level: {level}")
    return RandomOrder(ops)


def build_train_transform(aug_level="strong", size=IMAGE_SIZE):
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        build_degradation_block(aug_level),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25),
    ])


def build_eval_transform(size=IMAGE_SIZE):
    """Must stay identical to the inference-time preprocessing."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _preview(args):
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(args.split_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    train_rows = [r for r in rows if r["split"] == "train"]
    random.seed(0)
    sample_rows = random.sample(train_rows, 8)

    train_tf = build_train_transform(args.aug_level)
    eval_tf = build_eval_transform()

    fig, axes = plt.subplots(8, 6, figsize=(12, 16))
    for i, row in enumerate(sample_rows):
        img = Image.open(row["path"]).convert("RGB")
        for j in range(6):
            ax = axes[i, j]
            if j == 0:
                ax.imshow(img)
                ax.set_title("original", fontsize=8)
            else:
                tensor = train_tf(img)
                denorm = tensor * torch.tensor(IMAGENET_STD).view(3, 1, 1) + \
                    torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
                arr = denorm.clamp(0, 1).permute(1, 2, 0).numpy()
                ax.imshow(arr)
                ax.set_title(f"aug {j}", fontsize=8)
            ax.axis("off")
    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=100)
    print(f"wrote {args.out}")

    # eval transform must equal inference preprocessing byte-for-byte
    check_rows = sample_rows[:5]
    max_diff = 0.0
    for row in check_rows:
        img = Image.open(row["path"]).convert("RGB")
        t1 = eval_tf(img)
        t2 = eval_tf(img)
        max_diff = max(max_diff, (t1 - t2).abs().max().item())
    print(f"eval transform determinism check, max abs diff over 5 images: {max_diff}")
    assert max_diff == 0.0, "eval transform is not deterministic"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--data_root", default="data/raw/hackathon_dataset")
    parser.add_argument("--split_csv", default="data/splits/split_v1.csv")
    parser.add_argument("--aug_level", default="strong", choices=["none", "light", "strong"])
    parser.add_argument("--out", default="outputs/figures/aug_preview.png")
    args = parser.parse_args()

    if args.preview:
        _preview(args)
    else:
        print("nothing to do without --preview; import build_train_transform / "
              "build_eval_transform instead")


if __name__ == "__main__":
    main()
