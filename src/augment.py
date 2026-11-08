"""Train/eval image transforms.

Train transform: geometric augmentation (resized crop, flip, small rotation)
followed by a randomly ordered block of degradations (JPEG re-compression,
blur, noise, downscale/upscale, colour jitter, grayscale, sharpen), then the
usual tensor conversion, ImageNet normalisation and random erasing. The
degradation block exists to stop the model keying on dataset-specific
artifacts (see the JPEG-quality and brightness gaps noted in
outputs/data/data_report.json) and to make the robustness variants in
the robustness suite look like "more of the same" rather than unseen distribution shift.

SelfBlend turns a real image into a pseudo-fake by blending it with an
altered copy of itself inside a soft face-shaped mask, following the idea of
Self-Blended Images (Shiohara and Yamasaki, CVPR 2022). The faces are
already aligned, so the mask is a jittered ellipse around the face centre
instead of a landmark hull. train.py applies it to real training images only
(label 1 afterwards), before the train transform.

Eval transform (also used at inference) is deliberately minimal: resize to
224, tensor, normalize. It must match submission/inference_script.py
byte-for-byte, which --preview checks.

Usage:
    python src/augment.py --preview --data_root data/raw/hackathon_dataset \
        --split_csv data/splits/split_v1.csv --out outputs/figures/aug_preview.png
    python src/augment.py --preview --sbi   # writes outputs/figures/aug_preview_sbi.png
"""

import argparse
import io
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
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


class Mosaic:
    """Pixelates the whole image into square blocks, covering block sizes
    beyond what DownscaleUpscale reaches."""

    def __init__(self, block_range=(4, 16), p=0.15):
        self.block_range = block_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        block = random.randint(*self.block_range)
        small = img.resize((max(1, w // block), max(1, h // block)), Image.NEAREST)
        return small.resize((w, h), Image.NEAREST)


class HeavyDegrade:
    """Strong blur or a very low resolution round trip, past the ranges of
    GaussianBlur and DownscaleUpscale."""

    def __init__(self, sigma_range=(2.0, 5.0), scale_range=(0.08, 0.25), p=0.15):
        self.sigma_range = sigma_range
        self.scale_range = scale_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        if random.random() < 0.5:
            return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.sigma_range)))
        w, h = img.size
        scale = random.uniform(*self.scale_range)
        small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


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


# fixed facial regions as (x0, y0, x1, y1) fractions of the aligned face:
# eye band, nose, mouth, left cheek, right cheek
FACE_BOXES = [
    (0.15, 0.30, 0.85, 0.50),
    (0.35, 0.45, 0.65, 0.70),
    (0.28, 0.65, 0.72, 0.85),
    (0.10, 0.45, 0.35, 0.75),
    (0.65, 0.45, 0.90, 0.75),
]
_FILL = tuple(int(round(m * 255)) for m in IMAGENET_MEAN)


class FaceCut:
    """Blanks one of the fixed facial regions with the dataset mean colour."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        x0, y0, x1, y1 = random.choice(FACE_BOXES)
        img = img.copy()
        img.paste(_FILL, (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
        return img


class GridMask:
    """Blanks a square grid of cells (unit 64-128 px, keep ratio 0.6) at a
    random offset."""

    def __init__(self, p=0.5, unit_range=(64, 128), keep=0.6):
        self.p = p
        self.unit_range = unit_range
        self.keep = keep

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        unit = random.randint(*self.unit_range)
        drop = max(1, int(round(unit * (1 - self.keep))))
        ox, oy = random.randint(0, unit - 1), random.randint(0, unit - 1)
        img = img.copy()
        for y in range(oy - unit, h, unit):
            for x in range(ox - unit, w, unit):
                img.paste(_FILL, (max(x, 0), max(y, 0), min(x + drop, w), min(y + drop, h)))
        return img


class SelfBlend:
    """Blends an image with a colour/frequency/geometry-altered copy of
    itself inside a soft elliptical face mask.

    Source alterations: per-channel colour shift, saturation, brightness,
    contrast, then either a sharpen or a blur, then a small scale and shift.
    Mask: an ellipse around the aligned face centre (random size, centre,
    tilt) whose radius is warped by low-frequency sinusoids, Gaussian-blurred
    and scaled by a random blend ratio.
    """

    def __init__(self, rng=None):
        self.rng = rng or np.random

    def _alter(self, img):
        r = self.rng
        arr = np.asarray(img).astype(np.float32)
        arr = np.clip(arr + r.uniform(-20, 20, size=3), 0, 255).astype(np.uint8)
        out = Image.fromarray(arr)
        out = ImageEnhance.Color(out).enhance(r.uniform(0.8, 1.2))
        out = ImageEnhance.Brightness(out).enhance(r.uniform(0.85, 1.15))
        out = ImageEnhance.Contrast(out).enhance(r.uniform(0.85, 1.15))
        if r.rand() < 0.5:
            out = out.filter(ImageFilter.SHARPEN)
        else:
            out = out.filter(ImageFilter.GaussianBlur(radius=r.uniform(0.3, 1.2)))
        w, h = out.size
        scale = r.uniform(0.95, 1.05)
        tx, ty = r.uniform(-0.03, 0.03) * w, r.uniform(-0.03, 0.03) * h
        # inverse affine: output (x, y) samples input ((x - c) / s + c - t)
        a = 1.0 / scale
        cx, cy = w / 2, h / 2
        coeffs = (a, 0, cx - a * cx - tx * a, 0, a, cy - a * cy - ty * a)
        moved = out.transform((w, h), Image.AFFINE, coeffs, resample=Image.BILINEAR)
        # pixels the shift pulls in from outside the frame keep the unshifted copy
        valid = Image.new("L", (w, h), 255).transform((w, h), Image.AFFINE, coeffs,
                                                       resample=Image.NEAREST)
        return Image.composite(moved, out, valid)

    def _mask(self, w, h):
        r = self.rng
        cx = w * (0.5 + r.uniform(-0.04, 0.04))
        cy = h * (0.55 + r.uniform(-0.04, 0.04))
        rx = w * r.uniform(0.25, 0.40)
        ry = h * r.uniform(0.32, 0.48)
        tilt = np.deg2rad(r.uniform(-15, 15))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dx, dy = xx - cx, yy - cy
        u = dx * np.cos(tilt) + dy * np.sin(tilt)
        v = -dx * np.sin(tilt) + dy * np.cos(tilt)
        dist = np.sqrt((u / rx) ** 2 + (v / ry) ** 2)
        theta = np.arctan2(v, u)
        radius = np.ones_like(theta)
        for k in (2, 3, 4):
            radius += r.uniform(-0.08, 0.08) * np.sin(k * theta + r.uniform(0, 2 * np.pi))
        hard = (dist < radius).astype(np.uint8) * 255
        soft = Image.fromarray(hard).filter(ImageFilter.GaussianBlur(radius=r.uniform(2, 10)))
        mask = np.asarray(soft).astype(np.float32) / 255.0
        mask = mask / max(mask.max(), 1e-6)
        return mask * r.choice([0.25, 0.5, 0.75, 1.0, 1.0, 1.0])

    def __call__(self, img):
        w, h = img.size
        target = np.asarray(img).astype(np.float32)
        source = np.asarray(self._alter(img)).astype(np.float32)
        mask = self._mask(w, h)[..., None]
        out = source * mask + target * (1 - mask)
        return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))


def build_struct_drop(kind, p):
    if kind == "none":
        return transforms.Lambda(lambda img: img)
    if kind == "facecut":
        return FaceCut(p)
    if kind == "gridmask":
        return GridMask(p)
    raise ValueError(f"unknown struct_drop: {kind}")


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


def build_train_transform(aug_level="strong", size=IMAGE_SIZE, struct_drop="none", struct_p=0.5,
                          mosaic_p=0.0, mosaic_max=16, heavy_p=0.0):
    steps = [
        transforms.RandomResizedCrop(
            size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        build_degradation_block(aug_level),
    ]
    if mosaic_p > 0:
        steps.append(Mosaic(block_range=(4, mosaic_max), p=mosaic_p))
    if heavy_p > 0:
        steps.append(HeavyDegrade(p=heavy_p))
    if struct_drop != "none":
        steps.append(build_struct_drop(struct_drop, struct_p))
    steps += [
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25),
    ]
    return transforms.Compose(steps)


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

    train_tf = build_train_transform(args.aug_level, struct_drop=args.struct_drop,
                                     struct_p=args.struct_p)
    eval_tf = build_eval_transform()
    if args.sbi:
        sample_rows = random.sample([r for r in train_rows if r["label"] == "0"], 8)
        np.random.seed(0)
        blend = SelfBlend()

    fig, axes = plt.subplots(8, 6, figsize=(12, 16))
    for i, row in enumerate(sample_rows):
        img = Image.open(row["path"]).convert("RGB")
        for j in range(6):
            ax = axes[i, j]
            if j == 0:
                ax.imshow(img)
                ax.set_title("original", fontsize=8)
            elif args.sbi:
                sbi = blend(img)
                if j < 5:
                    ax.imshow(sbi)
                    ax.set_title(f"self-blend {j}", fontsize=8)
                else:
                    diff = np.abs(np.asarray(sbi, np.float32) - np.asarray(img, np.float32))
                    ax.imshow(np.clip(diff.mean(axis=2) * 4, 0, 255), cmap="gray", vmin=0, vmax=255)
                    ax.set_title("|blend - original| x4", fontsize=8)
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
    parser.add_argument("--struct_drop", default="none", choices=["none", "gridmask", "facecut"])
    parser.add_argument("--struct_p", type=float, default=0.5)
    parser.add_argument("--sbi", action="store_true",
                        help="preview self-blended pseudo-fakes of real train images")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.out is None:
        suffix = "_sbi" if args.sbi else ("" if args.struct_drop == "none" else "_" + args.struct_drop)
        args.out = f"outputs/figures/aug_preview{suffix}.png"

    if args.preview:
        _preview(args)
    else:
        print("nothing to do without --preview; import build_train_transform / "
              "build_eval_transform instead")


if __name__ == "__main__":
    main()
