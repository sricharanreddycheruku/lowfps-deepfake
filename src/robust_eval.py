"""Full robustness variant suite for a trained checkpoint (or several,
fused by averaging probabilities).

This supersedes the fast 3-variant proxy baked into src/train.py (that
proxy stays for per-epoch checkpoint selection, since this full suite is
too slow to run every epoch). Use this script for the real robustness
numbers: baseline comparisons, ablations and the
final ensemble.

Variant suite (deterministic, fixed parameters, no random draws other than
a per-image seed for the noise variants so repeated runs match exactly):
clean; JPEG q90/70/50/30; Gaussian blur sigma 1/2; Gaussian noise std
5/10/20; downscale x0.5/x0.25 then upscale back; brightness +/-30%;
contrast 0.6/1.4; grayscale; centre crop 90% + resize; rotation 5 deg;
horizontal flip; resize to 160 and 299 then back to 224; a combo variant
(JPEG 50 + blur 1 + noise 5).

--suite extended runs a second, separate set modelled on the corruptions
seen in the training images themselves: snow, rain streaks, black boxes,
coloured block dropout, pixelation, colour tint and JPEG q20 (plus clean).
It is reported on its own so the default 23-variant numbers stay comparable.

Per variant: AUC, F1 and accuracy at --threshold (default 0.5; src/threshold.py
picks the final threshold), and mean predicted confidence per class.
Summary row: clean AUC, mean AUC across all variants, worst-variant AUC,
std of AUC across variants.

Usage:
    python src/robust_eval.py --ckpt outputs/runs/smoke_efficientnet_b0/best.pt \
        --split_csv data/splits/split_v1.csv --out outputs/eval/smoke_efficientnet_b0

Multiple --ckpt values are fused by averaging per-image probabilities
(a simple fusion; src/ensemble.py compares fusion rules).
"""

import argparse
import csv
import io
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from augment import build_eval_transform
from models import build_model


def _jpeg(quality):
    def apply(img, idx):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    return apply


def _blur(sigma):
    def apply(img, idx):
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return apply


def _noise(std, salt):
    def apply(img, idx):
        arr = np.asarray(img).astype(np.float32)
        noise = np.random.RandomState(idx * 10007 + salt).normal(0, std, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    return apply


def _downscale_upscale(scale):
    def apply(img, idx):
        w, h = img.size
        small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)
    return apply


def _brightness(factor):
    def apply(img, idx):
        return ImageEnhance.Brightness(img).enhance(factor)
    return apply


def _contrast(factor):
    def apply(img, idx):
        return ImageEnhance.Contrast(img).enhance(factor)
    return apply


def _grayscale(img, idx):
    return img.convert("L").convert("RGB")


def _centercrop90(img, idx):
    w, h = img.size
    cw, ch = int(w * 0.9), int(h * 0.9)
    left, top = (w - cw) // 2, (h - ch) // 2
    cropped = img.crop((left, top, left + cw, top + ch))
    return cropped.resize((w, h), Image.BILINEAR)


def _rotation5(img, idx):
    return img.rotate(5, resample=Image.BILINEAR, fillcolor=(128, 128, 128))


def _hflip(img, idx):
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def _resize_roundtrip(size):
    def apply(img, idx):
        w, h = img.size
        return img.resize((size, size), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    return apply


def _combo(img, idx):
    img = _jpeg(50)(img, idx)
    img = _blur(1.0)(img, idx)
    img = _noise(5, salt=999)(img, idx)
    return img


def _clean(img, idx):
    return img


def _rng(idx, salt):
    return np.random.RandomState(idx * 10007 + salt)


def _snow(img, idx):
    rng = _rng(idx, 11)
    w, h = img.size
    layer = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(layer)
    for _ in range(300):
        x, y, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.8, 2.5)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=int(rng.uniform(180, 255)))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=0.6))
    return Image.composite(Image.new("RGB", (w, h), (255, 255, 255)), img, layer)


def _rain(img, idx):
    rng = _rng(idx, 12)
    w, h = img.size
    layer = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(layer)
    dx = rng.uniform(-0.3, 0.3)
    for _ in range(120):
        x, y, length = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(10, 25)
        draw.line((x, y, x + dx * length, y + length), fill=150, width=1)
    layer = layer.filter(ImageFilter.GaussianBlur(radius=0.5))
    return Image.composite(Image.new("RGB", (w, h), (220, 220, 220)), img, layer)


def _black_boxes(img, idx):
    rng = _rng(idx, 13)
    w, h = img.size
    img = img.copy()
    draw = ImageDraw.Draw(img)
    for _ in range(3):
        bw, bh = rng.uniform(0.12, 0.25) * w, rng.uniform(0.12, 0.25) * h
        x, y = rng.uniform(0, w - bw), rng.uniform(0, h - bh)
        draw.rectangle((x, y, x + bw, y + bh), fill=(0, 0, 0))
    return img


def _block_dropout(img, idx, block=16, p=0.1):
    rng = _rng(idx, 14)
    w, h = img.size
    img = img.copy()
    draw = ImageDraw.Draw(img)
    for y in range(0, h, block):
        for x in range(0, w, block):
            if rng.rand() < p:
                colour = tuple(int(c) for c in rng.randint(0, 256, size=3))
                draw.rectangle((x, y, x + block - 1, y + block - 1), fill=colour)
    return img


def _pixelate(factor):
    def apply(img, idx):
        w, h = img.size
        small = img.resize((max(1, w // factor), max(1, h // factor)), Image.NEAREST)
        return small.resize((w, h), Image.NEAREST)
    return apply


def _colour_tint(img, idx):
    rng = _rng(idx, 15)
    tint = Image.new("RGB", img.size, tuple(int(c) for c in rng.randint(0, 256, size=3)))
    return Image.blend(img, tint, 0.3)


EXTENDED_VARIANTS = {
    "clean": _clean,
    "snow": _snow,
    "rain": _rain,
    "black_boxes": _black_boxes,
    "block_dropout": _block_dropout,
    "pixelate8": _pixelate(8),
    "colour_tint": _colour_tint,
    "jpeg_q20": _jpeg(20),
}


def _pixelate_patch(block):
    def apply(img, idx):
        rng = _rng(idx, 16)
        w, h = img.size
        pw, ph = w // 2, h // 2
        left, top = int(rng.randint(0, w - pw + 1)), int(rng.randint(0, h - ph + 1))
        box = (left, top, left + pw, top + ph)
        out = img.copy()
        out.paste(_pixelate(block)(img.crop(box), idx), box)
        return out
    return apply


def _heavy_combo(img, idx):
    img = _jpeg(30)(img, idx)
    img = _blur(2.0)(img, idx)
    img = _noise(10, salt=998)(img, idx)
    return img


def _pixelate_jpeg(img, idx):
    return _jpeg(50)(_pixelate(8)(img, idx), idx)


STRESS_VARIANTS = {
    "clean": _clean,
    "pixelate5": _pixelate(5),
    "pixelate12": _pixelate(12),
    "pixelate24": _pixelate(24),
    "pixelate8_patch": _pixelate_patch(8),
    "pixelate8_jpeg50": _pixelate_jpeg,
    "downscale_0.125": _downscale_upscale(0.125),
    "jpeg_q10": _jpeg(10),
    "blur_sigma4": _blur(4.0),
    "noise_std40": _noise(40, salt=4),
    "heavy_combo": _heavy_combo,
}


VARIANTS = {
    "clean": _clean,
    "jpeg_q90": _jpeg(90),
    "jpeg_q70": _jpeg(70),
    "jpeg_q50": _jpeg(50),
    "jpeg_q30": _jpeg(30),
    "blur_sigma1": _blur(1.0),
    "blur_sigma2": _blur(2.0),
    "noise_std5": _noise(5, salt=1),
    "noise_std10": _noise(10, salt=2),
    "noise_std20": _noise(20, salt=3),
    "downscale_0.5": _downscale_upscale(0.5),
    "downscale_0.25": _downscale_upscale(0.25),
    "brightness_plus30": _brightness(1.3),
    "brightness_minus30": _brightness(0.7),
    "contrast_0.6": _contrast(0.6),
    "contrast_1.4": _contrast(1.4),
    "grayscale": _grayscale,
    "centercrop90": _centercrop90,
    "rotation5": _rotation5,
    "hflip": _hflip,
    "resize160": _resize_roundtrip(160),
    "resize299": _resize_roundtrip(299),
    "combo_jpeg50_blur1_noise5": _combo,
}


class VariantDataset(Dataset):
    def __init__(self, rows, variant_fn, eval_transform):
        self.rows = rows
        self.variant_fn = variant_fn
        self.eval_transform = eval_transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row["path"]).convert("RGB")
        img = self.variant_fn(img, idx)
        img = self.eval_transform(img)
        target = float(row["label"])
        return img, target


def load_rows(split_csv, split="val"):
    with open(split_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["split"] == split]


def load_models(ckpt_paths, device):
    models_list = []
    sizes = set()
    for path in ckpt_paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(ckpt["arch"], pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device, memory_format=torch.channels_last)
        model.eval()
        models_list.append(model)
        sizes.add(int(ckpt.get("img_size", 224)))
    if len(sizes) != 1:
        raise SystemExit(f"checkpoints use different input sizes {sorted(sizes)}; "
                         "evaluate mixed-size fusions with src/ensemble.py")
    return models_list, sizes.pop()


@torch.no_grad()
def predict_variant(models_list, rows, variant_fn, device, bs, num_workers, size=224):
    ds = VariantDataset(rows, variant_fn, build_eval_transform(size))
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers)
    all_targets, all_probs = [], []
    for images, targets in loader:
        images = images.to(device, memory_format=torch.channels_last)
        probs_per_model = []
        for model in models_list:
            logits = model(images).squeeze(1)
            probs_per_model.append(torch.sigmoid(logits).cpu().numpy())
        fused = np.mean(probs_per_model, axis=0)
        all_probs.extend(fused.tolist())
        all_targets.extend(targets.numpy().tolist())
    return np.array(all_targets), np.array(all_probs)


def compute_metrics(targets, probs, threshold):
    preds = (probs >= threshold).astype(int)
    auc = roc_auc_score(targets, probs)
    f1 = f1_score(targets, preds)
    acc = float(np.mean(preds == targets))
    real_mask = targets == 0
    fake_mask = targets == 1
    mean_conf_real = float(probs[real_mask].mean()) if real_mask.any() else float("nan")
    mean_conf_fake = float(probs[fake_mask].mean()) if fake_mask.any() else float("nan")
    return {
        "auc": auc,
        "f1": f1,
        "accuracy": acc,
        "mean_conf_real": mean_conf_real,
        "mean_conf_fake": mean_conf_fake,
    }


def run_suite(models_list, rows, device, bs, num_workers, threshold, size=224, variants=None):
    rows_out = []
    for name, variant_fn in (variants or VARIANTS).items():
        targets, probs = predict_variant(models_list, rows, variant_fn, device, bs, num_workers, size)
        metrics = compute_metrics(targets, probs, threshold)
        rows_out.append({"variant": name, **metrics})
        print(f"{name}: auc {metrics['auc']:.4f} f1 {metrics['f1']:.4f} acc {metrics['accuracy']:.4f}")

    aucs = [r["auc"] for r in rows_out]
    clean_auc = next(r["auc"] for r in rows_out if r["variant"] == "clean")
    summary = {
        "variant": "summary",
        "clean_auc": clean_auc,
        "mean_auc": float(np.mean(aucs)),
        "worst_auc": float(np.min(aucs)),
        "std_auc": float(np.std(aucs)),
    }
    return rows_out, summary


def write_outputs(rows_out, summary, out_prefix):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    csv_path = f"{out_prefix}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["variant", "auc", "f1", "accuracy", "mean_conf_real", "mean_conf_fake"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)
        writer.writerow({
            "variant": "summary",
            "auc": summary["mean_auc"],
            "f1": "", "accuracy": "",
            "mean_conf_real": "", "mean_conf_fake": "",
        })

    json_path = f"{out_prefix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"variants": rows_out, "summary": summary}, f, indent=2)

    print(f"wrote {csv_path} and {json_path}")
    print(f"summary: clean_auc {summary['clean_auc']:.4f} mean_auc {summary['mean_auc']:.4f} "
          f"worst_auc {summary['worst_auc']:.4f} std_auc {summary['std_auc']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", nargs="+", required=True)
    parser.add_argument("--split_csv", default="data/splits/split_v1.csv")
    parser.add_argument("--out", required=True, help="output path prefix, writes <out>.csv and <out>.json")
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--suite", default="default", choices=["default", "extended", "stress"],
                        help="default: the 23 standard variants; extended: corruptions seen in the data; "
                             "stress: harsher settings, not used in training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows(args.split_csv, split="val")
    models_list, size = load_models(args.ckpt, device)

    variants = {"default": VARIANTS, "extended": EXTENDED_VARIANTS, "stress": STRESS_VARIANTS}[args.suite]
    rows_out, summary = run_suite(models_list, rows, device, args.bs, args.num_workers,
                                  args.threshold, size, variants)
    write_outputs(rows_out, summary, args.out)


if __name__ == "__main__":
    main()
