"""Trains one backbone on the train split and tracks val AUC per epoch.

CLI mirrors the plan: --arch --data_root --split_csv --epochs --lr --wd --bs
--aug_level --label_smoothing --seed --out_dir (plus a few knobs needed to
make a CPU smoke run fast: --subset_frac, --num_workers, --no_ema) and the
recipe knobs used by the Chunk 5 ablations: --ema_decay, --backbone_lr_mult,
--warmup_epochs, --conflict {soft,drop}.

Recipe: BCEWithLogits with label smoothing on the noisy labels, AdamW with
the backbone at 0.1x the head's learning rate, 1 epoch of linear warmup then
cosine decay, AMP (CUDA only; a no-op on CPU), an EMA shadow of the weights,
fixed seeds, channels_last memory format.

Each epoch reports clean val AUC and a fast "robust" val AUC: the mean AUC
over three fixed degradations (JPEG q50, Gaussian blur sigma 1, Gaussian
noise std 10) applied deterministically to the val images. This is a cheap
stand-in for the full variant suite in src/robust_eval.py (Chunk 4, not
written yet) so training can already checkpoint on a signal that rewards
robustness, not just clean accuracy. The checkpoint metric is the mean of
clean and robust AUC.

Usage (full run):
    python src/train.py --arch efficientnet_b0 \
        --data_root data/raw/hackathon_dataset --split_csv data/splits/split_v1.csv \
        --epochs 25 --lr 3e-4 --wd 0.05 --bs 64 --aug_level strong \
        --label_smoothing 0.05 --seed 42 --out_dir outputs/runs/final_efficientnet_b0

Usage (CPU smoke test):
    python src/train.py --arch efficientnet_b0 --epochs 1 --subset_frac 0.05 \
        --bs 16 --num_workers 0 --out_dir outputs/runs/smoke_efficientnet_b0
"""

import argparse
import csv
import io
import json
import math
import os
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from augment import build_eval_transform, build_train_transform
from models import build_model, get_head_module


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_rows(split_csv):
    with open(split_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def subsample(rows, frac, seed):
    """Stratified-by-label subsample, used only for the fast CPU smoke
    test; a full run always gets frac=1.0."""
    if frac >= 1.0:
        return rows
    rng = random.Random(seed)
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    out = []
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        out.extend(label_rows[:max(1, int(len(label_rows) * frac))])
    return out


class FaceDataset(Dataset):
    def __init__(self, rows, transform, soft=True):
        self.rows = rows
        self.transform = transform
        # val rows must be scored against hard labels, soft 0.5 targets
        # only make sense as training targets
        self.soft = soft

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row["path"]).convert("RGB")
        img = self.transform(img)
        target = float(row["soft_label"] if self.soft else row["label"])
        return img, target


# the three fixed degradations used for the fast per-epoch robustness
# signal; deterministic per image index so repeated epochs are comparable
_ROBUST_VARIANTS = ["jpeg50", "blur1", "noise10"]


def _apply_robust_variant(img, variant, seed):
    rng = random.Random(seed)
    if variant == "jpeg50":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=50)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if variant == "blur1":
        return img.filter(ImageFilter.GaussianBlur(radius=1.0))
    if variant == "noise10":
        arr = np.asarray(img).astype(np.float32)
        noise = np.random.RandomState(seed).normal(0, 10.0, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    raise ValueError(f"unknown robust variant: {variant}")


class RobustEvalDataset(Dataset):
    """Applies one fixed, seeded degradation to each val image, then the
    usual eval transform."""

    def __init__(self, rows, variant, eval_transform):
        self.rows = rows
        self.variant = variant
        self.eval_transform = eval_transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row["path"]).convert("RGB")
        img = _apply_robust_variant(img, self.variant, seed=idx)
        img = self.eval_transform(img)
        target = float(row["label"])
        return img, target


def _layer_index(arch, name, n_features):
    """Depth of a parameter: 0 for the stem, higher for later stages. Used
    only for layer-wise LR decay (--llrd < 1)."""
    parts = name.split(".")
    if parts[0] == "features":
        return int(parts[1])
    if arch == "resnet50":
        if parts[0].startswith("layer"):
            return int(parts[0][5:])
        return 0
    return n_features


def make_optimizer(model, arch, lr, wd, backbone_mult=0.1, llrd=1.0):
    head_module = get_head_module(model, arch)
    head_params = list(head_module.parameters())
    head_ids = {id(p) for p in head_params}
    if llrd >= 1.0:
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
        return torch.optim.AdamW([
            {"params": backbone_params, "lr": lr * backbone_mult},
            {"params": head_params, "lr": lr},
        ], weight_decay=wd)
    # one group per depth, each decayed by llrd per level below the top
    depth = {}
    for name, p in model.named_parameters():
        if id(p) in head_ids:
            continue
        n_features = len(model.features) if hasattr(model, "features") else 5
        depth.setdefault(_layer_index(arch, name, n_features), []).append(p)
    top = max(depth)
    groups = [{"params": ps, "lr": lr * backbone_mult * llrd ** (top - d)}
              for d, ps in sorted(depth.items())]
    groups.append({"params": head_params, "lr": lr})
    return torch.optim.AdamW(groups, weight_decay=wd)


def lr_lambda(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class EMA:
    """Exponential moving average of model weights, kept as a plain state
    dict on the same device as the model."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = deepcopy(model.state_dict())

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
                else:
                    self.shadow[k] = v.clone()

    def copy_to(self, model):
        model.load_state_dict(self.shadow)


def mix_batch(images, targets, mode, alpha):
    """Mixup or CutMix on one batch; targets are mixed by the same ratio
    (by pixel area for CutMix) so they stay in [0, 1]."""
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(images.size(0), device=images.device)
    if mode == "mixup":
        images = lam * images + (1 - lam) * images[perm]
    else:
        h, w = images.shape[-2:]
        cut = math.sqrt(1 - lam)
        ch, cw = int(h * cut), int(w * cut)
        cy, cx = np.random.randint(h), np.random.randint(w)
        y0, y1 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
        x0, x1 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
        images = images.clone()
        images[:, :, y0:y1, x0:x1] = images[perm][:, :, y0:y1, x0:x1]
        lam = 1 - (y1 - y0) * (x1 - x0) / (h * w)
    return images, lam * targets + (1 - lam) * targets[perm]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_targets, all_probs = [], []
    for images, targets in loader:
        images = images.to(device, memory_format=torch.channels_last)
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_targets.extend(targets.numpy().tolist())
    return roc_auc_score(all_targets, all_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--data_root", default="data/raw/hackathon_dataset")
    parser.add_argument("--split_csv", default="data/splits/split_v1.csv")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--aug_level", default="strong", choices=["none", "light", "strong"])
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--subset_frac", type=float, default=1.0,
                         help="use only this fraction of train/val rows; for fast smoke tests")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--backbone_lr_mult", type=float, default=0.1,
                         help="backbone learning rate as a fraction of the head's")
    parser.add_argument("--warmup_epochs", type=float, default=1.0)
    parser.add_argument("--img_size", type=int, default=224,
                         help="network input side; images are resized to it in train and eval")
    parser.add_argument("--mix", default="none", choices=["none", "mixup", "cutmix"],
                         help="batch-level mixing applied with probability --mix_prob")
    parser.add_argument("--mix_prob", type=float, default=0.5)
    parser.add_argument("--mix_alpha", type=float, default=1.0)
    parser.add_argument("--conflict", default="soft", choices=["soft", "drop"],
                         help="soft keeps conflicting-label groups in train at target 0.5, "
                              "drop removes them from train")
    parser.add_argument("--llrd", type=float, default=1.0,
                         help="layer-wise LR decay factor per depth level (1 = off)")
    parser.add_argument("--trim_frac", type=float, default=0.0,
                         help="drop this fraction of highest-loss samples in each batch "
                              "once --trim_start epochs have passed (noisy-label guard)")
    parser.add_argument("--trim_start", type=int, default=2)
    parser.add_argument("--train_frac", type=float, default=1.0,
                         help="train on this label-stratified fraction of the train rows "
                              "only (val untouched); used for the data-scaling check")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    rows = load_rows(args.split_csv)
    train_rows = [r for r in rows if r["split"] == "train"]
    if args.conflict == "drop":
        train_rows = [r for r in train_rows if r["is_conflict"] != "1"]
    train_rows = subsample(train_rows, args.subset_frac * args.train_frac, args.seed)
    val_rows = subsample([r for r in rows if r["split"] == "val"], args.subset_frac, args.seed)

    train_ds = FaceDataset(train_rows, build_train_transform(args.aug_level, args.img_size))
    val_ds = FaceDataset(val_rows, build_eval_transform(args.img_size), soft=False)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                             num_workers=args.num_workers)
    robust_loaders = [
        DataLoader(RobustEvalDataset(val_rows, variant, build_eval_transform(args.img_size)),
                   batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
        for variant in _ROBUST_VARIANTS
    ]

    model = build_model(args.arch, pretrained=True)
    model = model.to(device, memory_format=torch.channels_last)

    optimizer = make_optimizer(model, args.arch, args.lr, args.wd, args.backbone_lr_mult, args.llrd)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_lambda(step, warmup_steps, total_steps))

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    ema = None if args.no_ema else EMA(model, decay=args.ema_decay)

    def smooth(target):
        # label smoothing pulls targets slightly off 0/1; soft labels at
        # 0.5 (conflicting-label groups) are left alone since they are
        # already maximally uncertain
        eps = args.label_smoothing
        return target * (1 - eps) + 0.5 * eps

    log_path = os.path.join(args.out_dir, "log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "clean_val_auc", "robust_val_auc",
                                 "score", "lr", "epoch_seconds"])

    best_score = -1.0
    global_step = 0
    run_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        n_batches = 0
        for images, targets in train_loader:
            images = images.to(device, memory_format=torch.channels_last)
            targets = targets.to(device)
            if args.mix != "none" and random.random() < args.mix_prob:
                images, targets = mix_batch(images, targets, args.mix, args.mix_alpha)
            targets = smooth(targets)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images).squeeze(1)
                per_sample = criterion(logits, targets)
                if args.trim_frac > 0 and epoch >= args.trim_start:
                    keep = max(1, int(round(per_sample.numel() * (1 - args.trim_frac))))
                    per_sample = torch.topk(per_sample, keep, largest=False).values
                loss = per_sample.mean()

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()
            global_step += 1

            if ema is not None:
                ema.update(model)

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)

        eval_model = model
        if ema is not None:
            eval_model = deepcopy(model)
            ema.copy_to(eval_model)

        clean_auc = evaluate(eval_model, val_loader, device)
        robust_aucs = [evaluate(eval_model, loader, device) for loader in robust_loaders]
        robust_auc = float(np.mean(robust_aucs))
        score = (clean_auc + robust_auc) / 2

        epoch_seconds = time.time() - epoch_start
        current_lr = optimizer.param_groups[-1]["lr"]
        print(f"epoch {epoch}: loss {train_loss:.4f} clean_auc {clean_auc:.4f} "
              f"robust_auc {robust_auc:.4f} score {score:.4f} ({epoch_seconds:.1f}s)")

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, train_loss, clean_auc, robust_auc,
                                     score, current_lr, epoch_seconds])

        if score > best_score:
            best_score = score
            torch.save({
                "arch": args.arch,
                "state_dict": eval_model.state_dict(),
                "img_size": args.img_size,
                "epoch": epoch,
                "clean_val_auc": clean_auc,
                "robust_val_auc": robust_auc,
                "score": score,
            }, os.path.join(args.out_dir, "best.pt"))

    config = vars(args).copy()
    config["best_score"] = best_score
    config["train_seconds"] = round(time.time() - run_start, 1)
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"done, best score {best_score:.4f}, wrote {args.out_dir}")


if __name__ == "__main__":
    main()
