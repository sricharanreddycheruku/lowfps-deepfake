"""Trains one backbone on the train split and tracks val AUC per epoch.

CLI mirrors the plan: --arch --data_root --split_csv --epochs --lr --wd --bs
--aug_level --label_smoothing --seed --out_dir (plus a few knobs needed to
make a CPU smoke run fast: --subset_frac, --num_workers, --no_ema) and the
recipe knobs used by the ablations: --ema_decay, --backbone_lr_mult,
--warmup_epochs, --conflict {soft,drop}.

Recipe: BCEWithLogits with label smoothing on the noisy labels, AdamW with
the backbone at 0.1x the head's learning rate, 1 epoch of linear warmup then
cosine decay, AMP (CUDA only; a no-op on CPU), an EMA shadow of the weights,
fixed seeds, channels_last memory format.

Each epoch reports clean val AUC and a fast "robust" val AUC: the mean AUC
over three fixed degradations (JPEG q50, Gaussian blur sigma 1, Gaussian
noise std 10) applied deterministically to the val images. This is a cheap
stand-in for the full variant suite in src/robust_eval.py so training can checkpoint on a signal that rewards
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
from torchvision.transforms.functional import gaussian_blur

from augment import SelfBlend, build_eval_transform, build_train_transform
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
    def __init__(self, rows, transform, soft=True, sbi_p=0.0):
        self.rows = rows
        self.transform = transform
        # probability that a hard-real image is replaced by a self-blended
        # pseudo-fake (target 1) before the train transform
        self.sbi_p = sbi_p
        # val rows must be scored against hard labels, soft 0.5 targets
        # only make sense as training targets
        self.soft = soft

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row["path"]).convert("RGB")
        target = float(row["soft_label"] if self.soft else row["label"])
        if self.sbi_p > 0 and target == 0.0 and random.random() < self.sbi_p:
            # python's random is reseeded per DataLoader worker, numpy's is not
            img = SelfBlend(np.random.RandomState(random.getrandbits(32)))(img)
            target = 1.0
        img = self.transform(img)
        return img, target, idx


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
        return img, target, idx


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


def sam_step(model, optimizer, scaler, compute_loss, loss, rho):
    """Sharpness-aware step. `loss` is the already-built first-pass loss;
    `compute_loss` reruns forward on the same batch. Gradients are unscaled
    before the perturbation (AMP), the step is skipped on inf/nan."""
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    params = [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
    norm = torch.norm(torch.stack([p.grad.norm(2) for p in params]), 2)
    if not torch.isfinite(norm):
        # scaler.step sees the inf from unscale_ and skips; update lowers the scale
        scaler.step(optimizer)
        scaler.update()
        return
    scale = rho / (norm + 1e-12)
    eps = []
    with torch.no_grad():
        for p in params:
            e = p.grad * scale
            p.add_(e)
            eps.append(e)
    optimizer.zero_grad(set_to_none=True)
    loss2 = compute_loss()
    scaler.scale(loss2).backward()
    inv = 1.0 / scaler.get_scale()
    finite = True
    with torch.no_grad():
        for p, e in zip(params, eps):
            p.sub_(e)
            if p.grad is not None:
                p.grad.mul_(inv)
                finite = finite and bool(torch.isfinite(p.grad).all())
    if finite:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.zero_grad(set_to_none=True)
        scaler.update(scaler.get_scale() * 0.5)


def elr_reg(logits, targets_ema, idx, beta):
    """Binary early-learning regularisation term; updates the running
    per-sample prediction targets in place."""
    p = torch.sigmoid(logits.float())
    with torch.no_grad():
        targets_ema[idx] = beta * targets_ema[idx] + (1 - beta) * p.detach()
    t = targets_ema[idx]
    agree = p * t + (1 - p) * (1 - t)
    return torch.log((1 - agree).clamp_min(1e-4))


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
    for images, targets, _ in loader:
        images = images.to(device, memory_format=torch.channels_last)
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_targets.extend(targets.numpy().tolist())
    return roc_auc_score(all_targets, all_probs)


def generalized_cross_entropy(logits, targets, q=0.7):
    """Bounded per-example loss, with soft targets treated as class weights."""
    logits = logits.float()
    log_fake = nn.functional.logsigmoid(logits)
    log_real = nn.functional.logsigmoid(-logits)
    return -(targets * torch.expm1(q * log_fake)
             + (1 - targets) * torch.expm1(q * log_real)) / q


def stronger_view(images, step):
    """Apply one extra spatial degradation to normalized NCHW images."""
    height, width = images.shape[-2:]
    family = step % 3
    severity = (step // 3) % 5
    if family == 0:
        block = 6 + 4 * severity
        size = (max(1, height // block), max(1, width // block))
        small = nn.functional.interpolate(images, size=size, mode="nearest-exact")
        return nn.functional.interpolate(small, size=(height, width), mode="nearest-exact")
    if family == 1:
        sigma = 2.0 + 0.5 * severity
        return gaussian_blur(images, [17, 17], [sigma, sigma])

    scale = 0.09 + 0.025 * severity
    size = (max(1, int(height * scale)), max(1, int(width * scale)))
    small = nn.functional.interpolate(images, size=size, mode="bilinear",
                                      align_corners=False, antialias=True)
    return nn.functional.interpolate(small, size=(height, width), mode="bilinear",
                                     align_corners=False)


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
    parser.add_argument("--sam", default="none", choices=["none", "sam"])
    parser.add_argument("--sam_rho", type=float, default=0.05)
    parser.add_argument("--loss", default="bce", choices=["bce", "elr", "gce"])
    parser.add_argument("--gce_warmup", type=int, default=2,
                        help="number of initial BCE epochs before switching to GCE")
    parser.add_argument("--elr_lambda", type=float, default=3.0)
    parser.add_argument("--elr_beta", type=float, default=0.7)
    parser.add_argument("--struct_drop", default="none", choices=["none", "gridmask", "facecut"],
                         help="blank a facial region or a grid of cells in train images")
    parser.add_argument("--struct_p", type=float, default=0.5)
    parser.add_argument("--mosaic_p", type=float, default=0.0,
                        help="probability of pixelating a train image into square blocks")
    parser.add_argument("--mosaic_max", type=int, default=16,
                        help="largest mosaic block size in px (smallest is 4)")
    parser.add_argument("--all_data", action="store_true",
                        help="train on the val rows too; val AUC in the log is then not held out")
    parser.add_argument("--save_at", type=int, default=0,
                        help="also write fixed.pt after this epoch (1-based), with no selection on val")
    parser.add_argument("--heavy_p", type=float, default=0.0,
                        help="probability of a strong blur or a very low resolution round trip")
    parser.add_argument("--sbi_p", type=float, default=0.0,
                        help="probability of turning a real train image into a self-blended "
                             "pseudo-fake (label 1)")
    parser.add_argument("--paired_target_mix", type=float, default=None,
                        help="enable extra degraded views after epoch two; this share of "
                             "their target comes from the detached anchor prediction (0 = labels only)")
    args = parser.parse_args()
    if args.loss == "elr" and args.mix != "none":
        parser.error("--loss elr cannot be combined with --mix")
    if args.sam == "sam" and args.loss == "elr":
        parser.error("--sam sam cannot be combined with --loss elr")

    if args.loss == "gce" and not 0 <= args.gce_warmup < args.epochs:
        parser.error("--gce_warmup must be nonnegative and less than --epochs")

    if args.paired_target_mix is not None:
        if not 0 <= args.paired_target_mix <= 1:
            parser.error("--paired_target_mix must be between 0 and 1")
        if args.epochs <= 2 or args.img_size < 17:
            parser.error("paired views need more than two epochs and --img_size at least 17")
        if args.loss != "bce" or args.mix != "none" or args.sam != "none" or args.trim_frac:
            parser.error("paired views require BCE without mixing, SAM or trimming")

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    rows = load_rows(args.split_csv)
    train_rows = rows if args.all_data else [r for r in rows if r["split"] == "train"]
    if args.conflict == "drop":
        train_rows = [r for r in train_rows if r["is_conflict"] != "1"]
    train_rows = subsample(train_rows, args.subset_frac * args.train_frac, args.seed)
    val_rows = subsample([r for r in rows if r["split"] == "val"], args.subset_frac, args.seed)

    train_ds = FaceDataset(train_rows, build_train_transform(args.aug_level, args.img_size,
                                                         args.struct_drop, args.struct_p,
                                                         args.mosaic_p, args.mosaic_max,
                                                         args.heavy_p),
                           sbi_p=args.sbi_p)
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

    elr_targets = torch.full((len(train_ds),), 0.5, device=device) if args.loss == "elr" else None

    log_path = os.path.join(args.out_dir, "log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "clean_val_auc", "robust_val_auc",
                                 "score", "lr", "epoch_seconds"])

    best_score = -1.0
    global_step = 0
    run_start = time.time()
    for epoch in range(args.epochs):
        use_gce = args.loss == "gce" and epoch >= args.gce_warmup
        use_pairs = args.paired_target_mix is not None and epoch >= 2
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        n_batches = 0
        for images, targets, idx in train_loader:
            images = images.to(device, memory_format=torch.channels_last)
            targets = targets.to(device)
            idx = idx.to(device)
            if args.mix != "none" and random.random() < args.mix_prob:
                images, targets = mix_batch(images, targets, args.mix, args.mix_alpha)
            targets = smooth(targets)

            if use_pairs:
                pair_count = max(1, images.shape[0] // 2)
                degraded = stronger_view(images[:pair_count], global_step)
                paired_images = torch.cat((images, degraded), dim=0)

            optimizer.zero_grad(set_to_none=True)

            def compute_loss(update_elr=False):
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    if use_pairs:
                        anchor, extra = model(paired_images).squeeze(1).split(
                            (len(images), pair_count))
                        teacher = anchor[:pair_count].detach().float().sigmoid()
                        extra_targets = targets[:pair_count].float().lerp(teacher, args.paired_target_mix)
                        return (criterion(anchor, targets).mean()
                                + 0.5 * criterion(extra, extra_targets).mean()) / 1.5

                    logits = model(images).squeeze(1)
                    per_sample = (generalized_cross_entropy(logits, targets)
                                  if use_gce else criterion(logits, targets))
                    if update_elr:
                        per_sample = per_sample + args.elr_lambda * elr_reg(
                            logits, elr_targets, idx, args.elr_beta)
                    if args.trim_frac > 0 and epoch >= args.trim_start:
                        keep = max(1, int(round(per_sample.numel() * (1 - args.trim_frac))))
                        per_sample = torch.topk(per_sample, keep, largest=False).values
                    return per_sample.mean()

            loss = compute_loss(args.loss == "elr")
            if args.sam == "sam":
                sam_step(model, optimizer, scaler, compute_loss, loss, args.sam_rho)
            elif use_amp:
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

        checkpoint = {
            "arch": args.arch,
            "state_dict": eval_model.state_dict(),
            "img_size": args.img_size,
            "epoch": epoch,
            "clean_val_auc": clean_auc,
            "robust_val_auc": robust_auc,
            "score": score,
        }
        if score > best_score:
            best_score = score
            torch.save(checkpoint, os.path.join(args.out_dir, "best.pt"))
        if epoch + 1 == args.save_at:
            torch.save(checkpoint, os.path.join(args.out_dir, "fixed.pt"))

    config = vars(args).copy()
    config["best_score"] = best_score
    config["train_seconds"] = round(time.time() - run_start, 1)
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"done, best score {best_score:.4f}, wrote {args.out_dir}")


if __name__ == "__main__":
    main()
