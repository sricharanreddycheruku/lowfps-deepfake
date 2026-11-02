"""Grad-CAM for the bundled ensemble members, written with plain hooks.

For every member the target layer (models.gradcam_target_layer) is hooked: a
forward hook keeps its output and registers a tensor hook that keeps the
gradient of the "fake" logit with respect to that output. The map is

    cam = relu(sum_c  mean_hw(grad_c) * act_c)

upsampled to the input size and min-max scaled per image. The three members
are different families, so the target is the last conv block for ResNet-50 and
ConvNeXt, and the output of the final norm of Swin-T (a channels-last
(N, H, W, C) tensor, permuted to (N, C, H, W) here). Each model has one logit,
so a positive weight always means "evidence for fake".

Figures written to --out (default outputs/figures):
    gradcam_<arch>_<case>.png    4x4 grids for case in tp, tn, fp, fn
    gradcam_<arch>_robust.png    same images clean / JPEG 30 / blur 2 / noise 10
    gradcam_ensemble_<case>.png  mean of the three members' normalised maps
    gradcam_ensemble_robust.png
plus gradcam_stats.json (where the map mass sits: border ring, eye band,
mouth band, centre) and gradcam_cases.csv (which images were drawn).

Cases use the fused ensemble probability at the bundle threshold on the val
split; images are sampled with a fixed seed. Single forward pass per image,
no flip TTA.

Usage:
    python src/gradcam.py --weights submission/model_weights.pt \
        --split_csv data/splits/split_v1.csv --out outputs/figures
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from augment import build_eval_transform  # noqa: E402
from models import build_model, gradcam_target_layer  # noqa: E402
from robust_eval import VARIANTS, load_rows  # noqa: E402

CASES = ("tp", "tn", "fp", "fn")
CASE_TITLES = {
    "tp": "true positive (fake, predicted fake)",
    "tn": "true negative (real, predicted real)",
    "fp": "false positive (real, predicted fake)",
    "fn": "false negative (fake, predicted real)",
}
CONDITIONS = ("clean", "jpeg_q30", "blur_sigma2", "noise_std10")
CONDITION_TITLES = {
    "clean": "clean", "jpeg_q30": "JPEG 30", "blur_sigma2": "blur 2", "noise_std10": "noise 10",
}


class GradCAM:
    """Hook-based Grad-CAM for a single-logit model."""

    def __init__(self, model, arch):
        self.model = model
        self.arch = arch
        self.activation = None
        self.gradient = None
        layer = gradcam_target_layer(arch, model)
        self.handle = layer.register_forward_hook(self._on_forward)

    def _to_map(self, tensor):
        # Swin-T keeps channels last: (N, H, W, C) -> (N, C, H, W)
        return tensor.permute(0, 3, 1, 2) if self.arch == "swin_t" else tensor

    def _on_forward(self, module, inputs, output):
        self.activation = self._to_map(output)
        if output.requires_grad:
            output.register_hook(self._on_backward)

    def _on_backward(self, grad):
        self.gradient = self._to_map(grad)

    def __call__(self, images):
        """images: (N, 3, H, W). Returns logits (N,) and cams (N, H, W) in [0, 1]."""
        self.model.zero_grad(set_to_none=True)
        images = images.clone().requires_grad_(True)
        logits = self.model(images).squeeze(1)
        logits.sum().backward()
        weights = self.gradient.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1).detach()
        flat = cam.flatten(1)
        low = flat.min(dim=1).values[:, None, None]
        high = flat.max(dim=1).values[:, None, None]
        cam = (cam - low) / (high - low + 1e-8)
        return logits.detach(), cam

    def close(self):
        self.handle.remove()


def load_members(bundle, device):
    members = []
    for m in bundle["members"]:
        model = build_model(m["arch"], pretrained=False)
        model.load_state_dict(m["state_dict"])
        members.append((m["name"], m["arch"], int(m["img_size"]), model.to(device).eval()))
    return members


def load_image(row, variant, idx, size):
    img = Image.open(row["path"]).convert("RGB")
    img = VARIANTS[variant](img, idx)
    return build_eval_transform(size)(img)


def display_image(row, variant, idx, size):
    """The degraded RGB image as shown in the figure (undone normalisation is
    not needed: it is rebuilt from the PIL image directly)."""
    img = Image.open(row["path"]).convert("RGB")
    img = VARIANTS[variant](img, idx).resize((size, size), Image.BILINEAR)
    return np.asarray(img)


def run_members(members, rows, idxs, variant, device, batch=16):
    """Per member: logits (n,) and cams (n, 224, 224) for the given row indices."""
    out = {}
    for name, arch, size, model in members:
        cam_fn = GradCAM(model, arch)
        logits, cams = [], []
        for start in range(0, len(idxs), batch):
            chunk = idxs[start:start + batch]
            x = torch.stack([load_image(rows[i], variant, i, size) for i in chunk]).to(device)
            lg, cam = cam_fn(x)
            if cam.shape[-1] != 224:
                cam = F.interpolate(cam[:, None], size=(224, 224), mode="bilinear",
                                    align_corners=False)[:, 0]
            logits.append(lg.cpu().numpy())
            cams.append(cam.cpu().numpy())
        cam_fn.close()
        out[name] = (np.concatenate(logits), np.concatenate(cams))
    return out


@torch.no_grad()
def fused_probs(members, rows, device, batch=64):
    """Plain (no TTA) mean-logit ensemble probability for case selection and display."""
    per_member = []
    for name, arch, size, model in members:
        tf = build_eval_transform(size)
        vals = []
        for start in range(0, len(rows), batch):
            x = torch.stack([tf(Image.open(r["path"]).convert("RGB"))
                             for r in rows[start:start + batch]]).to(device)
            vals.append(model(x).squeeze(1).float().cpu().numpy())
        per_member.append(np.concatenate(vals))
    return 1.0 / (1.0 + np.exp(-np.mean(per_member, axis=0)))


def overlay(image, cam, alpha=0.45):
    heat = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip((1 - alpha) * image / 255.0 + alpha * heat, 0, 1)


def draw_grid(path, title, images, cams, captions, ncols=4):
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 2.45 * nrows + 0.5))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, img, cam, cap in zip(np.ravel(axes), images, cams, captions):
        ax.imshow(overlay(img, cam))
        ax.set_title(cap, fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def region_masks(size=224):
    yy, xx = np.mgrid[0:size, 0:size] / size
    border = (yy < 0.15) | (yy > 0.85) | (xx < 0.15) | (xx > 0.85)
    eyes = (yy >= 0.25) & (yy < 0.48) & ~border
    mouth = (yy >= 0.62) & (yy < 0.85) & ~border
    centre = ~border & ~eyes & ~mouth
    return {"border_ring": border, "eye_band": eyes, "mouth_band": mouth, "centre_other": centre}


def region_share(cams):
    """Share of total map mass per region and its ratio to the region's area."""
    masks = region_masks(cams.shape[-1])
    total = cams.sum(axis=(1, 2)) + 1e-8
    stats = {}
    for key, mask in masks.items():
        share = (cams[:, mask].sum(axis=1) / total).mean()
        stats[key] = {"mass_share": float(share), "area_share": float(mask.mean())}
    return stats


def pick_cases(rows, probs, threshold, per_case, rng):
    labels = np.array([int(r["label"]) for r in rows])
    pred = (probs >= threshold).astype(int)
    masks = {
        "tp": (labels == 1) & (pred == 1), "tn": (labels == 0) & (pred == 0),
        "fp": (labels == 0) & (pred == 1), "fn": (labels == 1) & (pred == 0),
    }
    chosen = {}
    for case, mask in masks.items():
        pool = np.flatnonzero(mask)
        take = min(per_case, len(pool))
        chosen[case] = sorted(rng.choice(pool, size=take, replace=False).tolist())
    return chosen


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM figures for the bundled members")
    parser.add_argument("--weights", default="submission/model_weights.pt")
    parser.add_argument("--split_csv", default="data/splits/split_v1.csv")
    parser.add_argument("--out", default="outputs/figures")
    parser.add_argument("--per_case", type=int, default=16)
    parser.add_argument("--robust_n", type=int, default=6, help="images per member in the robustness figure")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    try:
        bundle = torch.load(args.weights, map_location="cpu", weights_only=True)
    except Exception:
        bundle = torch.load(args.weights, map_location="cpu", weights_only=False)
    members = load_members(bundle, device)
    rows = load_rows(args.split_csv, split="val")
    threshold = float(bundle["threshold"])
    rng = np.random.RandomState(args.seed)

    probs = fused_probs(members, rows, device)
    chosen = pick_cases(rows, probs, threshold, args.per_case, rng)
    print({c: len(v) for c, v in chosen.items()}, "of", len(rows), "val images")

    names = [m[0] for m in members]
    archs = {m[0]: m[1] for m in members}
    stats = {}
    case_rows = []

    # case grids: per member and ensemble
    for case in CASES:
        idxs = chosen[case]
        result = run_members(members, rows, idxs, "clean", device)
        images = [display_image(rows[i], "clean", i, 224) for i in idxs]
        captions = [f"p={probs[i]:.2f}" for i in idxs]
        for name in names:
            _, cams = result[name]
            draw_grid(os.path.join(args.out, f"gradcam_{archs[name]}_{case}.png"),
                      f"{archs[name]}: {CASE_TITLES[case]}", images, cams, captions)
            stats.setdefault(archs[name], {})[case] = region_share(cams)
        ens = np.mean([result[n][1] for n in names], axis=0)
        draw_grid(os.path.join(args.out, f"gradcam_ensemble_{case}.png"),
                  f"ensemble mean map: {CASE_TITLES[case]}", images, ens, captions)
        stats.setdefault("ensemble", {})[case] = region_share(ens)
        for i in idxs:
            case_rows.append({"case": case, "image_id": rows[i]["image_id"],
                              "label": rows[i]["label"], "ensemble_prob": f"{probs[i]:.4f}"})
        print("case", case, "done")

    # robustness: the same images under four conditions, drawn from TP and TN
    half = args.robust_n // 2
    idxs = chosen["tp"][:half] + chosen["tn"][:half]
    per_cond = {c: run_members(members, rows, idxs, c, device) for c in CONDITIONS}
    for name in names + ["ensemble"]:
        images, cams, captions = [], [], []
        for i_pos, i in enumerate(idxs):
            for cond in CONDITIONS:
                images.append(display_image(rows[i], cond, i, 224))
                if name == "ensemble":
                    cams.append(np.mean([per_cond[cond][n][1][i_pos] for n in names], axis=0))
                    z = np.mean([per_cond[cond][n][0][i_pos] for n in names])
                else:
                    cams.append(per_cond[cond][name][1][i_pos])
                    z = per_cond[cond][name][0][i_pos]
                captions.append(f"{CONDITION_TITLES[cond]} p={1 / (1 + np.exp(-z)):.2f}")
        tag = "ensemble" if name == "ensemble" else archs[name]
        title = (f"{tag}: same images under JPEG 30 / blur 2 / noise 10 "
                 f"(rows 1-{half} fake, {half + 1}-{2 * half} real)")
        draw_grid(os.path.join(args.out, f"gradcam_{tag}_robust.png"), title, images, cams, captions)
        cam_arr = np.stack(cams)
        stats.setdefault(tag, {})["robust_by_condition"] = {
            cond: region_share(cam_arr[k::len(CONDITIONS)]) for k, cond in enumerate(CONDITIONS)}
        # how much the map moves under degradation: correlation with the clean map
        corr = {}
        for k, cond in enumerate(CONDITIONS[1:], start=1):
            vals = [np.corrcoef(cam_arr[r * len(CONDITIONS)].ravel(),
                                cam_arr[r * len(CONDITIONS) + k].ravel())[0, 1]
                    for r in range(len(idxs))]
            corr[cond] = float(np.nanmean(vals))
        stats[tag]["corr_with_clean"] = corr
    print("robust figures done")

    with open(os.path.join(args.out, "gradcam_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold, "n_val": len(rows),
                   "cases": {c: len(v) for c, v in chosen.items()}, "stats": stats}, f, indent=2)
    with open(os.path.join(args.out, "gradcam_cases.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "image_id", "label", "ensemble_prob"])
        writer.writeheader()
        writer.writerows(case_rows)
    print("wrote figures to", args.out)


if __name__ == "__main__":
    main()
