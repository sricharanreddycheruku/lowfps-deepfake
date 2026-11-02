"""TensorFlow inference script for the Deepfake ML Hackathon submission.

Counterpart of inference_script.py for the SavedModel export (src/export_tf.py).
--weights is the TF model directory: it holds meta.json (fusion rule, TTA flag,
threshold, normalisation constants and the list of members) and one SavedModel
subdirectory per member. Preprocessing, horizontal-flip TTA, logit fusion and
the output format are the same as in the PyTorch script; PyTorch is not needed.

    python inference_script_tf.py \
        --test_dir ./test/images/ \
        --weights ./tf_model/ \
        --output predictions_tf.csv

Optional: --batch_size (default 64), --threshold (default: the value in
meta.json), --num_workers (image decoding threads, default min(8, cpu count)).
Members that were exported with a fixed batch size are fed padded chunks of
that size, whatever --batch_size is.
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


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


def make_loader(mean, std):
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    def load(path, size):
        """Same as Resize((size, size)) + ToTensor + Normalize, laid out as HWC."""
        img = Image.open(path).convert("RGB")
        if img.size != (size, size):
            img = img.resize((size, size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return (arr - mean) / std

    return load


def load_members(tf, model_dir, meta):
    members = []
    for m in meta["members"]:
        loaded = tf.saved_model.load(os.path.join(model_dir, m["dir"]))
        members.append((m, loaded, loaded.signatures["serving_default"]))
        print(f"[INFO] loaded {m['name']} ({m['arch']}, {m['img_size']}px, "
              f"batch {m.get('batch') or 'dynamic'})")
    return members


def run_member(tf, member, infer, batch):
    """Logits for an NHWC float32 batch; pads when the SavedModel has a fixed batch."""
    fixed = member.get("batch")
    if not fixed:
        return infer(**{member["input"]: tf.constant(batch)})[member["output"]].numpy().reshape(-1)
    out = []
    for start in range(0, len(batch), fixed):
        chunk = batch[start:start + fixed]
        pad = fixed - len(chunk)
        if pad:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], pad, axis=0)])
        res = infer(**{member["input"]: tf.constant(chunk)})[member["output"]].numpy().reshape(-1)
        out.append(res[:fixed - pad])
    return np.concatenate(out)


def predict(tf, model_dir, meta, paths, batch_size, num_workers):
    members = load_members(tf, model_dir, meta)
    load = make_loader(meta["mean"], meta["std"])
    sizes = sorted({m["img_size"] for m, _, _ in members})
    flips = (False, True) if meta["tta"] else (False,)
    per_member = [[] for _ in members]
    print(f"[INFO] {len(paths)} images, batch {batch_size}, tta {meta['tta']}, "
          f"fusion {meta['fusion']}, members {[m['arch'] for m, _, _ in members]}")

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for start in range(0, len(paths), batch_size):
            chunk = paths[start:start + batch_size]
            batches = {s: np.stack(list(pool.map(lambda p: load(p, s), chunk))) for s in sizes}
            for i, (m, _, infer) in enumerate(members):
                views = []
                for flip in flips:
                    x = batches[m["img_size"]]
                    if flip:
                        x = np.ascontiguousarray(x[:, :, ::-1, :])
                    views.append(run_member(tf, m, infer, x))
                per_member[i].append(np.mean(views, axis=0))
    logits = np.stack([np.concatenate(c) for c in per_member])
    weights = [m["weight"] for m, _, _ in members]
    return fuse(logits, meta["fusion"], weights)


def main():
    parser = argparse.ArgumentParser(description="Deepfake ML Hackathon inference (TensorFlow)")
    parser.add_argument("--test_dir", required=True, help="directory with the test images")
    parser.add_argument("--weights", required=True, help="TF model directory (contains meta.json)")
    parser.add_argument("--output", default="predictions_tf.csv", help="output CSV path")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=None,
                        help="label threshold; defaults to the value in meta.json")
    parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    start = time.time()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    with open(os.path.join(args.weights, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    if meta["failed"]:
        print("[WARN] members missing from this export: "
              + ", ".join(x["arch"] for x in meta["failed"]))

    paths = list_images(args.test_dir)
    if not paths:
        sys.exit(f"no images found in {args.test_dir}")
    ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    assert len(set(ids)) == len(ids), "duplicate image ids (same stem, different extension)"

    threshold = meta["threshold"] if args.threshold is None else args.threshold
    probs = predict(tf, args.weights, meta, paths, args.batch_size, args.num_workers)
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

    print(f"[DONE] wrote {args.output}")
    print(f"  images: {len(ids)}, predicted fake: {int(labels.sum())} "
          f"({labels.mean() * 100:.1f}%), mean confidence {probs.mean():.4f}")
    print(f"  elapsed: {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
