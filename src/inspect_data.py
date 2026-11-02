"""Inspect the raw dataset and write a data report plus a few diagnostic figures.

Reads train/real, train/fake, test/images and train_metadata.csv under
--data_root, cross-checks folder contents against the metadata csv, and
records per-class image statistics (size, mode, estimated JPEG quality,
brightness). Test images are only counted, never read for pixel statistics,
since the plan forbids computing any statistic on the test set before
inference.

Usage:
    python src/inspect_data.py --data_root data/raw/hackathon_dataset --out outputs/data
"""

import argparse
import csv
import json
import os
import random
from collections import Counter

from PIL import Image, ImageStat

# Standard JPEG baseline luminance quantization table (quality 50 reference).
# Used to back out an approximate JPEG quality from a decoded image's own
# quantization table, since the dataset does not carry quality in metadata.
STD_LUMINANCE_QT = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]


def estimate_jpeg_quality(img):
    """Back out an approximate JPEG quality (1-100) from the decoded
    image's luminance quantization table. Returns None if unavailable
    (e.g. not a JPEG, or PIL did not expose quantization tables)."""
    qt = getattr(img, "quantization", None)
    if not qt or 0 not in qt:
        return None
    table = list(qt[0])
    if len(table) != 64:
        return None
    scale = sum(table[i] / STD_LUMINANCE_QT[i] for i in range(64)) / 64 * 100
    if scale <= 0:
        return None
    if scale <= 100:
        quality = (200 - scale) / 2
    else:
        quality = 5000 / scale
    return max(1, min(100, round(quality)))


def mean_brightness(img):
    gray = img.convert("L")
    return ImageStat.Stat(gray).mean[0]


def scan_class(folder, label_name):
    """Walk one class folder and collect per-image stats."""
    records = []
    filenames = sorted(f for f in os.listdir(folder) if f.lower().endswith(".jpg"))
    for fname in filenames:
        path = os.path.join(folder, fname)
        image_id = os.path.splitext(fname)[0]
        file_size = os.path.getsize(path)
        with Image.open(path) as img:
            size = img.size
            mode = img.mode
            quality = estimate_jpeg_quality(img)
            img_rgb = img.convert("RGB")
            brightness = mean_brightness(img_rgb)
        records.append({
            "image_id": image_id,
            "class": label_name,
            "width": size[0],
            "height": size[1],
            "mode": mode,
            "file_size_bytes": file_size,
            "jpeg_quality_est": quality,
            "mean_brightness": brightness,
        })
    return records


def load_metadata(csv_path):
    rows = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["image_id"]] = int(row["label"])
    return rows


def summarize(records):
    sizes = Counter((r["width"], r["height"]) for r in records)
    modes = Counter(r["mode"] for r in records)
    qualities = [r["jpeg_quality_est"] for r in records if r["jpeg_quality_est"] is not None]
    file_sizes = [r["file_size_bytes"] for r in records]
    brightness = [r["mean_brightness"] for r in records]
    return {
        "count": len(records),
        "sizes": {f"{w}x{h}": c for (w, h), c in sizes.items()},
        "modes": dict(modes),
        "jpeg_quality_est": {
            "mean": sum(qualities) / len(qualities) if qualities else None,
            "min": min(qualities) if qualities else None,
            "max": max(qualities) if qualities else None,
        },
        "file_size_bytes": {
            "mean": sum(file_sizes) / len(file_sizes),
            "min": min(file_sizes),
            "max": max(file_sizes),
        },
        "mean_brightness": {
            "mean": sum(brightness) / len(brightness),
            "min": min(brightness),
            "max": max(brightness),
        },
    }


def make_figures(records, test_count, out_fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_fig_dir, exist_ok=True)

    real = [r for r in records if r["class"] == "real"]
    fake = [r for r in records if r["class"] == "fake"]

    # class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["real", "fake", "test (unlabelled)"], [len(real), len(fake), test_count],
           color=["#4472c4", "#c0504d", "#999999"])
    ax.set_ylabel("image count")
    ax.set_title("Class balance")
    fig.tight_layout()
    fig.savefig(os.path.join(out_fig_dir, "data_class_balance.png"), dpi=150)
    plt.close(fig)

    # JPEG quality histogram per class
    real_q = [r["jpeg_quality_est"] for r in real if r["jpeg_quality_est"] is not None]
    fake_q = [r["jpeg_quality_est"] for r in fake if r["jpeg_quality_est"] is not None]
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = list(range(0, 101, 5))
    ax.hist(real_q, bins=bins, alpha=0.6, label="real", color="#4472c4")
    ax.hist(fake_q, bins=bins, alpha=0.6, label="fake", color="#c0504d")
    ax.set_xlabel("estimated JPEG quality")
    ax.set_ylabel("count")
    ax.set_title("Estimated JPEG quality by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_fig_dir, "data_jpeg_quality_hist.png"), dpi=150)
    plt.close(fig)

    # sample grid: 8 real + 8 fake thumbnails
    rng = random.Random(42)
    sample_real = rng.sample(real, min(8, len(real)))
    sample_fake = rng.sample(fake, min(8, len(fake)))
    samples = sample_real + sample_fake
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for ax, rec in zip(axes.flat, samples):
        folder = "train/real" if rec["class"] == "real" else "train/fake"
        path = os.path.join(rec["_folder"], rec["image_id"] + ".jpg")
        with Image.open(path) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(rec["class"], fontsize=8)
        ax.axis("off")
    for ax in axes.flat[len(samples):]:
        ax.axis("off")
    fig.suptitle("Sample images (random, seed 42)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_fig_dir, "data_sample_grid.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out", required=True, help="output dir for data_report.json")
    args = parser.parse_args()

    real_dir = os.path.join(args.data_root, "train", "real")
    fake_dir = os.path.join(args.data_root, "train", "fake")
    test_dir = os.path.join(args.data_root, "test", "images")
    meta_path = os.path.join(args.data_root, "train_metadata.csv")

    print("Scanning train/real ...")
    real_records = scan_class(real_dir, "real")
    for r in real_records:
        r["_folder"] = real_dir
    print("Scanning train/fake ...")
    fake_records = scan_class(fake_dir, "fake")
    for r in fake_records:
        r["_folder"] = fake_dir
    all_records = real_records + fake_records

    # test set: count only, no pixel statistics
    test_count = sum(1 for f in os.listdir(test_dir) if f.lower().endswith(".jpg"))

    metadata = load_metadata(meta_path)

    # cross-check folders vs metadata csv
    folder_labels = {r["image_id"]: (0 if r["class"] == "real" else 1) for r in all_records}
    folder_ids = set(folder_labels)
    meta_ids = set(metadata)

    missing_from_metadata = sorted(folder_ids - meta_ids)
    missing_from_folders = sorted(meta_ids - folder_ids)
    label_disagreements = sorted(
        img_id for img_id in (folder_ids & meta_ids)
        if folder_labels[img_id] != metadata[img_id]
    )

    source_of_truth = "folders" if not label_disagreements else "folders (metadata disagrees on some rows; folder placement wins)"

    class_counts_folders = {"real": len(real_records), "fake": len(fake_records)}
    class_counts_metadata = dict(Counter(metadata.values()))
    class_counts_metadata = {
        "real": class_counts_metadata.get(0, 0),
        "fake": class_counts_metadata.get(1, 0),
    }

    report = {
        "train_counts": {
            "folders": class_counts_folders,
            "metadata_csv": class_counts_metadata,
            "missing_from_metadata": missing_from_metadata,
            "missing_from_folders": missing_from_folders,
            "label_disagreements": label_disagreements,
            "source_of_truth": source_of_truth,
        },
        "test_count": test_count,
        "test_count_expected": 3000,
        "real": summarize(real_records),
        "fake": summarize(fake_records),
    }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "data_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    fig_dir = os.path.normpath(os.path.join(args.out, "..", "figures"))
    make_figures(all_records, test_count, fig_dir)

    print(json.dumps({
        "train_counts_folders": class_counts_folders,
        "train_counts_metadata": class_counts_metadata,
        "label_disagreements": len(label_disagreements),
        "missing_from_metadata": len(missing_from_metadata),
        "missing_from_folders": len(missing_from_folders),
        "test_count": test_count,
        "real_quality_mean": report["real"]["jpeg_quality_est"]["mean"],
        "fake_quality_mean": report["fake"]["jpeg_quality_est"]["mean"],
        "real_brightness_mean": report["real"]["mean_brightness"]["mean"],
        "fake_brightness_mean": report["fake"]["mean_brightness"]["mean"],
    }, indent=2))


if __name__ == "__main__":
    main()
