"""Sanity checks for a predictions.csv produced by submission/inference_script.py.

Verifies the header, row count, id set (must equal the test filename stems),
uniqueness, label values, confidence range, NaNs, and that the label agrees
with the confidence at the given threshold. Prints the fraction predicted fake.
Exits with a non-zero status when any check fails.

Usage:
    python src/check_predictions.py --pred submission/predictions.csv \
        --test_dir data/raw/hackathon_dataset/test/images [--threshold 0.5]
"""

import argparse
import csv
import math
import os
import sys

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="predictions CSV")
    parser.add_argument("--test_dir", required=True, help="directory with the test images")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="label must equal (confidence >= threshold)")
    args = parser.parse_args()

    expected = sorted(os.path.splitext(n)[0] for n in os.listdir(args.test_dir)
                      if n.lower().endswith(IMAGE_EXTENSIONS))
    with open(args.pred, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    errors = []
    if header != ["image_id", "label", "confidence"]:
        errors.append(f"header is {header}")
    ids = [r[0] for r in rows]
    if len(rows) != len(expected):
        errors.append(f"{len(rows)} rows, expected {len(expected)}")
    if len(set(ids)) != len(ids):
        errors.append("duplicate image_id values")
    missing, extra = set(expected) - set(ids), set(ids) - set(expected)
    if missing:
        errors.append(f"{len(missing)} test images without a row, e.g. {sorted(missing)[:3]}")
    if extra:
        errors.append(f"{len(extra)} ids that are not test images, e.g. {sorted(extra)[:3]}")

    labels, confs = [], []
    for r in rows:
        if len(r) != 3:
            errors.append(f"malformed row {r}")
            break
        try:
            labels.append(int(r[1]))
            confs.append(float(r[2]))
        except ValueError:
            errors.append(f"non-numeric value in row {r}")
            break
    if len(confs) == len(rows):
        if any(math.isnan(c) for c in confs):
            errors.append("NaN confidence")
        if any(not 0.0 <= c <= 1.0 for c in confs):
            errors.append("confidence outside [0, 1]")
        if any(l not in (0, 1) for l in labels):
            errors.append("label not in {0, 1}")
        mismatched = sum(l != int(c >= args.threshold) for l, c in zip(labels, confs))
        if mismatched:
            errors.append(f"{mismatched} labels disagree with threshold {args.threshold}")
        print(f"rows {len(rows)}, predicted fake {sum(labels)} ({100 * sum(labels) / len(labels):.1f}%), "
              f"mean confidence {sum(confs) / len(confs):.4f}")

    if errors:
        for e in errors:
            print("FAIL:", e)
        sys.exit(1)
    print("OK: all checks passed")


if __name__ == "__main__":
    main()
