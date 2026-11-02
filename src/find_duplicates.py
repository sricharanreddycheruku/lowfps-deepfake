"""Find exact and near-duplicate training images and flag cross-label conflicts.

Exact duplicates are found by MD5 of the raw file bytes. Near-duplicates are
found by perceptual hash (pHash) with a Hamming distance threshold, which
also catches the exact duplicates. Every training image gets a group_id;
images in the same group are either byte-identical or visually near-identical.
A group containing both a real and a fake image is an annotation conflict
(the "annotation ambiguity" mentioned in the dataset README).

Usage:
    python src/find_duplicates.py --data_root data/raw/hackathon_dataset --out outputs/data
"""

import argparse
import csv
import hashlib
import os

import imagehash
import numpy as np
from PIL import Image

PHASH_HAMMING_THRESHOLD = 6


def list_train_images(data_root):
    """Return [(image_id, path, label)] for every training image, label
    taken from the folder (0=real, 1=fake)."""
    items = []
    for label, folder in ((0, "real"), (1, "fake")):
        d = os.path.join(data_root, "train", folder)
        for fname in sorted(os.listdir(d)):
            if not fname.lower().endswith(".jpg"):
                continue
            image_id = os.path.splitext(fname)[0]
            items.append((image_id, os.path.join(d, fname), label))
    return items


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--hamming_threshold", type=int, default=PHASH_HAMMING_THRESHOLD)
    args = parser.parse_args()

    items = list_train_images(args.data_root)
    image_ids = [x[0] for x in items]
    labels = {x[0]: x[2] for x in items}

    print(f"Hashing {len(items)} training images ...")
    md5s = {}
    phashes = {}
    for image_id, path, _label in items:
        md5s[image_id] = md5_of(path)
        with Image.open(path) as img:
            phashes[image_id] = imagehash.phash(img.convert("RGB"))

    uf = UnionFind(image_ids)

    # exact duplicates: same md5
    by_md5 = {}
    for image_id in image_ids:
        by_md5.setdefault(md5s[image_id], []).append(image_id)
    exact_dup_groups = 0
    for ids in by_md5.values():
        if len(ids) > 1:
            exact_dup_groups += 1
            for other in ids[1:]:
                uf.union(ids[0], other)

    # near duplicates: pHash Hamming distance <= threshold, checked over all
    # pairs (not just same-bucket) so no near-duplicate pair is missed.
    # Hashes are packed into uint64 and compared in row blocks to bound memory.
    hash_ints = np.array([int(str(phashes[i]), 16) for i in image_ids], dtype=np.uint64)
    n = len(image_ids)
    block = 500
    near_dup_pairs = 0
    for start in range(0, n, block):
        end = min(start + block, n)
        xor = hash_ints[start:end, None] ^ hash_ints[None, :]
        dist = np.bitwise_count(xor)
        for local_i, global_i in enumerate(range(start, end)):
            row = dist[local_i]
            close = np.nonzero(row <= args.hamming_threshold)[0]
            for global_j in close:
                if global_j <= global_i:
                    continue
                near_dup_pairs += 1
                uf.union(image_ids[global_i], image_ids[int(global_j)])

    # assign compact group ids
    roots = {}
    group_ids = {}
    for image_id in image_ids:
        r = uf.find(image_id)
        if r not in roots:
            roots[r] = len(roots)
        group_ids[image_id] = roots[r]

    # group stats
    group_members = {}
    for image_id in image_ids:
        group_members.setdefault(group_ids[image_id], []).append(image_id)

    near_dup_groups = sum(1 for members in group_members.values() if len(members) > 1)
    conflicting_groups = [
        gid for gid, members in group_members.items()
        if len({labels[m] for m in members}) > 1
    ]

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, "dup_groups.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "label", "group_id", "md5", "phash"])
        for image_id, _path, label in items:
            writer.writerow([image_id, label, group_ids[image_id], md5s[image_id], str(phashes[image_id])])

    summary = {
        "total_images": len(items),
        "exact_duplicate_groups": exact_dup_groups,
        "near_duplicate_pairs_found": near_dup_pairs,
        "groups_with_more_than_one_member": near_dup_groups,
        "conflicting_label_groups": len(conflicting_groups),
        "conflicting_group_ids": conflicting_groups,
    }
    print(summary)
    with open(os.path.join(args.out, "dup_summary.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
