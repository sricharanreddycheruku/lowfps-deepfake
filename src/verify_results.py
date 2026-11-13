"""Recomputes the reported out-of-fold numbers from the saved per-image logits
and checks them against the stored results.

It reruns outputs/ablations/v20/pooled_oof.py (pooled 5-fold AUC over the
7,000 labelled images, bootstrap intervals, fold 0/4 comparison with the
v11 recipe) into a temporary file and compares every number with
outputs/ablations/v20/pooled_oof.json. Exit code 0 means all match.

Run from the project root:
    python src/verify_results.py
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}{k}.")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten(v, f"{prefix}{i}.")
    else:
        yield prefix.rstrip("."), obj


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--script", default="outputs/ablations/v20/pooled_oof.py")
    ap.add_argument("--expected", default="outputs/ablations/v20/pooled_oof.json")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "pooled_oof.json"
        subprocess.run([sys.executable, args.script, "--out", str(out)],
                       check=True, stdout=subprocess.DEVNULL)
        got = dict(flatten(json.loads(out.read_text())))
    want = dict(flatten(json.loads(Path(args.expected).read_text())))

    bad = []
    for key in sorted(set(want) | set(got)):
        a, b = want.get(key), got.get(key)
        if isinstance(a, float) and isinstance(b, float):
            if abs(a - b) > args.tol:
                bad.append((key, a, b))
        elif a != b:
            bad.append((key, a, b))

    a = json.loads(Path(args.expected).read_text())
    chosen = a["decision"]["selected"]
    s = a["sets"][chosen]
    print(f"set {chosen} pooled out-of-fold, {a['n_images']} images")
    for k in ("clean_tta", "accuracy_tta", "default_mean", "default_worst", "pixelate8",
              "extended_mean", "stress_mean"):
        print(f"  {k:14} {s[k]:.5f}")
    print(f"  clean_tta 95% CI {s['clean_tta_ci95'][0]:.4f} to {s['clean_tta_ci95'][1]:.4f}")
    print(f"  checks {a['decision']['checks']}")
    if bad:
        for key, x, y in bad[:20]:
            print(f"MISMATCH {key}: stored {x}, recomputed {y}")
        sys.exit(f"FAIL: {len(bad)} of {len(want)} values differ")
    print(f"PASS: all {len(want)} stored values reproduced")


if __name__ == "__main__":
    main()
