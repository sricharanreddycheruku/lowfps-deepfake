#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/deepfake/v14_inference"
umask 077
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
py=../v14/env/bin/python
for attempt in $(seq 1 12); do
    read -r free util < <(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ,)
    if [ "$free" -ge 10000 ] && [ "$util" -lt 25 ]; then break; fi
    sleep 10
done
[ "$free" -ge 10000 ] && [ "$util" -lt 25 ] || { echo 'GPU stayed busy'; exit 1; }
for spec in base1 fast1 fast2 base2; do
    arch=${spec%?}
    echo "Starting $spec at $(date -Is)"
    "$py" -u "inference_$arch.py" --test_dir images --weights weights.pt --output "$spec.csv" --batch_size 128 --num_workers 4
done
"$py" - <<'PYTHON'
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def comparison(left, right):
    a, b = read(left), read(right)
    assert len(a) == len(b) == 3000
    assert [r["image_id"] for r in a] == [r["image_id"] for r in b]
    diff = np.abs([float(x["confidence"]) - float(y["confidence"]) for x, y in zip(a, b)])
    return {
        "sha256": hashlib.sha256(Path(right).read_bytes()).hexdigest(),
        "identical": Path(left).read_bytes() == Path(right).read_bytes(),
        "max_confidence_delta": float(diff.max()),
        "label_flips": sum(x["label"] != y["label"] for x, y in zip(a, b)),
    }

report = {
    "reference_vs_base1": comparison("reference.csv", "base1.csv"),
    **{f"base1_vs_{arm}": comparison("base1.csv", f"{arm}.csv") for arm in ("base2", "fast1", "fast2")},
}
Path("parity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PYTHON
