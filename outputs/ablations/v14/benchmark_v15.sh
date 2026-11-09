#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/deepfake/v14_inference"
umask 077
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
py=../v14/env/bin/python

cp ../v14/submission/v15_model_weights.pt v15_weights.pt

echo "Starting v15 inference benchmark on 3,000 test images at $(date -Is)"
START=$(date +%s)
"$py" -u "inference_base.py" \
    --test_dir images \
    --weights "v15_weights.pt" \
    --output "v15_predictions.csv" \
    --batch_size 128 \
    --num_workers 4
END=$(date +%s)
RUNTIME=$((END - START))
echo "v15 inference finished in $RUNTIME seconds."

"$py" - <<'PYTHON'
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

ref = read("reference.csv")
v15 = read("v15_predictions.csv")
assert len(ref) == len(v15) == 3000
assert [r["image_id"] for r in ref] == [r["image_id"] for r in v15]

p_ref = np.array([float(r["confidence"]) for r in ref])
p_v15 = np.array([float(r["confidence"]) for r in v15])
y_ref = np.array([int(r["label"]) for r in ref])
y_v15 = np.array([int(r["label"]) for r in v15])

corr = float(np.corrcoef(p_ref, p_v15)[0, 1])
diff = np.abs(p_ref - p_v15)
flips = int(np.sum(y_ref != y_v15))

stats = {
    "samples": len(v15),
    "v15_sha256": hashlib.sha256(Path("v15_predictions.csv").read_bytes()).hexdigest(),
    "runtime_seconds": int(Path("v15_predictions.csv").stat().st_mtime - Path("v15_weights.pt").stat().st_mtime),
    "pearson_correlation_with_v11": corr,
    "mean_abs_confidence_delta": float(np.mean(diff)),
    "max_abs_confidence_delta": float(np.max(diff)),
    "label_flips": flips,
    "label_flip_percentage": float(flips / len(v15) * 100),
    "v11_mean_confidence": float(np.mean(p_ref)),
    "v15_mean_confidence": float(np.mean(p_v15)),
    "v11_fake_count": int(np.sum(y_ref == 1)),
    "v15_fake_count": int(np.sum(y_v15 == 1)),
}
Path("v15_comparison.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
print(json.dumps(stats, indent=2))
PYTHON
