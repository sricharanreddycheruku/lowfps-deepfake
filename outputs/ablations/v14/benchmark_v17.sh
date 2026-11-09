#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/deepfake/v14_inference"
umask 077
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
py=../v14/env/bin/python

cp ../v14/submission/v17_model_weights.pt v17_weights.pt

echo "Starting v17 inference benchmark on 3,000 test images at $(date -Is)"
START=$(date +%s)
"$py" -u "inference_v17.py" \
    --test_dir images \
    --weights "v17_weights.pt" \
    --output "v17_predictions.csv" \
    --batch_size 128 \
    --num_workers 4
END=$(date +%s)
RUNTIME=$((END - START))
echo "v17 inference finished in $RUNTIME seconds."

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
v16 = read("v16_predictions.csv")
v17 = read("v17_predictions.csv")
assert len(ref) == len(v16) == len(v17) == 3000

p_ref = np.array([float(r["confidence"]) for r in ref])
p_v16 = np.array([float(r["confidence"]) for r in v16])
p_v17 = np.array([float(r["confidence"]) for r in v17])
y_ref = np.array([int(r["label"]) for r in ref])
y_v16 = np.array([int(r["label"]) for r in v16])
y_v17 = np.array([int(r["label"]) for r in v17])

corr_v11 = float(np.corrcoef(p_ref, p_v17)[0, 1])
corr_v16 = float(np.corrcoef(p_v16, p_v17)[0, 1])
flips_v11 = int(np.sum(y_ref != y_v17))
flips_v16 = int(np.sum(y_v16 != y_v17))

stats = {
    "samples": len(v17),
    "v17_sha256": hashlib.sha256(Path("v17_predictions.csv").read_bytes()).hexdigest(),
    "runtime_seconds": int(Path("v17_predictions.csv").stat().st_mtime - Path("v17_weights.pt").stat().st_mtime),
    "pearson_corr_v11": corr_v11,
    "pearson_corr_v16": corr_v16,
    "flips_vs_v11": flips_v11,
    "flips_vs_v16": flips_v16,
    "v11_fake_count": int(np.sum(y_ref == 1)),
    "v16_fake_count": int(np.sum(y_v16 == 1)),
    "v17_fake_count": int(np.sum(y_v17 == 1)),
    "mean_conf_v11": float(np.mean(p_ref)),
    "mean_conf_v16": float(np.mean(p_v16)),
    "mean_conf_v17": float(np.mean(p_v17)),
}
Path("v17_comparison.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
print(json.dumps(stats, indent=2))
PYTHON
