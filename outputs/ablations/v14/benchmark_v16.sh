#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/deepfake/v14_inference"
umask 077
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
py=../v14/env/bin/python

cp ../v14/submission/v16_model_weights.pt v16_weights.pt

echo "Starting v16 inference benchmark on 3,000 test images at $(date -Is)"
START=$(date +%s)
"$py" -u "inference_v16.py" \
    --test_dir images \
    --weights "v16_weights.pt" \
    --output "v16_predictions.csv" \
    --batch_size 128 \
    --num_workers 4
END=$(date +%s)
RUNTIME=$((END - START))
echo "v16 inference finished in $RUNTIME seconds."

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
v16 = read("v16_predictions.csv")
assert len(ref) == len(v15) == len(v16) == 3000

p_ref = np.array([float(r["confidence"]) for r in ref])
p_v15 = np.array([float(r["confidence"]) for r in v15])
p_v16 = np.array([float(r["confidence"]) for r in v16])
y_ref = np.array([int(r["label"]) for r in ref])
y_v15 = np.array([int(r["label"]) for r in v15])
y_v16 = np.array([int(r["label"]) for r in v16])

corr_v11 = float(np.corrcoef(p_ref, p_v16)[0, 1])
corr_v15 = float(np.corrcoef(p_v15, p_v16)[0, 1])
flips_v11 = int(np.sum(y_ref != y_v16))
flips_v15 = int(np.sum(y_v15 != y_v16))

stats = {
    "samples": len(v16),
    "v16_sha256": hashlib.sha256(Path("v16_predictions.csv").read_bytes()).hexdigest(),
    "pearson_corr_v11": corr_v11,
    "pearson_corr_v15": corr_v15,
    "flips_vs_v11": flips_v11,
    "flips_vs_v15": flips_v15,
    "v11_fake_count": int(np.sum(y_ref == 1)),
    "v15_fake_count": int(np.sum(y_v15 == 1)),
    "v16_fake_count": int(np.sum(y_v16 == 1)),
    "mean_conf_v11": float(np.mean(p_ref)),
    "mean_conf_v15": float(np.mean(p_v15)),
    "mean_conf_v16": float(np.mean(p_v16)),
}
Path("v16_comparison.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
print(json.dumps(stats, indent=2))
PYTHON
