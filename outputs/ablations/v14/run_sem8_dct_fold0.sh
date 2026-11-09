#!/bin/bash
set -euo pipefail
cd /home/sem8/deepfake/v14
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
py=/home/sem8/deepfake/v14/env/bin/python

mkdir -p results/fold0/dct_consistency
echo "Starting dct_b0 Fold 0 consistency training at $(date -Is)"
"$py" -u src/train.py \
    --arch dct_b0 \
    --epochs 12 \
    --save_at 12 \
    --lr 5e-4 \
    --ema_decay 0.99 \
    --backbone_lr_mult 1.0 \
    --label_smoothing 0.05 \
    --aug_level strong \
    --conflict drop \
    --mosaic_p 0.5 \
    --mosaic_max 16 \
    --bs 64 \
    --num_workers 4 \
    --seed 42 \
    --split_csv data/splits/split_v1.csv \
    --out_dir results/fold0/dct_consistency \
    --paired_target_mix 0.5
"$py" -u outputs/ablations/v14/score_views.py --split data/splits/split_v1.csv --run results/fold0/dct_consistency
echo "dct_b0 Fold 0 consistency completed at $(date -Is)"
