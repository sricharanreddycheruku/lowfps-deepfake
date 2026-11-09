#!/bin/bash
set -euo pipefail
cd /home/sem8/deepfake/v14
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
py=/home/sem8/deepfake/v14/env/bin/python

mkdir -p results/v17_full/srm_s42
echo "Starting srm_b0 full data seed 42 training at $(date -Is)"
"$py" -u src/train.py \
    --arch srm_b0 \
    --all_data \
    --paired_target_mix 0.5 \
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
    --out_dir results/v17_full/srm_s42
echo "srm_b0 full data seed 42 completed at $(date -Is)"
