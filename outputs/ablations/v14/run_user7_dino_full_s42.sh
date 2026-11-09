#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/deepfake_work"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
py=/home/user7/miniconda3/envs/gpu_env/bin/python

mkdir -p results/v17_full/dino_s42
echo "Starting dino_vits16 full data seed 42 training at $(date -Is)"
"$py" -u src/train.py \
    --arch dino_vits16 \
    --all_data \
    --paired_target_mix 0.5 \
    --epochs 12 \
    --save_at 12 \
    --lr 2e-4 \
    --ema_decay 0.99 \
    --backbone_lr_mult 0.5 \
    --label_smoothing 0.05 \
    --aug_level strong \
    --conflict drop \
    --mosaic_p 0.5 \
    --mosaic_max 16 \
    --bs 48 \
    --num_workers 4 \
    --seed 42 \
    --split_csv data/splits/split_v1.csv \
    --out_dir results/v17_full/dino_s42
echo "dino_vits16 full data seed 42 completed at $(date -Is)"
