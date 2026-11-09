#!/bin/bash
set -e
cd /home/sem8/deepfake/v14
export HF_HOME=/home/sem8/deepfake/v14/hf_cache
export TORCH_HOME=/home/sem8/deepfake/v14/torch_cache
export PYTHONUNBUFFERED=1

mkdir -p results/v16_full/swin_22k_s7
echo "Starting swin_small_22k s7 full data training at $(date --iso-8601=seconds)"

/home/sem8/deepfake/v14/env/bin/python src/train.py \
    --arch swin_small_22k \
    --all_data \
    --paired_target_mix 0.5 \
    --epochs 12 \
    --save_at 12 \
    --lr 3e-4 \
    --ema_decay 0.99 \
    --backbone_lr_mult 1.0 \
    --label_smoothing 0.05 \
    --aug_level strong \
    --conflict drop \
    --mosaic_p 0.5 \
    --mosaic_max 16 \
    --bs 48 \
    --num_workers 4 \
    --seed 7 \
    --split_csv data/splits/split_v1.csv \
    --out_dir results/v16_full/swin_22k_s7

echo "Completed swin_small_22k s7 full data training at $(date --iso-8601=seconds)"
