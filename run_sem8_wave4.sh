#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/deepfake/v14"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
py=/home/sem8/deepfake/v14/env/bin/python
split="data/splits/split_v1.csv"

echo "=========================================================="
echo "STARTING SEM8 WAVE 4: SWIN_BASE_22K (88M) AT $(date -Is)"
echo "=========================================================="

mkdir -p results/v19_full/swin_base_22k_s42
mkdir -p results/v19_full/swin_base_22k_s7

# 1. Swin-Base 22k Full Data (Seed 42)
echo ">>> [1/2] Training swin_base_22k Full Data (Seed 42)..."
"$py" -u src/train.py \
    --arch swin_base_22k \
    --all_data \
    --epochs 12 --save_at 10 --lr 1.5e-4 --ema_decay 0.99 \
    --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 --sbi_p 0.15 \
    --bs 24 --num_workers 4 --seed 42 --split_csv "$split" \
    --out_dir results/v19_full/swin_base_22k_s42 --paired_target_mix 0.5

# 2. Swin-Base 22k Full Data (Seed 7)
echo ">>> [2/2] Training swin_base_22k Full Data (Seed 7)..."
"$py" -u src/train.py \
    --arch swin_base_22k \
    --all_data \
    --epochs 12 --save_at 10 --lr 1.5e-4 --ema_decay 0.99 \
    --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 --sbi_p 0.15 \
    --bs 24 --num_workers 4 --seed 7 --split_csv "$split" \
    --out_dir results/v19_full/swin_base_22k_s7 --paired_target_mix 0.5

echo "=========================================================="
echo "ALL SEM8 WAVE 4 COMPLETED AT $(date -Is)"
echo "=========================================================="
