#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/deepfake/v14"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
py=/home/sem8/deepfake/v14/env/bin/python
split="data/splits/split_v1.csv"

echo "=========================================================="
echo "STARTING SEM8 WAVE 3: EFFICIENTNET-V2-S 21K FULL DATA AT $(date -Is)"
echo "=========================================================="

mkdir -p results/v18_full/effv2s_21k_s42
mkdir -p results/v18_full/effv2s_21k_s7

# 1. EfficientNetV2-S 21k Full Data (Seed 42)
echo ">>> [1/2] Training effv2s_21k Full Data (Seed 42)..."
"$py" -u src/train.py \
    --arch effv2s_21k \
    --all_data \
    --epochs 12 --save_at 10 --lr 3e-4 --ema_decay 0.99 \
    --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 --sbi_p 0.15 \
    --bs 48 --num_workers 4 --seed 42 --split_csv "$split" \
    --out_dir results/v18_full/effv2s_21k_s42 --paired_target_mix 0.5

# 2. EfficientNetV2-S 21k Full Data (Seed 7)
echo ">>> [2/2] Training effv2s_21k Full Data (Seed 7)..."
"$py" -u src/train.py \
    --arch effv2s_21k \
    --all_data \
    --epochs 12 --save_at 10 --lr 3e-4 --ema_decay 0.99 \
    --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 --sbi_p 0.15 \
    --bs 48 --num_workers 4 --seed 7 --split_csv "$split" \
    --out_dir results/v18_full/effv2s_21k_s7 --paired_target_mix 0.5

echo "=========================================================="
echo "ALL SEM8 WAVE 3 COMPLETED AT $(date -Is)"
echo "=========================================================="
