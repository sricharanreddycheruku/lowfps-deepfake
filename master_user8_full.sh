#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/deepfake_work"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
py=/home/user8/x1/env-py310-torch271/bin/python
split="data/splits/split_v1.csv"

echo "=========================================================="
echo "STARTING USER8 FULL DATASET (CONVNEXT S & BASE) AT $(date -Is)"
echo "=========================================================="

# 1. ConvNeXt-Small 22k + SBI on 100% full data (Seed 42)
echo ">>> [1/4] Training convnext_small_22k_sbi full data (Seed 42)..."
mkdir -p results/v18_full/cnxt_22k_sbi_s42
"$py" -u src/train.py \
    --arch convnext_small_22k \
    --all_data \
    --epochs 12 --save_at 10 --lr 3e-4 --ema_decay 0.99 \
    --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 --sbi_p 0.15 \
    --bs 40 --num_workers 4 --seed 42 --split_csv "$split" \
    --out_dir results/v18_full/cnxt_22k_sbi_s42 --paired_target_mix 0.5

# 2. ConvNeXt-Small 22k + SBI on 100% full data (Seed 7)
echo ">>> [2/4] Training convnext_small_22k_sbi full data (Seed 7)..."
mkdir -p results/v18_full/cnxt_22k_sbi_s7
"$py" -u src/train.py \
    --arch convnext_small_22k \
    --all_data \
    --epochs 12 --save_at 10 --lr 3e-4 --ema_decay 0.99 \
    --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 --sbi_p 0.15 \
    --bs 40 --num_workers 4 --seed 7 --split_csv "$split" \
    --out_dir results/v18_full/cnxt_22k_sbi_s7 --paired_target_mix 0.5

# 3. ConvNeXt-Base on 100% full data (Seed 42)
echo ">>> [3/4] Training convnext_base full data (Seed 42)..."
mkdir -p results/v18_full/cnxt_base_s42
"$py" -u src/train.py \
    --arch convnext_base \
    --all_data \
    --epochs 12 --save_at 10 --lr 3e-4 --ema_decay 0.99 \
    --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 \
    --bs 24 --num_workers 4 --seed 42 --split_csv "$split" \
    --out_dir results/v18_full/cnxt_base_s42 --paired_target_mix 0.5

# 4. ConvNeXt-Base + SBI on 100% full data (Seed 7)
echo ">>> [4/4] Training convnext_base_sbi full data (Seed 7)..."
mkdir -p results/v18_full/cnxt_base_sbi_s7
"$py" -u src/train.py \
    --arch convnext_base \
    --all_data \
    --epochs 12 --save_at 10 --lr 3e-4 --ema_decay 0.99 \
    --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 --sbi_p 0.15 \
    --bs 24 --num_workers 4 --seed 7 --split_csv "$split" \
    --out_dir results/v18_full/cnxt_base_sbi_s7 --paired_target_mix 0.5

echo "=========================================================="
echo "ALL USER8 FULL JOBS COMPLETED AT $(date -Is)"
echo "=========================================================="
