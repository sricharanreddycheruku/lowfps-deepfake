#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/deepfake_work"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
py=/home/user7/miniconda3/envs/gpu_env/bin/python
split="data/splits/split_v1.csv"

echo "=========================================================="
echo "STARTING USER7 WAVE 2: CONVNEXT_SMALL_22K 256x256 FULL DATA AT $(date -Is)"
echo "=========================================================="

mkdir -p results/v18_full_256/cnxt_22k_sbi_s42
mkdir -p results/v18_full_256/cnxt_22k_sbi_s7

# 1. ConvNeXt-Small 22k + SBI at 256x256 Full Data (Seed 42)
echo ">>> [1/2] Training convnext_small_22k_sbi at 256x256 Full Data (Seed 42)..."
"$py" -u src/train.py \
    --arch convnext_small_22k \
    --all_data \
    --img_size 256 \
    --epochs 12 --save_at 10 --lr 3e-4 --ema_decay 0.99 \
    --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 --sbi_p 0.15 \
    --bs 32 --num_workers 4 --seed 42 --split_csv "$split" \
    --out_dir results/v18_full_256/cnxt_22k_sbi_s42 --paired_target_mix 0.5

# 2. ConvNeXt-Small 22k + SBI at 256x256 Full Data (Seed 7)
echo ">>> [2/2] Training convnext_small_22k_sbi at 256x256 Full Data (Seed 7)..."
"$py" -u src/train.py \
    --arch convnext_small_22k \
    --all_data \
    --img_size 256 \
    --epochs 12 --save_at 10 --lr 3e-4 --ema_decay 0.99 \
    --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 --sbi_p 0.15 \
    --bs 32 --num_workers 4 --seed 7 --split_csv "$split" \
    --out_dir results/v18_full_256/cnxt_22k_sbi_s7 --paired_target_mix 0.5

echo "=========================================================="
echo "ALL USER7 WAVE 2 COMPLETED AT $(date -Is)"
echo "=========================================================="
