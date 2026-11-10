#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/deepfake_work"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
py=/home/user7/miniconda3/envs/gpu_env/bin/python
split="data/splits/split_v1.csv"

echo "=========================================================="
echo "STARTING USER7 FULL DATASET (SWIN & MAXVIT) AT $(date -Is)"
echo "=========================================================="

# 1. Swin-Small 22k + SBI on 100% full data (Seed 42)
echo ">>> [1/4] Training swin_small_22k_sbi full data (Seed 42)..."
mkdir -p results/v18_full/swin_22k_sbi_s42
"$py" -u src/train.py \
    --arch swin_small_22k \
    --all_data \
    --epochs 12 --save_at 10 --lr 3e-4 --ema_decay 0.99 \
    --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 --sbi_p 0.15 \
    --bs 40 --num_workers 4 --seed 42 --split_csv "$split" \
    --out_dir results/v18_full/swin_22k_sbi_s42 --paired_target_mix 0.5

# 2. Swin-Small 22k + SBI on 100% full data (Seed 7)
echo ">>> [2/4] Training swin_small_22k_sbi full data (Seed 7)..."
mkdir -p results/v18_full/swin_22k_sbi_s7
"$py" -u src/train.py \
    --arch swin_small_22k \
    --all_data \
    --epochs 12 --save_at 10 --lr 3e-4 --ema_decay 0.99 \
    --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 --sbi_p 0.15 \
    --bs 40 --num_workers 4 --seed 7 --split_csv "$split" \
    --out_dir results/v18_full/swin_22k_sbi_s7 --paired_target_mix 0.5

# 3. MaxViT-T + SBI on 100% full data (Seed 42)
echo ">>> [3/4] Training maxvit_t_sbi full data (Seed 42)..."
mkdir -p results/v18_full/maxvit_t_sbi_s42
"$py" -u src/train.py \
    --arch maxvit_t \
    --all_data \
    --epochs 12 --save_at 10 --lr 3e-4 --ema_decay 0.99 \
    --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 --sbi_p 0.15 \
    --bs 32 --num_workers 4 --seed 42 --split_csv "$split" \
    --out_dir results/v18_full/maxvit_t_sbi_s42 --paired_target_mix 0.5

# 4. MaxViT-T + SBI on 100% full data (Seed 7)
echo ">>> [4/4] Training maxvit_t_sbi full data (Seed 7)..."
mkdir -p results/v18_full/maxvit_t_sbi_s7
"$py" -u src/train.py \
    --arch maxvit_t \
    --all_data \
    --epochs 12 --save_at 10 --lr 3e-4 --ema_decay 0.99 \
    --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 --sbi_p 0.15 \
    --bs 32 --num_workers 4 --seed 7 --split_csv "$split" \
    --out_dir results/v18_full/maxvit_t_sbi_s7 --paired_target_mix 0.5

echo "=========================================================="
echo "ALL USER7 FULL JOBS COMPLETED AT $(date -Is)"
echo "=========================================================="
