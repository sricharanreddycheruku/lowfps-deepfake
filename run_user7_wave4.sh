#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/deepfake_work"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
py=/home/user7/miniconda3/envs/gpu_env/bin/python
split="data/splits/split_v1.csv"

echo "=========================================================="
echo "STARTING USER7 WAVE 4: DEEP CONVERGENCE AT $(date -Is)"
echo "=========================================================="

mkdir -p results/v19_deep/maxvit_t_sbi_s7_e20
mkdir -p results/v19_deep/cnxt_base_s42_e20

# 1. MaxViT-T 20-Epoch Deep Training (Seed 7)
echo ">>> [1/2] Training maxvit_t_sbi 20 Epochs (Seed 7)..."
"$py" -u src/train.py \
    --arch maxvit_t \
    --all_data \
    --epochs 20 --save_at 16 --lr 3e-4 --ema_decay 0.99 \
    --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 --sbi_p 0.15 \
    --bs 32 --num_workers 4 --seed 7 --split_csv "$split" \
    --out_dir results/v19_deep/maxvit_t_sbi_s7_e20 --paired_target_mix 0.5

# 2. ConvNeXt-Base 20-Epoch Deep Training (Seed 42)
echo ">>> [2/2] Training convnext_base 20 Epochs (Seed 42)..."
"$py" -u src/train.py \
    --arch convnext_base \
    --all_data \
    --epochs 20 --save_at 16 --lr 2.5e-4 --ema_decay 0.99 \
    --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 \
    --bs 24 --num_workers 4 --seed 42 --split_csv "$split" \
    --out_dir results/v19_deep/cnxt_base_s42_e20 --paired_target_mix 0.5

echo "=========================================================="
echo "ALL USER7 WAVE 4 COMPLETED AT $(date -Is)"
echo "=========================================================="
