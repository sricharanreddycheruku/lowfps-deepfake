#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/deepfake_work"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
py=/home/user8/x1/env-py310-torch271/bin/python
split="data/splits/split_v1.csv"

echo "=========================================================="
echo "STARTING USER8 WAVE 3: GCE NOISE-ROBUST MODELS AT $(date -Is)"
echo "=========================================================="

mkdir -p results/v18_full/cnxt_22k_gce_s42
mkdir -p results/v18_full/swin_22k_gce_s42

# 1. ConvNeXt-Small 22k with GCE (Seed 42)
echo ">>> [1/2] Training convnext_small_22k with GCE (Seed 42)..."
"$py" -u src/train.py \
    --arch convnext_small_22k \
    --all_data \
    --loss gce \
    --gce_warmup 2 \
    --epochs 12 --save_at 10 --lr 3e-4 --ema_decay 0.99 \
    --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 --sbi_p 0.15 \
    --bs 40 --num_workers 4 --seed 42 --split_csv "$split" \
    --out_dir results/v18_full/cnxt_22k_gce_s42

# 2. Swin-Small 22k with GCE (Seed 42)
echo ">>> [2/2] Training swin_small_22k with GCE (Seed 42)..."
"$py" -u src/train.py \
    --arch swin_small_22k \
    --all_data \
    --loss gce \
    --gce_warmup 2 \
    --epochs 12 --save_at 10 --lr 3e-4 --ema_decay 0.99 \
    --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 --sbi_p 0.15 \
    --bs 40 --num_workers 4 --seed 42 --split_csv "$split" \
    --out_dir results/v18_full/swin_22k_gce_s42

echo "=========================================================="
echo "ALL USER8 WAVE 3 COMPLETED AT $(date -Is)"
echo "=========================================================="
