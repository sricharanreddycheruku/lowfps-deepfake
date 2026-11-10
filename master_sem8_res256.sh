#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/deepfake/v14"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
py=/home/sem8/deepfake/v14/env/bin/python
split="data/splits/split_v1.csv"

echo "=========================================================="
echo "STARTING SEM8 RES256 & FULL DATA TRACK AT $(date -Is)"
echo "=========================================================="

# 1. ConvNeXt-Small 22k + SBI at 256x256 on Fold 0
echo ">>> [1/4] Training convnext_small_22k_sbi at 256x256 on Fold 0..."
mkdir -p results/fold0_res256/cnxt_22k_sbi
"$py" -u src/train.py \
    --arch convnext_small_22k \
    --img_size 256 \
    --epochs 12 --save_at 9 --lr 3e-4 --ema_decay 0.99 \
    --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 --sbi_p 0.15 \
    --bs 32 --num_workers 4 --seed 42 --split_csv "$split" \
    --out_dir results/fold0_res256/cnxt_22k_sbi --paired_target_mix 0.5
"$py" -u outputs/ablations/v14/score_views.py --split "$split" --run results/fold0_res256/cnxt_22k_sbi

# 2. ConvNeXt-Base at 256x256 on Fold 0
echo ">>> [2/4] Training convnext_base at 256x256 on Fold 0..."
mkdir -p results/fold0_res256/cnxt_base
"$py" -u src/train.py \
    --arch convnext_base \
    --img_size 256 \
    --epochs 12 --save_at 9 --lr 3e-4 --ema_decay 0.99 \
    --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 \
    --bs 20 --num_workers 4 --seed 42 --split_csv "$split" \
    --out_dir results/fold0_res256/cnxt_base --paired_target_mix 0.5
"$py" -u outputs/ablations/v14/score_views.py --split "$split" --run results/fold0_res256/cnxt_base

# 3. DINO ViT-S/16 + SBI on 100% full dataset (Seed 42)
echo ">>> [3/4] Training dino_vits16_sbi on 100% full dataset (Seed 42)..."
mkdir -p results/v18_full/dino_sbi_s42
"$py" -u src/train.py \
    --arch dino_vits16 \
    --all_data \
    --epochs 12 --save_at 10 --lr 2e-4 --ema_decay 0.99 \
    --backbone_lr_mult 0.5 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 --sbi_p 0.15 \
    --bs 48 --num_workers 4 --seed 42 --split_csv "$split" \
    --out_dir results/v18_full/dino_sbi_s42 --paired_target_mix 0.5

# 4. DINO ViT-S/16 + SBI on 100% full dataset (Seed 7)
echo ">>> [4/4] Training dino_vits16_sbi on 100% full dataset (Seed 7)..."
mkdir -p results/v18_full/dino_sbi_s7
"$py" -u src/train.py \
    --arch dino_vits16 \
    --all_data \
    --epochs 12 --save_at 10 --lr 2e-4 --ema_decay 0.99 \
    --backbone_lr_mult 0.5 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 --sbi_p 0.15 \
    --bs 48 --num_workers 4 --seed 7 --split_csv "$split" \
    --out_dir results/v18_full/dino_sbi_s7 --paired_target_mix 0.5

echo "=========================================================="
echo "ALL SEM8 RES256 & FULL JOBS COMPLETED AT $(date -Is)"
echo "=========================================================="
