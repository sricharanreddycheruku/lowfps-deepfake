#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/deepfake_work"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
py=/home/user8/x1/env-py310-torch271/bin/python
split="data/splits/split_v1_fold3.csv"
fold_dir="results/fold3"
mkdir -p "$fold_dir"

echo "=========================================================="
echo "STARTING FOLD 3 SUITE ON USER8 AT $(date -Is)"
echo "=========================================================="

# 1. Swin-Small 22k + SBI
echo ">>> [1/5] Training swin_small_22k_sbi on Fold 3..."
mkdir -p "$fold_dir/swin_22k_sbi_consistency"
"$py" -u src/train.py \
    --arch swin_small_22k \
    --epochs 12 --save_at 9 --lr 3e-4 --ema_decay 0.99 \
    --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 --sbi_p 0.15 \
    --bs 48 --num_workers 4 --seed 42 --split_csv "$split" \
    --out_dir "$fold_dir/swin_22k_sbi_consistency" --paired_target_mix 0.5
"$py" -u outputs/ablations/v14/score_views.py --split "$split" --run "$fold_dir/swin_22k_sbi_consistency"

# 2. ConvNeXt-Small 22k + SBI
echo ">>> [2/5] Training convnext_small_22k_sbi on Fold 3..."
mkdir -p "$fold_dir/cnxt_22k_sbi_consistency"
"$py" -u src/train.py \
    --arch convnext_small_22k \
    --epochs 12 --save_at 9 --lr 3e-4 --ema_decay 0.99 \
    --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 --sbi_p 0.15 \
    --bs 48 --num_workers 4 --seed 42 --split_csv "$split" \
    --out_dir "$fold_dir/cnxt_22k_sbi_consistency" --paired_target_mix 0.5
"$py" -u outputs/ablations/v14/score_views.py --split "$split" --run "$fold_dir/cnxt_22k_sbi_consistency"

# 3. MaxViT-T
echo ">>> [3/5] Training maxvit_t on Fold 3..."
mkdir -p "$fold_dir/maxvit_t_consistency"
"$py" -u src/train.py \
    --arch maxvit_t \
    --epochs 12 --save_at 9 --lr 3e-4 --ema_decay 0.99 \
    --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 \
    --bs 48 --num_workers 4 --seed 42 --split_csv "$split" \
    --out_dir "$fold_dir/maxvit_t_consistency" --paired_target_mix 0.5
"$py" -u outputs/ablations/v14/score_views.py --split "$split" --run "$fold_dir/maxvit_t_consistency"

# 4. DINO ViT-S/16 + SBI
echo ">>> [4/5] Training dino_vits16_sbi on Fold 3..."
mkdir -p "$fold_dir/dino_sbi_consistency"
"$py" -u src/train.py \
    --arch dino_vits16 \
    --epochs 12 --save_at 9 --lr 2e-4 --ema_decay 0.99 \
    --backbone_lr_mult 0.5 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 --sbi_p 0.15 \
    --bs 48 --num_workers 4 --seed 42 --split_csv "$split" \
    --out_dir "$fold_dir/dino_sbi_consistency" --paired_target_mix 0.5
"$py" -u outputs/ablations/v14/score_views.py --split "$split" --run "$fold_dir/dino_sbi_consistency"

# 5. ConvNeXt-Base
echo ">>> [5/5] Training convnext_base on Fold 3..."
mkdir -p "$fold_dir/cnxt_base"
"$py" -u src/train.py \
    --arch convnext_base \
    --epochs 12 --save_at 9 --lr 3e-4 --ema_decay 0.99 \
    --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 \
    --bs 32 --num_workers 4 --seed 42 --split_csv "$split" \
    --out_dir "$fold_dir/cnxt_base" --paired_target_mix 0.5
"$py" -u outputs/ablations/v14/score_views.py --split "$split" --run "$fold_dir/cnxt_base"

echo "=========================================================="
echo "ALL FOLD 3 MODELS COMPLETED SUCCESSFULLY AT $(date -Is)"
echo "=========================================================="
