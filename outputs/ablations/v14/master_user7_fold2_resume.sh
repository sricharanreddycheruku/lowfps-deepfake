#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/deepfake_work"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
py=/home/user7/miniconda3/envs/gpu_env/bin/python
split="data/splits/split_v1_fold2.csv"
fold_dir="results/fold2"

echo "=========================================================="
echo "RESUMING FOLD 2 SUITE ON USER7 AT $(date -Is)"
echo "=========================================================="

# 3. MaxViT-T
echo ">>> [3/5] Training maxvit_t on Fold 2..."
mkdir -p "$fold_dir/maxvit_t_consistency"
"$py" -u src/train.py \
    --arch maxvit_t \
    --epochs 12 --save_at 9 --lr 3e-4 --ema_decay 0.99 \
    --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 \
    --bs 24 --num_workers 4 --seed 42 --split_csv "$split" \
    --out_dir "$fold_dir/maxvit_t_consistency" --paired_target_mix 0.5
"$py" -u outputs/ablations/v14/score_views.py --split "$split" --run "$fold_dir/maxvit_t_consistency"

# 4. DINO ViT-S/16 + SBI
echo ">>> [4/5] Training dino_vits16_sbi on Fold 2..."
mkdir -p "$fold_dir/dino_sbi_consistency"
"$py" -u src/train.py \
    --arch dino_vits16 \
    --epochs 12 --save_at 9 --lr 2e-4 --ema_decay 0.99 \
    --backbone_lr_mult 0.5 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 --sbi_p 0.15 \
    --bs 32 --num_workers 4 --seed 42 --split_csv "$split" \
    --out_dir "$fold_dir/dino_sbi_consistency" --paired_target_mix 0.5
"$py" -u outputs/ablations/v14/score_views.py --split "$split" --run "$fold_dir/dino_sbi_consistency"

# 5. ConvNeXt-Base
echo ">>> [5/5] Training convnext_base on Fold 2..."
mkdir -p "$fold_dir/cnxt_base"
"$py" -u src/train.py \
    --arch convnext_base \
    --epochs 12 --save_at 9 --lr 3e-4 --ema_decay 0.99 \
    --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong \
    --conflict drop --mosaic_p 0.5 --mosaic_max 16 \
    --bs 24 --num_workers 4 --seed 42 --split_csv "$split" \
    --out_dir "$fold_dir/cnxt_base" --paired_target_mix 0.5
"$py" -u outputs/ablations/v14/score_views.py --split "$split" --run "$fold_dir/cnxt_base"

echo "=========================================================="
echo "ALL FOLD 2 MODELS COMPLETED SUCCESSFULLY AT $(date -Is)"
echo "=========================================================="
