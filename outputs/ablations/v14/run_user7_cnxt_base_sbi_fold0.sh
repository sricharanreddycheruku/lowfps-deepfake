#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/deepfake_work"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
py=/home/user7/miniconda3/envs/gpu_env/bin/python

mkdir -p results/fold0/cnxt_base_sbi_consistency
echo "Starting convnext_base SBI Fold 0 consistency training at $(date -Is)"
"$py" -u src/train.py \
    --arch convnext_base \
    --epochs 12 \
    --save_at 9 \
    --lr 3e-4 \
    --ema_decay 0.99 \
    --backbone_lr_mult 1.0 \
    --label_smoothing 0.05 \
    --aug_level strong \
    --conflict drop \
    --mosaic_p 0.5 \
    --mosaic_max 16 \
    --sbi_p 0.15 \
    --bs 32 \
    --num_workers 4 \
    --seed 42 \
    --split_csv data/splits/split_v1.csv \
    --out_dir results/fold0/cnxt_base_sbi_consistency \
    --paired_target_mix 0.5
"$py" -u outputs/ablations/v14/score_views.py --split data/splits/split_v1.csv --run results/fold0/cnxt_base_sbi_consistency
echo "convnext_base SBI Fold 0 consistency completed at $(date -Is)"
