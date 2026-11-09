#!/bin/bash
set -euo pipefail
cd /home/sem8/deepfake/v14
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
py=/home/sem8/deepfake/v14/env/bin/python

mkdir -p results/fold4/cnxt_base_consistency
echo "Starting convnext_base Fold 4 consistency training at $(date -Is)"
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
    --bs 32 \
    --num_workers 4 \
    --seed 42 \
    --split_csv data/splits/split_v1_fold4.csv \
    --out_dir results/fold4/cnxt_base_consistency \
    --paired_target_mix 0.5
"$py" -u outputs/ablations/v14/score_views.py --split data/splits/split_v1_fold4.csv --run results/fold4/cnxt_base_consistency
echo "convnext_base Fold 4 consistency completed at $(date -Is)"
