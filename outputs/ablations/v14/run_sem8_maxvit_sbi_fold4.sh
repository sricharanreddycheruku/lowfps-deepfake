#!/usr/bin/env bash
set -euo pipefail
cd /home/sem8/deepfake/v14
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
py=/home/sem8/deepfake/v14/env/bin/python

mkdir -p results/fold4/maxvit_t_sbi_consistency
echo "Starting maxvit_t SBI consistency Fold 4 training at $(date -Is)"
"$py" -u src/train.py \
    --arch maxvit_t \
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
    --bs 48 \
    --num_workers 4 \
    --seed 42 \
    --split_csv data/splits/split_v1_fold4.csv \
    --out_dir results/fold4/maxvit_t_sbi_consistency \
    --paired_target_mix 0.5
"$py" -u outputs/ablations/v14/score_views.py --split data/splits/split_v1_fold4.csv --run results/fold4/maxvit_t_sbi_consistency
echo "maxvit_t SBI consistency Fold 4 completed at $(date -Is)"
