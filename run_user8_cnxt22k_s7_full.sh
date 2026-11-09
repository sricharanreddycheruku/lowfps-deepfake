#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/deepfake_work"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
py=/home/user8/x1/env-py310-torch271/bin/python

mkdir -p results/v16_full/cnxt_22k_s7
echo "Starting convnext_small_22k s7 full data training at $(date -Is)"
"$py" -u src/train.py --arch convnext_small_22k --all_data --paired_target_mix 0.5 --epochs 12 --save_at 12 --lr 3e-4 --ema_decay 0.99 --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong --conflict drop --mosaic_p 0.5 --mosaic_max 16 --bs 48 --num_workers 4 --seed 7 --split_csv data/splits/split_v1.csv --out_dir results/v16_full/cnxt_22k_s7
echo "Completed convnext_small_22k s7 full data training at $(date -Is)"
