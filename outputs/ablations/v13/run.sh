#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/deepfake/v13"
umask 077
export TORCH_HOME="$PWD/torch_cache" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for loss in gce bce; do
    read -r free util < <(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ,)
    [ "$free" -ge 10000 ] && [ "$util" -lt 25 ] || { echo 'GPU is busy'; exit 1; }
    echo "Starting $loss at $(date -Is)"
    env/bin/python -u src/train.py --arch resnet50 --epochs 12 --save_at 12 --lr 1e-3 --ema_decay 0.99 --backbone_lr_mult 1 --label_smoothing 0.05 --aug_level strong --conflict drop --mosaic_p 0.5 --mosaic_max 16 --bs 64 --num_workers 4 --seed 42 --loss "$loss" --out_dir "results/$loss"
done
env/bin/python -u evaluate.py
