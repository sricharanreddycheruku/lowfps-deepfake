#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/deepfake/v14"
umask 077
export TORCH_HOME="$PWD/torch_cache" HF_HOME="$PWD/hf_cache"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
split=${1:-data/splits/split_v1.csv}
round=${2:-fold0}
for attempt in $(seq 1 12); do
    read -r free util < <(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ,)
    if [ "$free" -ge 10000 ] && [ "$util" -lt 25 ]; then break; fi
    sleep 10
done
[ "$free" -ge 10000 ] && [ "$util" -lt 25 ] || { echo 'GPU stayed busy'; exit 1; }
out="results/$round/dino"
echo "Starting $round/dino at $(date -Is)"
env/bin/python -u src/train.py --arch dino_vits16 --epochs 12 --save_at 12 --lr 2e-4 --ema_decay 0.99 --backbone_lr_mult 0.5 --label_smoothing 0.05 --aug_level strong --conflict drop --mosaic_p 0.5 --mosaic_max 16 --bs 48 --num_workers 4 --seed 42 --split_csv "$split" --out_dir "$out"
env/bin/python -u outputs/ablations/v14/score_views.py --split "$split" --run "$out"
