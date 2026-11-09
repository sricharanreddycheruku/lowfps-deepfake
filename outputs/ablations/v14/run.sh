#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/deepfake/v14"
umask 077
export TORCH_HOME="$PWD/torch_cache" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
split=${1:-data/splits/split_v1.csv}
round=${2:-fold0}
shift "$(( $# < 2 ? $# : 2 ))"
if [ "$#" -eq 0 ]; then set -- control aug consistency; fi
for arm in "$@"; do
    for attempt in $(seq 1 12); do
        read -r free util < <(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ,)
        if [ "$free" -ge 10000 ] && [ "$util" -lt 25 ]; then break; fi
        sleep 10
    done
    [ "$free" -ge 10000 ] && [ "$util" -lt 25 ] || { echo 'GPU stayed busy'; exit 1; }
    case "$arm" in
        control) extra=() ;;
        aug) extra=(--paired_target_mix 0) ;;
        consistency) extra=(--paired_target_mix 0.5) ;;
        *) echo "Unknown arm: $arm"; exit 1 ;;
    esac
    out="results/$round/$arm"
    echo "Starting $round/$arm at $(date -Is)"
    env/bin/python -u src/train.py --arch resnet50 --epochs 12 --save_at 12 --lr 1e-3 --ema_decay 0.99 --backbone_lr_mult 1 --label_smoothing 0.05 --aug_level strong --conflict drop --mosaic_p 0.5 --mosaic_max 16 --bs 64 --num_workers 4 --seed 42 --split_csv "$split" --out_dir "$out" "${extra[@]}"
    env/bin/python -u outputs/ablations/v14/score_views.py --split "$split" --run "$out"
done
