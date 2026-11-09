#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/deepfake/v14"
umask 077
export TORCH_HOME="$PWD/torch_cache" HF_HOME="$PWD/hf_cache"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
split=${1:-data/splits/split_v1.csv}
round=${2:-fold0}

wait_gpu() {
    for attempt in $(seq 1 12); do
        read -r free util < <(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ,)
        if [ "$free" -ge 10000 ] && [ "$util" -lt 25 ]; then return 0; fi
        sleep 10
    done
    [ "$free" -ge 10000 ] && [ "$util" -lt 25 ] || { echo 'GPU stayed busy'; exit 1; }
}

# 1. ConvNeXt-Small with Paired Consistency
out_cnxt="results/$round/cnxt_small_consistency"
if [ ! -s "$out_cnxt/validation_logits.npz" ]; then
    wait_gpu
    echo "Starting $round/cnxt_small_consistency at $(date -Is)"
    env/bin/python -u src/train.py --arch convnext_small --epochs 12 --save_at 9 --lr 3e-4 --ema_decay 0.99 --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong --conflict drop --mosaic_p 0.5 --mosaic_max 16 --bs 48 --num_workers 4 --seed 42 --split_csv "$split" --out_dir "$out_cnxt" --paired_target_mix 0.5
    env/bin/python -u outputs/ablations/v14/score_views.py --split "$split" --run "$out_cnxt"
    rm -f "$out_cnxt/best.pt"
fi

# 2. Swin-T with Paired Consistency
out_swin="results/$round/swin_t_consistency"
if [ ! -s "$out_swin/validation_logits.npz" ]; then
    wait_gpu
    echo "Starting $round/swin_t_consistency at $(date -Is)"
    env/bin/python -u src/train.py --arch swin_t --epochs 12 --save_at 12 --lr 3e-4 --ema_decay 0.99 --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong --conflict drop --mosaic_p 0.5 --mosaic_max 16 --bs 48 --num_workers 4 --seed 42 --split_csv "$split" --out_dir "$out_swin" --paired_target_mix 0.5
    env/bin/python -u outputs/ablations/v14/score_views.py --split "$split" --run "$out_swin"
    rm -f "$out_swin/best.pt"
fi

echo "Consistent ensemble members completed at $(date -Is)"
