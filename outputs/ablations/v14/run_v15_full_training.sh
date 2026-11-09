#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/deepfake/v14"
umask 077
export TORCH_HOME="$PWD/torch_cache" HF_HOME="$PWD/hf_cache"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
split="data/splits/split_v1.csv"
out_base="results/v15_full"
mkdir -p "$out_base"

wait_gpu() {
    for attempt in $(seq 1 12); do
        read -r free util < <(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ,)
        if [ "$free" -ge 10000 ] && [ "$util" -lt 25 ]; then return 0; fi
        sleep 10
    done
    [ "$free" -ge 10000 ] && [ "$util" -lt 25 ] || { echo 'GPU stayed busy'; exit 1; }
}

# 1. ResNet-50 Seed 42
out="$out_base/resnet50_s42"
if [ ! -s "$out/fixed.pt" ]; then
    wait_gpu
    echo "Starting resnet50_s42 at $(date -Is)"
    env/bin/python -u src/train.py --arch resnet50 --epochs 12 --save_at 12 --lr 1e-3 --ema_decay 0.99 --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong --conflict drop --mosaic_p 0.5 --mosaic_max 16 --bs 64 --num_workers 4 --seed 42 --split_csv "$split" --out_dir "$out" --paired_target_mix 0.5 --all_data
fi

# 2. ResNet-50 Seed 7
out="$out_base/resnet50_s7"
if [ ! -s "$out/fixed.pt" ]; then
    wait_gpu
    echo "Starting resnet50_s7 at $(date -Is)"
    env/bin/python -u src/train.py --arch resnet50 --epochs 12 --save_at 12 --lr 1e-3 --ema_decay 0.99 --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong --conflict drop --mosaic_p 0.5 --mosaic_max 16 --bs 64 --num_workers 4 --seed 7 --split_csv "$split" --out_dir "$out" --paired_target_mix 0.5 --all_data
fi

# 3. ConvNeXt-Small Seed 42
out="$out_base/cnxt_small_s42"
if [ ! -s "$out/fixed.pt" ]; then
    wait_gpu
    echo "Starting cnxt_small_s42 at $(date -Is)"
    env/bin/python -u src/train.py --arch convnext_small --epochs 12 --save_at 9 --lr 3e-4 --ema_decay 0.99 --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong --conflict drop --mosaic_p 0.5 --mosaic_max 16 --bs 48 --num_workers 4 --seed 42 --split_csv "$split" --out_dir "$out" --paired_target_mix 0.5 --all_data
fi

# 4. ConvNeXt-Small Seed 7
out="$out_base/cnxt_small_s7"
if [ ! -s "$out/fixed.pt" ]; then
    wait_gpu
    echo "Starting cnxt_small_s7 at $(date -Is)"
    env/bin/python -u src/train.py --arch convnext_small --epochs 12 --save_at 9 --lr 3e-4 --ema_decay 0.99 --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong --conflict drop --mosaic_p 0.5 --mosaic_max 16 --bs 48 --num_workers 4 --seed 7 --split_csv "$split" --out_dir "$out" --paired_target_mix 0.5 --all_data
fi

# 5. Swin-T Seed 42
out="$out_base/swin_t_s42"
if [ ! -s "$out/fixed.pt" ]; then
    wait_gpu
    echo "Starting swin_t_s42 at $(date -Is)"
    env/bin/python -u src/train.py --arch swin_t --epochs 12 --save_at 12 --lr 3e-4 --ema_decay 0.99 --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong --conflict drop --mosaic_p 0.5 --mosaic_max 16 --bs 48 --num_workers 4 --seed 42 --split_csv "$split" --out_dir "$out" --paired_target_mix 0.5 --all_data
fi

# 6. Swin-T Seed 7
out="$out_base/swin_t_s7"
if [ ! -s "$out/fixed.pt" ]; then
    wait_gpu
    echo "Starting swin_t_s7 at $(date -Is)"
    env/bin/python -u src/train.py --arch swin_t --epochs 12 --save_at 12 --lr 3e-4 --ema_decay 0.99 --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong --conflict drop --mosaic_p 0.5 --mosaic_max 16 --bs 48 --num_workers 4 --seed 7 --split_csv "$split" --out_dir "$out" --paired_target_mix 0.5 --all_data
fi

echo "v15 full-data training suite completed at $(date -Is)"
