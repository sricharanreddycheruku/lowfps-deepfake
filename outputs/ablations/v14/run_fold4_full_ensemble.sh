#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/deepfake/v14"
umask 077
export TORCH_HOME="$PWD/torch_cache" HF_HOME="$PWD/hf_cache"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
split="data/splits/split_v1_fold4.csv"
round="fold4"

wait_gpu() {
    for attempt in $(seq 1 12); do
        read -r free util < <(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ,)
        if [ "$free" -ge 10000 ] && [ "$util" -lt 25 ]; then return 0; fi
        sleep 10
    done
    [ "$free" -ge 10000 ] && [ "$util" -lt 25 ] || { echo 'GPU stayed busy'; exit 1; }
}

# 1. ResNet-50 Paired Consistency
out_res_cons="results/$round/consistency"
if [ ! -s "$out_res_cons/validation_logits.npz" ]; then
    wait_gpu
    echo "Starting $round/consistency at $(date -Is)"
    env/bin/python -u src/train.py --arch resnet50 --epochs 12 --save_at 12 --lr 1e-3 --ema_decay 0.99 --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong --conflict drop --mosaic_p 0.5 --mosaic_max 16 --bs 64 --num_workers 4 --seed 42 --split_csv "$split" --out_dir "$out_res_cons" --paired_target_mix 0.5
    env/bin/python -u outputs/ablations/v14/score_views.py --split "$split" --run "$out_res_cons"
    rm -f "$out_res_cons/best.pt"
fi

# 2. ConvNeXt-Small Control Baseline
out_cnxt_base="results/$round/cnxt_small"
if [ ! -s "$out_cnxt_base/validation_logits.npz" ]; then
    wait_gpu
    echo "Starting $round/cnxt_small at $(date -Is)"
    env/bin/python -u src/train.py --arch convnext_small --epochs 12 --save_at 9 --lr 3e-4 --ema_decay 0.99 --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong --conflict drop --mosaic_p 0.5 --mosaic_max 16 --bs 48 --num_workers 4 --seed 42 --split_csv "$split" --out_dir "$out_cnxt_base"
    env/bin/python -u outputs/ablations/v14/score_views.py --split "$split" --run "$out_cnxt_base"
    rm -f "$out_cnxt_base/best.pt"
fi

# 3. ConvNeXt-Small Paired Consistency
out_cnxt_cons="results/$round/cnxt_small_consistency"
if [ ! -s "$out_cnxt_cons/validation_logits.npz" ]; then
    wait_gpu
    echo "Starting $round/cnxt_small_consistency at $(date -Is)"
    env/bin/python -u src/train.py --arch convnext_small --epochs 12 --save_at 9 --lr 3e-4 --ema_decay 0.99 --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong --conflict drop --mosaic_p 0.5 --mosaic_max 16 --bs 48 --num_workers 4 --seed 42 --split_csv "$split" --out_dir "$out_cnxt_cons" --paired_target_mix 0.5
    env/bin/python -u outputs/ablations/v14/score_views.py --split "$split" --run "$out_cnxt_cons"
    rm -f "$out_cnxt_cons/best.pt"
fi

# 4. Swin-T Control Baseline
out_swin_base="results/$round/swin_t"
if [ ! -s "$out_swin_base/validation_logits.npz" ]; then
    wait_gpu
    echo "Starting $round/swin_t at $(date -Is)"
    env/bin/python -u src/train.py --arch swin_t --epochs 12 --save_at 12 --lr 3e-4 --ema_decay 0.99 --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong --conflict drop --mosaic_p 0.5 --mosaic_max 16 --bs 48 --num_workers 4 --seed 42 --split_csv "$split" --out_dir "$out_swin_base"
    env/bin/python -u outputs/ablations/v14/score_views.py --split "$split" --run "$out_swin_base"
    rm -f "$out_swin_base/best.pt"
fi

# 5. Swin-T Paired Consistency
out_swin_cons="results/$round/swin_t_consistency"
if [ ! -s "$out_swin_cons/validation_logits.npz" ]; then
    wait_gpu
    echo "Starting $round/swin_t_consistency at $(date -Is)"
    env/bin/python -u src/train.py --arch swin_t --epochs 12 --save_at 12 --lr 3e-4 --ema_decay 0.99 --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong --conflict drop --mosaic_p 0.5 --mosaic_max 16 --bs 48 --num_workers 4 --seed 42 --split_csv "$split" --out_dir "$out_swin_cons" --paired_target_mix 0.5
    env/bin/python -u outputs/ablations/v14/score_views.py --split "$split" --run "$out_swin_cons"
    rm -f "$out_swin_cons/best.pt"
fi

echo "Fold 4 full ensemble comparison suite completed at $(date -Is)"
