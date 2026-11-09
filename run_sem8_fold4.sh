#!/bin/bash
set -e
cd /home/sem8/deepfake/v14
export HF_HOME=/home/sem8/deepfake/v14/hf_cache
export TORCH_HOME=/home/sem8/deepfake/v14/torch_cache
export PYTHONUNBUFFERED=1

echo "Starting effv2s_21k Fold 4 consistency training at $(date --iso-8601=seconds)"

/home/sem8/deepfake/v14/env/bin/python src/train.py \
    --arch effv2s_21k \
    --data_root data/raw/hackathon_dataset \
    --split_csv data/splits/split_v1_fold4.csv \
    --epochs 15 \
    --lr 3e-4 \
    --wd 0.05 \
    --bs 32 \
    --aug_level strong \
    --label_smoothing 0.05 \
    --seed 42 \
    --num_workers 4 \
    --paired_target_mix 0.5 \
    --save_at 15 \
    --out_dir results/fold4/effv2s_21k_consistency

echo "Finished training at $(date --iso-8601=seconds). Running view evaluation..."

cp results/fold4/effv2s_21k_consistency/best.pt results/fold4/effv2s_21k_consistency/fixed.pt

/home/sem8/deepfake/v14/env/bin/python outputs/ablations/v14/score_views.py \
    --split data/splits/split_v1_fold4.csv \
    --run results/fold4/effv2s_21k_consistency

echo "Completed effv2s_21k Fold 4 eval at $(date --iso-8601=seconds)"
