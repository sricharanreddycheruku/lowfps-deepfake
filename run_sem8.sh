#!/bin/bash
set -e
cd /home/sem8/deepfake/v14
export HF_HOME=/home/sem8/deepfake/v14/hf_cache
export TORCH_HOME=/home/sem8/deepfake/v14/torch_cache
export PYTHONUNBUFFERED=1

echo "Starting effv2s_21k consistency training at $(date --iso-8601=seconds)"

/home/sem8/deepfake/v14/env/bin/python src/train.py \
    --arch effv2s_21k \
    --data_root data/raw/hackathon_dataset \
    --split_csv data/splits/split_v1.csv \
    --epochs 15 \
    --lr 3e-4 \
    --wd 0.05 \
    --bs 32 \
    --aug_level strong \
    --label_smoothing 0.05 \
    --seed 42 \
    --num_workers 4 \
    --paired_target_mix 0.5 \
    --out_dir results/effv2s_21k_consistency

echo "Finished training at $(date --iso-8601=seconds). Running robust evaluation..."

/home/sem8/deepfake/v14/env/bin/python src/robust_eval.py \
    --weights results/effv2s_21k_consistency/best.pt \
    --arch effv2s_21k \
    --split_csv data/splits/split_v1.csv \
    --split val \
    --out_prefix results/effv2s_21k_consistency/validation

echo "Completed effv2s_21k robust eval at $(date --iso-8601=seconds)"
