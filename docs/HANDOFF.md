# Deepfake ML Hackathon (team lowfps): continuing the work

Task: binary classifier on 224x224 aligned face JPEGs, 0 = real and 1 = fake. The score is `roc_auc_score` on `confidence` (P(fake)). The 100 points split as follows:

- AUC: 50
- Inference efficiency on a T4 (under 5 min for full marks): 20
- Robustness on undisclosed degraded test variants: 20
- Documentation: 10
- Bonus for Grad-CAM, a diverse ensemble, and a TensorFlow export

Deadline: 15 Nov, with 14 Nov evening as our internal freeze.

## Where we stand

The submission is finished and verified: ResNet-50 + ConvNeXt-Small + Swin-T, mean-logit fusion, flip TTA. Validation AUC is 0.9065 clean and 0.9016 on the worst degraded variant. Inference takes about 56 s warm (93 s cold) for 3,000 images on a Kaggle T4. On a fresh Python 3.10 install the script reproduces `predictions.csv` byte for byte.

## What is in this zip

- `source_code.zip`: all code, with `README.md` containing the exact commands
- `model_weights.pt`: the shipped weights (403 MB)
- `technical_report.pdf` and `report.pdf`: what was tried and why
- `predictions.csv`: the current test predictions
- `gradcam/`: Grad-CAM figures and notes
- `HANDOFF.md`: this file

The dataset is not included. Get it from the organisers to retrain.

## Setup

1. Use Python 3.10 and unzip `source_code.zip`.
2. Install `requirements-train.txt`. It pins CPU wheels, so install the CUDA build of torch and torchvision instead if you have a GPU.
3. Put the dataset at `data/raw/hackathon_dataset/` (`train/{real,fake}`, `test/images`, `train_metadata.csv`).
4. Run the commands in `README.md` in this order: `inspect_data`, `find_duplicates`, `make_split`, `train`, `robust_eval`, `ensemble`, `bundle`.
5. Kaggle's free T4 is enough for single-model runs of 12 epochs.

## How the code is organised

- `src/make_split.py` builds `data/splits/split_v1.csv`. Near-duplicate groups never straddle train and val. Every script uses this one split, so keep it.
- `src/models.py` builds each backbone as `(N,3,H,W)` to `(N,1)` logits.
- `src/train.py` trains with EMA weights, and `src/robust_eval.py` runs the 23-variant degradation suite.
- `src/ensemble.py` and `src/bundle.py` build `model_weights.pt`.
- `submission/inference_script.py` must keep the CLI `--test_dir --weights --output`.

## Already tried, with no gain beyond seed noise (about 0.003 AUC)

- Backbones: ConvNeXt-Base, Swin-S (12 and 22 epochs), EfficientNetV2-S, RegNet-Y-16GF, MaxViT-T, ViT-B/16. Swin-V2-T did not train at the learning rates tried.
- Layer-wise learning-rate decay, dropping high-loss samples, more seeds, and rank-mean fusion.
- The best alternative was 3 ConvNeXt-Small seeds + Swin-S, at about +0.002 clean and +0.0035 worst-variant AUC. Its 95% bootstrap interval includes zero.
- About 12% of validation images get confidently wrong answers, which looks like label noise.

## Ideas not yet tried

- A noise-robust loss, or relabelling the confident-error cases
- Frequency-domain or artifact-focused inputs alongside the RGB models
- Higher-resolution face crops (the 288px test did not help, so test this carefully)
- Better use of the conflicting-label groups (they stay in train with a soft label of 0.5)

## Rules that disqualify us if broken

- Train only on the provided dataset. ImageNet-pretrained backbones are fine, but deepfake-specific detectors are not.
- Never use test images for training in any form, including pseudo-labels or statistics computed on them.
- The inference script must reproduce `predictions.csv` and contain no hardcoded paths.
- Compare any new model on the same `split_v1` val fold. Switch the ensemble only if it beats the current one on both clean and worst-variant AUC by clearly more than 0.003.

## Known limits

- The TensorFlow Swin-T model came from one manual conversion. `export_tf.py --only swin_t` will probably fail.
- The TensorFlow path has not been timed on a T4.
