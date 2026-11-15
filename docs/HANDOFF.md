# Deepfake ML Hackathon (team lowfps): continuing the work

Task: binary classifier on 224x224 aligned face JPEGs, 0 = real and 1 = fake. The score is `roc_auc_score` on `confidence` (P(fake)). The 100 points split as follows:

- AUC: 50
- Inference efficiency on a T4 (under 5 min for full marks): 20
- Robustness on undisclosed degraded test variants: 20
- Documentation: 10
- Bonus for Grad-CAM, a diverse ensemble, and a TensorFlow export

Deadline: 15 Nov, with 14 Nov evening as our internal freeze.

## Where we stand

**Final system: the v20 bundle.** It has eight members:
- Swin-Small and ConvNeXt-Small with ImageNet-22k pretraining (timm), plus MaxViT-T and ConvNeXt-Base (torchvision), each at seeds 42 and 7.
- Refit on all labelled rows with the 108 conflicting-label images dropped, using the weights after epoch 9 of 12.
- Mean-logit fusion, flip TTA, threshold 0.5. Weights are stored in fp16.

The bundle has no held-out score. The evidence is a five-fold cross-validation of the same four architectures and recipe, where each image is scored by the fold models that did not see it. Pooled over the 7,000 labelled images, AUC is:
- 0.9080 clean (95% group-bootstrap interval 0.9010 to 0.9155)
- 0.9068 mean and 0.9018 worst over the 22 degraded default views
- 0.8615 on 8 px pixelation
- accuracy 0.872 at 0.5

The ship rule was written to `outputs/ablations/v20/v20_prereg.json` before these numbers were computed, and every check passed. On folds 0 and 4 the new set is ahead of the v11 recipe on every summary. The architectures were shortlisted after looking at folds 0 and 4, so the pooled figure is mildly optimistic. `python src/verify_results.py` recomputes the pooled table from the saved logits.

**Checks on the training host.**
- Two inference runs gave byte-identical `predictions.csv` (101 s each on an RTX 4060 Ti), and the format check passed.
- The Spearman correlation with the v11 test predictions is 0.9724 (a broken-model gate, at least 0.95, never used to choose).
- fp16 storage changes the confidences by at most 4.5e-4 against the fp32 bundle, with no label flip.
- `python -m unittest discover -s tests` passes (10 tests).

**Efficiency.** On a Kaggle Tesla T4 the v20 bundle took 202.1 s cold (194.8 s in the script) and 205.0 s on a second run, peak GPU memory 3,756 MiB. That is under the registered 240 s gate and the 5-minute budget. The v11 six-member bundle took 88.6 s cold on a T4.

**Previous system (v11).** Six members (ResNet-50, ConvNeXt-Small and Swin-T at seeds 42 and 7). Fold-0 AUC was 0.9075 clean and pixelation 0.8562. The TensorFlow export in this package is of v11 members, not v20 members (technical report, section 9). The Grad-CAM figures are of the shipped v20 bundle (section 8).

**Clean-room check.** A fresh Python 3.10 environment was created and only `requirements.txt` was installed (torch 2.11.0, torchvision 0.26.0, timm 1.0.31). In it, the shipped script was run on CPU with the shipped weights on the first 64 sorted test images. The confidences match `predictions.csv` to within 4.3e-4 (mean 3.3e-5) with no label flip; the small difference comes from CPU fp32 against GPU fp16.

## What is in this zip

- `source_code.zip`: all code, with `README.md` containing the exact commands, plus the unit tests in `tests/`
- `model_weights.pt`: the shipped weights (867 MB, eight members in fp16)
- `technical_report.pdf` and `report.pdf`: what was tried and why
- `predictions.csv`: the test predictions of the shipped weights
- `tf_model/`: the TensorFlow export of the v11 system
- `gradcam/`: Grad-CAM figures and notes (v20 bundle members)
- `HANDOFF.md`: this file

The dataset is not included. Get it from the organisers to retrain.

## Setup

1. Use Python 3.10 and unzip `source_code.zip`.
2. Install `requirements-train.txt`. It pins CPU wheels, so install the CUDA build of torch and torchvision instead if you have a GPU. `timm` is needed for the ImageNet-22k members.
3. Put the dataset at `data/raw/hackathon_dataset/` (`train/{real,fake}`, `test/images`, `train_metadata.csv`).
4. Run the commands in `README.md` in this order: `inspect_data`, `find_duplicates`, `make_split`, `train`, `bundle`, then the inference script.
5. One v20 member takes 12 to 19 minutes for 12 epochs on an RTX 4060 Ti; a Kaggle T4 is slower but enough.

## How the code is organised

- `src/make_split.py` builds `data/splits/split_v1.csv` and the other fold files. Near-duplicate groups never straddle train and val. Every script uses these splits, so keep them.
- `src/models.py` builds each backbone as `(N,3,H,W)` to `(N,1)` logits.
- `src/augment.py` holds the augmentations, including mosaic and Self-Blended Images.
- `src/train.py` trains with EMA weights and optional paired consistency.
- `src/robust_eval.py` runs the degradation suites.
- `src/bundle.py` builds `model_weights.pt`.
- `submission/inference_script.py` must keep the CLI `--test_dir --weights --output`.

## Already tried, with no gain beyond seed noise (about 0.003 AUC) on fold 0

- On fold 0 against the v1 to v6 bundles: ConvNeXt-Base, Swin-S, EfficientNetV2-S, RegNet-Y-16GF, MaxViT-T and ViT-B/16 as add-on members. Several of these later helped in the v20 recipe, with paired consistency, mosaic and five folds of evidence. Swin-V2-T did not train at the learning rates tried.
- Layer-wise learning-rate decay, dropping high-loss samples, more seeds, and rank-mean fusion.
- The pre-registered v5 round: SRM-noise and DCT-input members, SAM, an early-learning-regularisation loss, GridMask and FaceCut, a centre-crop TTA view, logistic stacking and a patch-level member. None met the switching rule.
- The pre-registered v6 round: ImageNet-22k/21k backbones and Self-Blended Images as single-model swaps, confident-learning label cleaning, and a simulated five-fold ensemble. None passed on fold 0.
- v9 and v10: heavy blur, downscale and noise augmentation. These lowered clean AUC.
- A DINO ViT-S/16 member (set C in v20). It lowered the pooled scores slightly.
- About 12% of validation images get confidently wrong answers, which looks like label noise.

## Ideas not yet tried

- Relabelling (not dropping) the confident-error cases; dropping them was tried in v6
- Higher-resolution face crops (the 288px test did not help, so test this carefully)
- Grad-CAM maps and a TensorFlow export of the v20 members

## Rules that disqualify us if broken

- Train only on the provided dataset. ImageNet-pretrained backbones are fine, but deepfake-specific detectors are not.
- Never use test images for training in any form, including pseudo-labels or statistics computed on them.
- The inference script must reproduce `predictions.csv` and contain no hardcoded paths.
- Register any new switching rule before scoring, and compare on pooled five-fold out-of-fold logits with the same splits.

## Known limits

- The TensorFlow path covers the v11 members only and has not been timed on a T4. The six-member v11 export passes parity at 3.0e-07 on 500 validation images.
- The final weights cannot be scored on our data. The gain from the refit and from the second seed is not measured.
- Coarse pixelation remains the weakest degradation (24 px: 0.734 pooled).
