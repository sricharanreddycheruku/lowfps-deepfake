# Deepfake ML Hackathon (team lowfps): continuing the work

Task: binary classifier on 224x224 aligned face JPEGs, 0 = real and 1 = fake. The score is `roc_auc_score` on `confidence` (P(fake)). The 100 points split as follows:

- AUC: 50
- Inference efficiency on a T4 (under 5 min for full marks): 20
- Robustness on undisclosed degraded test variants: 20
- Documentation: 10
- Bonus for Grad-CAM, a diverse ensemble, and a TensorFlow export

Deadline: 15 Nov, with 14 Nov evening as our internal freeze.

## Where we stand

The final system is the v11 bundle: six members (ResNet-50, ConvNeXt-Small and Swin-T at seeds 42 and 7), refit on all labelled rows with fixed stopping epochs (12, 9 for ConvNeXt-Small, 12), mean-logit fusion, flip TTA, threshold 0.5. It has no held-out score. Reported figures are from models trained on 80% of the data: the six-member set of the same recipe has fold-0 AUC 0.9075 clean, 0.9059 mean and 0.9010 worst of 23 variants (best-epoch checkpoints; fixed stopping cost 0.0006 clean on fold 0), and pixelate8 is 0.8562 (it was 0.7229 before the mosaic augmentation). Cross-check on folds 1 to 3 (recipe B, the shipped one, against recipe A, the older mosaic 0.15 recipe, seed 42 only; this is the final state of the run, which the deadline stopped before recipe B was scored on fold 4 and before any other seed): mean B is 0.9079 clean against 0.9073 for A, difference +0.0005 with a standard deviation of 0.0013, so no clean-AUC difference can be claimed; pixelate8 is 0.8476 against 0.8342 (+0.0133, sd 0.0060); worst-of-23 is 0.8976 against 0.8989 (-0.0013, sd 0.0019).

Where the bundle is. `model_weights.pt` (805 MB), `predictions.csv`, the TensorFlow export and the checks were copied from the GPU host (`outputs/v11_bundle`) after a sha256 check against the host manifest and are now in `submission/` and in the package; the host copy is untouched. A clean-room run of the packaged script in a fresh Python 3.10 environment (torch 2.11.0, CPU) on the first 200 test images matched the shipped `predictions.csv` to 3.6e-4 with no label flips (CPU fp32 against the host's GPU fp16, so not byte-identical). Checks done on the host: a repeat run is identical (maximum confidence difference 0, no label flips), the Spearman correlation with the previous bundle's test predictions is 0.9846 (a broken-model check only), the format check passed, and the TensorFlow export has parity 3.0e-07 on 500 validation images and 6.8e-04 against `predictions.csv` on the test set with no label flips. The status log recorded exit code 1 for the bundle step because `src/bundle.py` writes the file and then asserts that it is under 500 MB, a limit left over from the three-member bundle (this one is 805 MB; the organisers allow 5 GB). The save had completed, so the weights are complete, and the later steps (inference, repeat run, TensorFlow export and parity) all loaded and used the six members. The post-save reload and `--verify` checks of `bundle.py` did not run for this bundle. If the bundle is rebuilt, raise that assert first.

Efficiency. The three-member bundle was timed on a Kaggle T4 (56 s warm, 93 s cold). The six-member bundle was timed on a Kaggle T4 too: 88.6 s cold wall, 86.9 s second run, peak 2,229 MiB, so the registered gate (under 240 s cold) is met. That run used random weights of the same architectures and format and noise JPEGs, since run time does not depend on weight values. The 305 s host run used a shared GPU and does not count.

## What is in this zip

- `source_code.zip`: all code, with `README.md` containing the exact commands
- `model_weights.pt`: the shipped weights (805 MB for the six-member bundle)
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
- The pre-registered v5 round (E1-E9; the union candidate E10 was not scored): SRM-noise and DCT-input members, SAM, an early-learning-regularisation loss, GridMask and FaceCut, a centre-crop TTA view, logistic stacking and a patch-level member. None met the switching rule; see section 5 of the technical report.
- The best alternative was 3 ConvNeXt-Small seeds + Swin-S, at about +0.002 clean and +0.0035 worst-variant AUC. Its 95% bootstrap interval includes zero.
- The pre-registered v6 round: ImageNet-22k/21k backbones, Self-Blended Images, confident-learning label cleaning and a five-fold ensemble simulated on fold 0. No lever passed the screen; the simulated ensemble and its fallback both failed the switch rule (clean -0.0029 and -0.0014). Full five-fold training was not run. See report section 5.
- About 12% of validation images get confidently wrong answers, which looks like label noise.

- The pixelation rounds. v7 (mosaic augmentation 0.15) and v8 (probability 0.5, the winner) passed; v9 (heavy blur, downscale and noise augmentation) and v10 (heavy augmentation on top of mosaic 0.5, a second seed) found no candidate that passed. Details are in section 5 of the technical report.

## Ideas not yet tried

- Relabelling (not dropping) the confident-error cases; dropping them was tried in v6
- Higher-resolution face crops (the 288px test did not help, so test this carefully)
- Better use of the conflicting-label groups (the final recipe drops them from training)

## Rules that disqualify us if broken

- Train only on the provided dataset. ImageNet-pretrained backbones are fine, but deepfake-specific detectors are not.
- Never use test images for training in any form, including pseudo-labels or statistics computed on them.
- The inference script must reproduce `predictions.csv` and contain no hardcoded paths.
- Compare any new model on the same `split_v1` val fold. Switch the ensemble only if it gains at least 0.004 on both clean and worst-variant AUC, the 95% paired-bootstrap interval of each gain stays above zero, and the T4 time stays under 150 s.

## Known limits

- The earlier three-member `tf_model/` is the reproducible export from `src/export_tf.py`: ResNet-50 and ConvNeXt-Small go through onnx2tf and Swin-T is rebuilt directly in TensorFlow ops (fixed batch 16). Parity on 500 validation images: 4.2e-07 in confidence. `inference_script_tf.py` on the 3,000 test images (CPU host, batch 64): 5 min 49 s, passes `check_predictions.py`, max confidence difference to `predictions.csv` 1.9e-03, one label flip.
- The TensorFlow path has not been timed on a T4. The final six-member TensorFlow export passes parity (all six members converted) but its speed is unmeasured.
- The final weights cannot be scored on our data, so the expected gain from the refit (about +0.003 to +0.005 clean AUC) is an extrapolation from fold-0 learning curves.
