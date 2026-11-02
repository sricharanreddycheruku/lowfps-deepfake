# Deepfake ML Hackathon - team lowfps

Binary classifier for 224x224 aligned face JPEGs (0 = real, 1 = fake), scored by ROC-AUC on `confidence` = P(fake). The submitted system is an ensemble of ResNet-50, ConvNeXt-Small and Swin-T (ImageNet-pretrained torchvision backbones, fine-tuned on the provided training set only) with horizontal-flip test-time augmentation and logit averaging. Validation AUC 0.9065 clean, 0.9055 mean over 23 degradation variants. Tesla T4 inference time for the 3,000 test images: 56 to 93 s. Details are in `technical_report.pdf`.

## Folder structure

```
src/                     training, evaluation and tooling scripts
  inspect_data.py        counts, formats, JPEG quality, brightness, file-size statistics
  find_duplicates.py     perceptual-hash near-duplicate groups and label conflicts
  make_split.py          stratified group split (split_v1.csv)
  augment.py             train and eval transforms (--preview draws a grid)
  models.py              architectures and head helpers (python src/models.py self-check)
  train.py               one training run per backbone
  robust_eval.py         23-variant robustness suite
  threshold.py           threshold candidates scored across the suite
  ensemble.py            member subsets, fusion rule and TTA choice from cached logits
  bundle.py              writes the single weights file used by inference
  bench.py               throughput benchmark
  collect_runs.py, plot_ablations.py, plot_robust_baseline.py   tables and figures
  gradcam.py             Grad-CAM figures for the bundle
  export_tf.py           PyTorch -> ONNX -> TensorFlow SavedModel and parity check
  check_predictions.py   format check for a predictions CSV
  build_report.py        report figures and PDF build
submission/
  inference_script.py    PyTorch inference (self-contained, reads the bundle)
  inference_script_tf.py TensorFlow inference
  model_weights.pt       bundle: 3 fp32 state dicts, fusion rule, mean/std, threshold
  predictions.csv        output on the 3,000 test images
  requirements.txt, requirements_tf.txt
docs/                    technical report, brief report, Grad-CAM notes
data/                    dataset and split file (not part of the code archive)
outputs/                 logs, evaluation tables and figures
```

## Environment

Python 3.10 on Linux or Windows.

Inference (PyTorch): `pip install -r submission/requirements.txt` (torch 2.11.0, torchvision 0.26.0, numpy 2.2.6, pillow 12.3.0). The default PyPI torch wheel and the CUDA 12.8 build gave identical predictions in our tests; install the build that matches your driver. A GPU is optional; the script falls back to CPU (about 25 s for 50 images on a laptop CPU).

Training and analysis: `pip install -r requirements-train.txt`. That file pins CPU wheels used for local smoke tests; on a GPU machine install the matching CUDA builds of torch 2.11.0 and torchvision 0.26.0 first (the final models were trained with those), then the remaining packages (pandas, scikit-learn, matplotlib, imagehash, tqdm). Set `TORCH_HOME` to a writable folder before the first run so the ImageNet weights can be downloaded.

Reports: `pip install -r requirements-docs.txt`.

TensorFlow inference: `pip install -r submission/requirements_tf.txt` (TensorFlow 2.17.1, no PyTorch needed). Re-running the export additionally needs the packages listed in that file.

## Data

Extract the organiser's archive into a separate, empty folder so that the layout is

```
data/raw/hackathon_dataset/
  train/real/*.jpg  train/fake/*.jpg
  test/images/*.jpg
  train_metadata.csv
```

All commands below run from the repository root. Only the training folders are used for training; the test folder is used by the inference script alone.

## Reproducing the results

1. Data analysis, near-duplicates and split (seed 42, fold 0 = validation):

```
python src/inspect_data.py --data_root data/raw/hackathon_dataset --out outputs/data
python src/find_duplicates.py --data_root data/raw/hackathon_dataset --out outputs/data
python src/make_split.py --dup_csv outputs/data/dup_groups.csv --data_root data/raw/hackathon_dataset --out data/splits/split_v1.csv --seed 42 --val_frac 0.2
```

2. Optional checks: `python src/models.py` builds every architecture and `python src/augment.py --preview` draws augmented samples.

3. Train the three members (12 epochs each, seed 42; a single RTX 4060 Ti needs about 9, 20 and 13 minutes). Run `python src/train.py --help` for all flags.

```
python src/train.py --arch resnet50 --epochs 12 --lr 1e-3 --backbone_lr_mult 1.0 --ema_decay 0.99 --conflict drop --out_dir outputs/runs/final_resnet50
python src/train.py --arch convnext_small --epochs 12 --lr 3e-4 --backbone_lr_mult 1.0 --ema_decay 0.99 --conflict drop --out_dir outputs/runs/e_cnxt_small
python src/train.py --arch swin_t --epochs 12 --lr 3e-4 --backbone_lr_mult 1.0 --ema_decay 0.99 --conflict drop --out_dir outputs/runs/final_swin_t
```

Each run writes `best.pt` (EMA weights at the best epoch), `log.csv` and `config.json`. A one-epoch CPU smoke test: `python src/train.py --arch efficientnet_b0 --epochs 1 --subset_frac 0.05 --bs 16 --num_workers 0 --out_dir outputs/runs/smoke`.

4. Robustness suite on one member (or several, fused by mean probability):

```
python src/robust_eval.py --ckpt outputs/runs/final_resnet50/best.pt --split_csv data/splits/split_v1.csv --out outputs/eval/final_resnet50
```

5. Ensemble analysis (caches per-member logits, compares subsets and fusion rules, writes `outputs/ablations/ensemble_chosen.json` and the validation predictions):

```
python src/ensemble.py --ckpt outputs/runs/final_resnet50/best.pt outputs/runs/e_cnxt_small/best.pt outputs/runs/final_swin_t/best.pt --out outputs/ablations/ensemble
```

6. Bundle (one file with the three state dicts, fusion rule, normalisation and threshold) and check it against the validation predictions:

```
python src/bundle.py --chosen outputs/ablations/ensemble_chosen.json --out submission/model_weights.pt --verify outputs/ablations/ensemble_val_preds.csv
```

Optional alternative (four members: convnext_small seeds 42, 7, 123 plus swin_s, half-precision storage; needs `swin_s` support in the inference script's `build_model`):

```
python src/train.py --arch swin_s --lr 3e-4 --seed 42 --epochs 12 --ema_decay 0.99 --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong --conflict drop --bs 64 --out_dir outputs/runs/v2_swin_s
python src/ensemble.py --ckpt outputs/runs/e_cnxt_small/best.pt outputs/runs/v2_cnxt_small_s7/best.pt outputs/runs/v2_cnxt_small_s123/best.pt outputs/runs/v2_swin_s/best.pt --members e_cnxt_small v2_cnxt_small_s7 v2_cnxt_small_s123 v2_swin_s --out outputs/ablations/ens_v2c
python src/bundle.py --chosen outputs/ablations/ens_v2c_chosen.json --out model_weights_v2.pt --fp16 --max_diff 2e-3 --verify outputs/ablations/ens_v2c_val_preds.csv
```

7. Inference, PyTorch (this is the command that produces `predictions.csv`). The weights file is not inside this code archive; it is shipped next to it as `model_weights.pt`, so either copy it to `submission/model_weights.pt` first or pass its location in `--weights`:

```
python submission/inference_script.py --test_dir data/raw/hackathon_dataset/test/images --weights submission/model_weights.pt --output predictions.csv
```

Optional arguments: `--batch_size` (default 128), `--threshold` (default: the bundle's 0.5), `--num_workers`. Output columns: `image_id,label,confidence`. Check the file with `python src/check_predictions.py --pred predictions.csv --test_dir data/raw/hackathon_dataset/test/images`. Confidences can differ in the 4th decimal between GPUs and batch sizes (fp16 kernels), with at most one label flip in 3,000 in our runs; two runs on the same machine and batch size are identical.

8. Inference, TensorFlow (needs the `tf_model/` folder shipped with the submission):

```
python submission/inference_script_tf.py --test_dir data/raw/hackathon_dataset/test/images --weights tf_model --output predictions_tf.csv
```

9. Grad-CAM figures (validation images, per member and ensemble) and TensorFlow export:

```
python src/gradcam.py --weights submission/model_weights.pt --split_csv data/splits/split_v1.csv --out outputs/figures
python src/export_tf.py --weights submission/model_weights.pt --out tf_model --work work_tf
python src/export_tf.py --parity --weights submission/model_weights.pt --out tf_model --split_csv data/splits/split_v1.csv --n 500 --parity_json outputs/eval/tf_parity.json
```

The export converts ResNet-50 and ConvNeXt-Small. The Swin-T SavedModel in the shipped `tf_model/` came from a manual conversion that this script does not reproduce (onnx2tf fails in Swin-T's first attention block); `--only swin_t` is expected to fail. See section 9 of the technical report.

10. Reports: `python src/build_report.py --figures`, then `python src/build_report.py --md docs/technical_report.md --out docs/technical_report.pdf --title "Deepfake detection: technical report" --body_size 9 --extra_root outputs/figures`. The brief report is built the same way from `docs/brief.md` with `--out docs/report.pdf --body_size 9.4`.

## Hardware, timing and seeds

- Training and ablations: one NVIDIA RTX 4060 Ti (16 GB), batch 64, fp16 autocast.
- Inference timing on a real Tesla T4 (Kaggle, Python 3.13, torch 2.11.0+cu128): 3,000 images, batch 128, flip TTA, cold run 93.1 s wall (86.0 s in the script), warm run 55.7 s (51.9 s). Peak GPU memory 1,841 MiB. On the RTX 4060 Ti the same run takes 26.8 s. The TensorFlow path was only timed on a CPU host (7 min 36 s under load), not on a T4.
- Seeds: 42 for the split and all training runs.
- Compliance: only the provided training data and ImageNet-pretrained torchvision weights are used; test images are never used for training or for any statistic.
