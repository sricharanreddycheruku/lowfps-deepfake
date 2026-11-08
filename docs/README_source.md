# Deepfake ML Hackathon - team lowfps

Binary classifier for 224x224 aligned face JPEGs (0 = real, 1 = fake), scored by ROC-AUC on `confidence` = P(fake). The submitted system is an ensemble of six members, ResNet-50, ConvNeXt-Small and Swin-T at two seeds each (ImageNet-pretrained torchvision backbones, fine-tuned on the provided training set only), refit on all labelled images with fixed stopping epochs, with horizontal-flip test-time augmentation and mean-logit fusion. The refit has no held-out score: the six-member set of the same recipe trained on 80% of the data reaches validation AUC 0.9075 clean and 0.9059 mean over 23 degradation variants (fold 0; best-epoch checkpoints, fixed stopping cost 0.0006 clean), and the recipe is cross-checked on folds 1 to 3 with one seed (technical report, section 5; fold 4 was not completed). Mosaic augmentation raised the 8 px pixelation variant from 0.7229 to 0.8562. Inference time on a Tesla T4 for the 3,000 test images was measured for the earlier three-member bundle (56 s warm, 93 s cold); the six-member bundle is estimated at about 143 to 190 s and has not yet been timed on a T4. Details are in `technical_report.pdf`.

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
  ensemble_pool.py       fixed candidate sets scored against the shipped one (paired bootstrap, per-variant AUC)
  stack.py               out-of-fold logistic stacking over cached member logits
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
  model_weights.pt       bundle: 6 fp32 state dicts, fusion rule, mean/std, threshold
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

3. Train the members (12 epochs each; a single RTX 4060 Ti needs about 9, 20 and 13 minutes for ResNet-50, ConvNeXt-Small and Swin-T on the fold-0 split). The final system adds `--mosaic_p 0.5 --all_data --save_at 12` (`--save_at 9` for ConvNeXt-Small) at seeds 42 and 7 and uses the resulting `fixed.pt` files. Run `python src/train.py --help` for all flags.

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

6. Bundle (one file with the six state dicts, fusion rule, normalisation and threshold) and check it against the validation predictions:

```
python src/bundle.py --chosen outputs/ablations/ensemble_chosen.json --out submission/model_weights.pt --verify outputs/ablations/ensemble_val_preds.csv
```

The final bundle uses the same command with the chosen-members file written for the six refit checkpoints (`outputs/ablations/ensemble_chosen_v11.json`, the chosen-members file of the refit: `members` are `v11_full_{s42,s7}_{resnet50,cnxt_small,swin_t}`, `ckpts` the six matching `outputs/runs/<member>/fixed.pt`, `weights` 1/6 each, `fusion` `mean_logit`, `tta` true, `threshold_policy` `fixed_0.5`, `threshold` 0.5); it has no validation check file, because the refit members have seen the validation images.

Fusion note: the shipped rule is the per-image mean of logits over 6 members x 2 views. A rank average was used only offline in a later comparison round (`outputs/ablations/ens_v4_pool.json`; see the report, section 5) and is not part of inference.

Later improvement rounds (v2 to v6) scored candidate sets against the shipped one on the same validation split. The pre-registered v5 round (spec in `outputs/ablations/v5_prereg.json`) used the commands below; none of the candidates passed the switching rule (report, section 5), so the shipped bundle is unchanged. New architectures (`srm_b0`, `dct_b0`, `patch_b0`) and the training options `--sam sam --sam_rho 0.05`, `--loss elr --elr_lambda 3.0 --elr_beta 0.7` and `--struct_drop {facecut,gridmask} --struct_p 0.5` are in `src/models.py`, `src/train.py` and `src/augment.py`.

```
python src/train.py --arch srm_b0 --lr 1e-3 --epochs 12 --seed 42 --ema_decay 0.99 --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong --conflict drop --bs 64 --out_dir outputs/runs/v5_srm_b0
python src/train.py --arch convnext_small --lr 3e-4 --epochs 12 --ema_decay 0.99 --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong --conflict drop --bs 64 --seed 42 --sam sam --sam_rho 0.05 --out_dir outputs/runs/v5_cnxt_s_sam_s42
python src/ensemble_pool.py --sets outputs/ablations/ens_v5_sets.json --fusion mean_logit --views orig,flip --out outputs/ablations/ens_v5
python src/stack.py --members final_resnet50 e_cnxt_small final_swin_t v5_patch_b0 --c 1.0 --seed 42 --final --out outputs/ablations/stack_v5.json
python src/stack.py --members final_resnet50 e_cnxt_small final_swin_t v5_srm_b0 v5_dct_b0 v5_patch_b0 --c 1.0 --seed 42 --final --out outputs/ablations/stack_v5_prereg_pool.json
```

The first stack command is the pool actually reported in the table (shipped plus the patch member); the second is the pool the pre-registration describes (shipped plus every finished add-on member). `ensemble_pool.py` also writes `<out>_per_variant.csv` (fused AUC on each of the 23 variants). The remaining v5 run commands (`dct_b0`, `patch_b0`, ELR, face cutout, GridMask, the second seeds) are listed in `v5_prereg.json`.

The pre-registered v6 round (spec and every run command in `outputs/ablations/v6_prereg.json`, results in its `results` field) also left the bundle unchanged. It needs `timm` from `requirements-train.txt` for the `convnext_small_22k`, `swin_small_22k` and `effv2s_21k` architectures. The main commands, shown for one fold or seed each:

```
python src/make_split.py --dup_csv outputs/data/dup_groups.csv --data_root data/raw/hackathon_dataset --out data/splits/split_v1_h0_v1.csv --holdout_fold 0 --val_fold 1
python src/train.py --arch convnext_small --lr 3e-4 --epochs 12 --ema_decay 0.99 --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong --conflict drop --bs 64 --seed 42 --split_csv data/splits/split_v1_h0_v1.csv --out_dir outputs/runs/v6_s0_cnxt_s_v1
python src/train.py --arch convnext_small --lr 3e-4 --epochs 12 --ema_decay 0.99 --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong --conflict drop --bs 64 --seed 42 --split_csv data/splits/split_v1.csv --sbi_p 0.3 --out_dir outputs/runs/v6_cnxt_s_sbi_s42
python src/ensemble_pool.py --sets outputs/ablations/ens_v6_sets.json --ref shipped --fusion mean_logit --views orig,flip --out outputs/ablations/ens_v6_pool
python src/ensemble_pool.py --sets outputs/ablations/ens_v6_sets_fallback.json --ref shipped --fusion mean_logit --views orig,flip --out outputs/ablations/ens_v6_fb
python src/robust_eval.py --ckpt outputs/runs/final_resnet50/best.pt outputs/runs/e_cnxt_small/best.pt outputs/runs/final_swin_t/best.pt --suite extended --out outputs/eval/shipped_ext
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

9. Grad-CAM figures (validation images, per member and ensemble) and TensorFlow export. The Grad-CAM figures in the submission come from the fold-0 bundle, since the refit members have seen every validation image:

```
python src/gradcam.py --weights submission/model_weights.pt --split_csv data/splits/split_v1.csv --out outputs/figures
python src/export_tf.py --weights submission/model_weights.pt --out tf_model --work work_tf
python src/export_tf.py --parity --weights submission/model_weights.pt --out tf_model --split_csv data/splits/split_v1.csv --n 500 --parity_json outputs/eval/tf_parity.json
```

ResNet-50 and ConvNeXt-Small go through ONNX and onnx2tf (dynamic batch). Swin-T cannot be converted by onnx2tf, so `export_tf.py` rebuilds it directly in TensorFlow ops from the PyTorch weights, with the relative position bias and the shifted-window masks stored as constants and a fixed batch of 16 (the TF inference script pads the last batch); `--only swin_t` exports just that member. The export needs the extra packages listed in `submission/requirements_tf.txt` (including `ai-edge-litert`). See section 9 of the technical report.

10. Reports: `python src/build_report.py --figures`, then `python src/build_report.py --md docs/technical_report.md --out docs/technical_report.pdf --title "Deepfake detection: technical report" --body_size 9 --extra_root outputs/figures`. The brief report is built the same way from `docs/brief.md` with `--out docs/report.pdf --body_size 9.4`.

## Hardware, timing and seeds

- Training and ablations: one NVIDIA RTX 4060 Ti (16 GB), batch 64, fp16 autocast.
- The six-member bundle was timed on a real Tesla T4 (Kaggle, torch 2.11.0+cu128): 3,000 images, batch 128, flip TTA, cold run 88.6 s wall (81.2 s in the script), second run 86.9 s (82.9 s), peak GPU memory 2,229 MiB. That run used randomly initialised weights of the six architectures in the real file format, because run time does not depend on the weight values. The earlier three-member bundle took 93.1 s cold (86.0 s in the script) and 55.7 s warm on the same kind of T4, and 26.8 s on the RTX 4060 Ti. The TensorFlow path was only timed on a CPU host (5 min 49 s on 24 cores for the three-member export, `outputs/eval/tf_test_run.json`), not on a T4.
- Seeds: 42 for the split; 42 and 7 for the member runs of the final system (seed 42 for the earlier fold-0 runs).
- Compliance: only the provided training data and ImageNet-pretrained torchvision weights are used; test images are never used for training or for any statistic.
