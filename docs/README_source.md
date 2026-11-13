# Deepfake ML Hackathon - team lowfps

Binary classifier for 224x224 aligned face JPEGs (0 = real, 1 = fake), scored by ROC-AUC on `confidence` = P(fake). The submitted system is an ensemble of eight members: Swin-Small and ConvNeXt-Small with ImageNet-22k pretraining (timm), MaxViT-T and ConvNeXt-Base (torchvision), each at seeds 42 and 7. All are ImageNet-pretrained backbones fine-tuned on the provided training set only. The members are refit on all labelled images with a fixed stopping epoch, and fused by the mean of logits with horizontal-flip test-time augmentation. The refit has no held-out score. In five-fold cross-validation of the same four architectures and recipe, every labelled image is scored by models that never saw it: pooled over all 7,000 images, AUC is 0.9080 clean (95% interval 0.9010 to 0.9155), 0.9068 averaged over 22 degradation variants, 0.9018 on the worst variant and 0.8615 on 8 px pixelation. The selection rule was registered before these numbers were computed (technical report, section 5). On a Kaggle Tesla T4 the eight-member bundle takes 202.1 s cold for the 3,000 test images (194.8 s inside the script), under the 5-minute budget. On the RTX 4060 Ti they take 101 s. Grad-CAM and the TensorFlow export cover the previous six-member system (v11), not the final one. Details are in `technical_report.pdf`.

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
  inference_script_tf.py TensorFlow inference (v11 export in tf_model/)
  model_weights.pt       bundle: 8 state dicts stored in fp16, fusion rule, mean/std, threshold
  predictions.csv        output on the 3,000 test images
  requirements.txt, requirements_tf.txt
tests/                   unit tests (split integrity, bundle, inference output, shipped predictions)
  verify_results.py      (in src/) recomputes the pooled out-of-fold table from saved logits
docs/                    technical report, brief report, Grad-CAM notes
data/                    dataset and split file (not part of the code archive)
outputs/                 logs, evaluation tables and figures
```

## Environment

Python 3.10 on Linux or Windows.

Inference (PyTorch): `pip install -r submission/requirements.txt` (torch 2.11.0, torchvision 0.26.0, timm 1.0.31, numpy 2.2.6, pillow 12.3.0; timm builds the two ImageNet-22k members, and no weights are downloaded at inference). The default PyPI torch wheel and the CUDA 12.8 build gave identical predictions in our tests; install the build that matches your driver. A GPU is optional; the script falls back to CPU (about 25 s for 50 images on a laptop CPU).

Training and analysis: `pip install -r requirements-train.txt`. That file pins CPU wheels used for local smoke tests; on a GPU machine install the matching CUDA builds of torch 2.11.0 and torchvision 0.26.0 first (the final models were trained with those, and the seed-7 members with torch 2.7.1), then the remaining packages (pandas, scikit-learn, matplotlib, imagehash, tqdm). Set `TORCH_HOME` to a writable folder before the first run so the ImageNet weights can be downloaded.

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

3. Train the members. Run `python src/train.py --help` for all flags. Each final member takes 12 to 19 minutes for 12 epochs on one RTX 4060 Ti. For each seed in 42 and 7, run the command below with each of the following arch flags:
   - `--arch swin_small_22k --bs 48 --sbi_p 0.15`
   - `--arch convnext_small_22k --bs 48 --sbi_p 0.15`
   - `--arch maxvit_t --bs 48`
   - `--arch convnext_base --bs 32`

   It uses the resulting `fixed.pt` (the weights after epoch 9):

```
python src/train.py <arch flags> --epochs 12 --save_at 9 --lr 3e-4 --ema_decay 0.99 --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong --conflict drop --mosaic_p 0.5 --mosaic_max 16 --paired_target_mix 0.5 --all_data --seed <seed> --split_csv data/splits/split_v1.csv --out_dir runs/<name>_s<seed>
```

For the five-fold cross-validation, the same command is run without `--all_data`, with `--split_csv data/splits/split_v1_fold<k>.csv` for each fold `k` (fold 0 is `split_v1.csv`). The pooled out-of-fold table of the report is recomputed from the saved per-image logits with `python src/verify_results.py`, and `python -m unittest discover -s tests` runs the unit tests.

The previous system (v11) used the fold-0 commands below plus `--mosaic_p 0.5 --all_data --save_at 12` (`--save_at 9` for ConvNeXt-Small) at seeds 42 and 7. A single RTX 4060 Ti needs about 9, 20 and 13 minutes for ResNet-50, ConvNeXt-Small and Swin-T on the fold-0 split.

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

6. Bundle (one file with the members' state dicts, fusion rule, normalisation and threshold). For a fold-0 bundle, check it against the validation predictions:

```
python src/bundle.py --chosen outputs/ablations/ensemble_chosen.json --out submission/model_weights.pt --verify outputs/ablations/ensemble_val_preds.csv
```

The final bundle is written from the chosen-members file of the refit, `outputs/ablations/ensemble_chosen_v20.json`. Its `members` are `v20_full_{s42,s7}_{swin22k,cnxt22k,maxvit,cnxt_base}` and its `ckpts` the eight matching `fixed.pt` files, with weights 1/8 each, `mean_logit` fusion, `tta` true and threshold 0.5. The weights are stored in fp16:

```
python src/bundle.py --chosen outputs/ablations/ensemble_chosen_v20.json --out submission/model_weights.pt --fp16
```

That bundle has no validation check file, because the refit members have seen the validation images. The previous v11 bundle was built the same way from `outputs/ablations/ensemble_chosen_v11.json` (six members, fp32).

Fusion note: the shipped rule is the per-image mean of logits over 8 members x 2 views. A rank average was used only offline in a later comparison round (`outputs/ablations/ens_v4_pool.json`; see the report, section 5) and is not part of inference.

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

8. Inference, TensorFlow (needs the `tf_model/` folder shipped with the submission). The TF export is of the previous six-member v11 system, so its output reproduces the v11 predictions, not the v20 `predictions.csv`:

```
python submission/inference_script_tf.py --test_dir data/raw/hackathon_dataset/test/images --weights tf_model --output predictions_tf.csv
```

9. Grad-CAM figures (validation images, per member and ensemble) and TensorFlow export. Both were made from v11 bundles: the Grad-CAM figures from the v11 fold-0 bundle, since refit members have seen every validation image, and the TF export from the v11 refit bundle. `<v11 bundle>` below is that file; it is not part of this submission:

```
python src/gradcam.py --weights <v11 bundle> --split_csv data/splits/split_v1.csv --out outputs/figures
python src/export_tf.py --weights <v11 bundle> --out tf_model --work work_tf
python src/export_tf.py --parity --weights <v11 bundle> --out tf_model --split_csv data/splits/split_v1.csv --n 500 --parity_json outputs/eval/tf_parity.json
```

ResNet-50 and ConvNeXt-Small go through ONNX and onnx2tf (dynamic batch). Swin-T cannot be converted by onnx2tf, so `export_tf.py` rebuilds it directly in TensorFlow ops from the PyTorch weights, with the relative position bias and the shifted-window masks stored as constants and a fixed batch of 16 (the TF inference script pads the last batch); `--only swin_t` exports just that member. The export needs the extra packages listed in `submission/requirements_tf.txt` (including `ai-edge-litert`). See section 9 of the technical report.

10. Reports: `python src/build_report.py --figures`, then `python src/build_report.py --md docs/technical_report.md --out docs/technical_report.pdf --title "Deepfake detection: technical report" --body_size 9 --extra_root outputs/figures`. The brief report is built the same way from `docs/brief.md` with `--out docs/report.pdf --body_size 9.4`.

## Hardware, timing and seeds

- Training and ablations: NVIDIA RTX 4060 Ti (16 GB) hosts, fp16 autocast; batch 48 (32 for ConvNeXt-Base) for the final members, 64 for earlier runs.
- Final eight-member bundle: on a real Tesla T4 (Kaggle, torch 2.11.0+cu128, timm 1.0.29) the cold run took 202.1 s wall (194.8 s in the script) and a second run 205.0 s (201.0 s), peak GPU memory 3,756 MiB, return code 0. As for the six-member timing, the bundle held randomly initialised fp16 weights of the same eight architectures in the real file format and the images were random-noise JPEGs. On the RTX 4060 Ti it takes 101.1 s for the 3,000 test images (batch 128, flip TTA), peak GPU memory 3,756 MiB, and two runs give byte-identical output.
- The previous six-member bundle was timed on a real Tesla T4 (Kaggle, torch 2.11.0+cu128): 3,000 images, batch 128, flip TTA, cold run 88.6 s wall (81.2 s in the script), second run 86.9 s (82.9 s), peak GPU memory 2,229 MiB. That run used randomly initialised weights of the six architectures in the real file format, because run time does not depend on the weight values. The earlier three-member bundle took 93.1 s cold (86.0 s in the script) and 55.7 s warm on the same kind of T4, and 26.8 s on the RTX 4060 Ti. The TensorFlow path was only timed on a CPU host (5 min 49 s on 24 cores for the three-member export, `outputs/eval/tf_test_run.json`), not on a T4.
- Seeds: 42 for the split; 42 and 7 for the member runs of the final system (seed 42 for the fold runs).
- Compliance: only the provided training data and general ImageNet-pretrained weights are used. That means torchvision weights, plus the timm ImageNet-22k ConvNeXt-Small and Swin-Small; no deepfake detector is used. The Self-Blended Images pseudo-fakes come from training reals only. Test images are never used for training or for any statistic.
