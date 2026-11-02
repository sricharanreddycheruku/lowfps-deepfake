# Deepfake face detection: technical report

Team lowfps - Deepfake ML Hackathon, IIIT Bangalore (Synergy '26)

Binary classifier on 224x224 aligned face JPEGs (0 = real, 1 = fake), scored by ROC-AUC on the confidence P(fake). The shipped system is an ensemble of three ImageNet-pretrained backbones (ResNet-50, ConvNeXt-Small, Swin-T) with horizontal-flip test-time augmentation and logit averaging. On our held-out validation split it reaches AUC 0.9065 clean and 0.9055 averaged over 23 degradation variants (worst variant 0.9016). Inference on 3,000 images takes 56 to 93 s on a Tesla T4.

**Where to find each scoring item.** AUC: sections 4, 5, 6. Inference efficiency: section 7. Robustness: sections 4 (augmentation), 6 (per-variant results) and 10 (limits). Documentation: this report and the source README. Bonus items: Grad-CAM in section 8, diverse ensemble in section 3, TensorFlow export in section 9.

## 1. Problem statement and approach

The task gives 7,000 labelled training faces (3,478 real, 3,522 fake) and 3,000 unlabelled test faces, some of which are degraded in ways that are not disclosed. Final ranking depends on AUC, the time to score the test set on a T4 (under 5 minutes earns full marks), robustness to degraded variants and documentation.

Rationale for the approach:

- **Pretrained general backbones, fine-tuned end to end.** The rules allow ImageNet weights but not deepfake-specific detectors. Fine-tuning every layer (backbone learning rate equal to the head's) clearly beat a lower backbone rate in our ablation (section 5).
- **Diverse members.** One convolutional network with a residual design (ResNet-50), one modern convolutional network (ConvNeXt-Small) and one shifted-window transformer (Swin-T). They make partly different errors, although on this data the errors are strongly correlated (section 6).
- **Degradation-aware training.** Random JPEG re-compression, blur, noise and down/up-scaling during training, so that degraded test variants look like more of the same rather than a new distribution.
- **Efficiency by design.** Three mid-sized networks at 224 px, fp16 autocast, flip TTA only. No large models, no multi-crop.

## 2. Data analysis

Counts and checks (all computed on the training set only; the test set was only counted):

| item | result |
| --- | --- |
| training images | 7,000 (real 3,478, fake 3,522); folders and metadata file agree exactly |
| image format | all 224x224 RGB; estimated JPEG quality is 92 for every image in both classes, so quantisation tables are not a usable shortcut |
| file size and brightness | mean file size real 14,005 B vs fake 13,762 B (1.7 percent apart); mean brightness 117.1 vs 120.1 (2.6 percent apart). Small, but they motivate brightness, contrast and re-encoding augmentation |
| exact duplicates (MD5) | 0 |
| near duplicates (pHash, Hamming distance 6 or less) | 293 pairs forming 279 groups with more than one member |
| groups with both a real and a fake label | 53 groups (108 images), the "annotation ambiguity" mentioned in the problem statement |

Split. Near-duplicate groups are kept whole and the split is a stratified group 5-fold with fold 0 as validation (seed 42): 5,592 training images and 1,408 validation images (fake share 50.25 and 50.57 percent), with zero group overlap, so near-duplicates cannot leak between training and validation. The 108 conflicting-label images live in the training part. The first experiments kept them with a soft target of 0.5; the final recipe drops them (section 5).

![Class balance of the training data](outputs/figures/data_class_balance.png|0.33)

## 3. Architecture

![Ensemble: three backbones, flip TTA, mean of logits, sigmoid](report_architecture.png)

Every member is a torchvision backbone initialised from ImageNet weights, with the classification layer replaced by Dropout(0.3) and a single-logit Linear layer. Inputs are resized to 224 (a no-op for this data), converted to tensors and normalised with ImageNet statistics. Training and inference use the same transform.

| member | family | params | head | best epoch / 12 | train time (RTX 4060 Ti) | fp16 img/s (RTX 4060 Ti, batch 128) |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet-50 | residual CNN | 23.5M | Dropout + Linear(2048, 1) | 10 | 9 min | 1,422 |
| ConvNeXt-Small | modern CNN | 49.5M | Dropout + Linear(768, 1) | 7 | 20 min | 706 |
| Swin-T | shifted-window transformer | 27.5M | Dropout + Linear(768, 1) | 12 | 13 min | 772 |

Fusion. Each member produces one logit per view (original and horizontally flipped). The ensemble logit is the unweighted mean over the 3 members and 2 views and the confidence is its sigmoid. We tested weighted logits and probability averaging; both gained less than 0.002 AUC, so the simplest rule was kept. The bundle holds the three fp32 state dicts, the fusion rule, normalisation constants, input size and decision threshold in one file of 402.6 MB.

Why ConvNeXt-Small rather than ConvNeXt-Tiny. It was the best single model on clean, mean and worst-case AUC and gave the best one-per-family ensemble, at a small cost in time. The margin over ConvNeXt-Tiny is of the same size as seed noise (section 6), so we do not claim it is a real improvement.

## 4. Training pipeline

- **Loss.** Binary cross-entropy on logits with label smoothing 0.05.
- **Optimiser.** AdamW, weight decay 0.05, batch size 64. Peak learning rate 1e-3 for ResNet-50 and 3e-4 for ConvNeXt-Small and Swin-T, the same for head and backbone.
- **Schedule.** One epoch of linear warm-up, then cosine decay to zero over 12 epochs.
- **EMA.** Exponential moving average of the weights, decay 0.99; the EMA weights are evaluated and shipped.
- **Precision.** fp16 autocast with gradient scaling on CUDA, channels-last memory format.
- **Checkpoint choice.** After each epoch the clean validation AUC and a cheap robust AUC (JPEG q50, blur sigma 1, noise std 10) are computed; the checkpoint with the best mean of the two is kept.
- **Seed.** 42 for the split and for every run.

Training augmentation (aug level "strong"), applied in this order:

| step | setting |
| --- | --- |
| random resized crop | scale 0.8 to 1.0, aspect ratio 0.9 to 1.1 |
| horizontal flip | p = 0.5 |
| rotation | up to 10 degrees |
| degradation block, applied in a freshly shuffled order | JPEG re-encode quality 30 to 95 (p 0.5); Gaussian blur sigma 0.1 to 2.0 (p 0.3); Gaussian noise std 2 to 20 (p 0.3); down-scale by 0.25 to 0.75 then up-scale (p 0.3); sharpen (p 0.2) |
| colour jitter | brightness, contrast, saturation 0.3, hue 0.05 |
| grayscale | p = 0.1 |
| random erasing | p = 0.25 |

![Augmentation preview](outputs/figures/aug_preview.png|0.55)

## 5. Hyperparameter selection and ablations

All ablations use EfficientNet-B0 (cheap) for 20 epochs on the same split and the same seed unless stated, and are scored with the 23-variant suite (section 6): clean AUC / mean AUC over variants / worst-variant AUC. Single runs, so differences below about 0.003 are inside seed noise (section 6).

| question | setting | clean | mean | worst |
| --- | --- | --- | --- | --- |
| backbone learning rate | 0.1x head (first recipe) | 0.8772 | 0.8711 | 0.7904 |
| | 1.0x head | 0.8907 | 0.8895 | 0.8838 |
| EMA (at 0.1x backbone rate) | none | 0.8775 | 0.8719 | 0.7995 |
| | decay 0.99 | 0.8772 | 0.8711 | 0.7904 |
| | decay 0.999 | 0.8613 | 0.8570 | 0.8112 |
| peak learning rate | 1e-4 | 0.8886 | 0.8839 | 0.8641 |
| | 3e-4 | 0.8907 | 0.8895 | 0.8838 |
| | 1e-3 | 0.9024 | 0.8988 | 0.8874 |
| | 3e-3 | 0.9013 | 0.8997 | 0.8906 |
| label smoothing | 0 / 0.05 / 0.1 | 0.8915 / 0.8907 / 0.8911 | 0.8898 / 0.8895 / 0.8898 | 0.8821 / 0.8838 / 0.8832 |
| augmentation level | light | 0.8938 | 0.8888 | 0.8744 |
| | strong | 0.8907 | 0.8895 | 0.8838 |
| conflicting-label images | soft target 0.5 | 0.8907 | 0.8895 | 0.8838 |
| | dropped | 0.8959 | 0.8932 | 0.8795 |
| combined | LR 1e-3, soft | 0.9024 | 0.8988 | 0.8874 |
| | LR 1e-3, dropped | 0.9018 | 0.9014 | 0.8951 |

Reading the table. Raising the backbone learning rate to the head's was the largest single gain, mostly in the worst case (0.790 to 0.884). A higher learning rate (1e-3) helped; dropping the conflicting images gave a better mean and worst case together with it. Label smoothing made no measurable difference (we kept 0.05). EMA at 0.99 neither helped nor hurt at 20 epochs and 0.999 hurt; we kept 0.99 for the 12-epoch final runs, where the averaging window is a reasonable fraction of training. Strong augmentation costs a little clean AUC and buys worst-case robustness, which is why it was kept. The first-recipe runs (10 epochs, 0.1x backbone rate: no augmentation 0.7826 clean, strong 0.7794) were much weaker; the later results suggest the optimiser setup, not augmentation, was the limit, but those runs also differ in length, so this is not a clean comparison.

![Backbone comparison at 20 epochs](outputs/figures/ablation_backbones.png|0.8)

Backbone choice (20 epochs, one run each; clean / mean / worst): EfficientNet-B0 0.8907 / 0.8895 / 0.8838; EfficientNet-V2-S 0.9016 / 0.8971 / 0.8878; ResNet-50 0.9027 / 0.8998 / 0.8931; ConvNeXt-Tiny 0.9041 / 0.9019 / 0.8942; Swin-T 0.9027 / 0.9019 / 0.8932. EfficientNet-V2-S was dropped because its mean and worst case trailed the other three.

Final-stage experiments (12 epochs, ConvNeXt-Tiny unless stated, clean / mean / worst):

| experiment | clean | mean | worst |
| --- | --- | --- | --- |
| ConvNeXt-Tiny seed 42 (final member candidate) | 0.9020 | 0.9005 | 0.8963 |
| ConvNeXt-Tiny seed 7 (seed noise) | 0.9051 | 0.9032 | 0.8967 |
| ConvNeXt-Small | 0.9062 | 0.9045 | 0.8983 |
| ConvNeXt-Tiny at input 288 | 0.9042 | 0.9016 | 0.8899 |
| CutMix p 0.5 | 0.9012 | 0.9000 | 0.8950 |
| MixUp p 0.5, alpha 0.4 | 0.9010 | 0.8995 | 0.8937 |

None of CutMix, MixUp or 288-pixel input improved over the plain recipe and 288 pixels was worse in the worst case, so all three were rejected. Two seeds of ConvNeXt-Tiny differ by 0.003 in mean AUC, which is as large as the differences between most configurations here. Hyperparameters were therefore chosen for stability and cost rather than for the last 0.001.

## 6. Results

Validation set: 1,408 images, never used for training, with near-duplicates of training images excluded by the split. Model selection (best epoch, member choice, threshold) also used this set, so the numbers are mildly optimistic for the shipped weights. We did not retrain on train plus validation because best epochs vary from 4 to 12 across members and seed noise is about 0.003, so the gain would be unverifiable.

**Ensemble versus members** (flip TTA on; AUC clean / mean over 23 variants / worst variant / std over variants):

| model | clean | mean | worst | std |
| --- | --- | --- | --- | --- |
| ResNet-50 | 0.9036 | 0.9023 | 0.8958 | 0.0022 |
| ConvNeXt-Small | 0.9060 | 0.9047 | 0.8992 | 0.0018 |
| Swin-T | 0.9025 | 0.9010 | 0.8958 | 0.0024 |
| **Ensemble (shipped)** | **0.9065** | **0.9055** | **0.9016** | **0.0015** |
| previous triple (ResNet-50 + ConvNeXt-Tiny + Swin-T) | 0.9052 | 0.9044 | 0.9011 | - |

The ensemble beats the best single model by +0.0008 mean and +0.0024 worst-case AUC. The gain is small, and a single ConvNeXt-Small with TTA is nearly as good. The members' errors are strongly correlated, so fusion cannot remove errors they all make. The most plausible reason is that part of the remaining error is label noise and hard cases (section 10), but we have not shown that directly. Flip TTA changed mean AUC by at most +0.0015 for each shipped member and lowered neither mean nor worst case.

**Threshold, accuracy and F1.** The decision threshold is fixed at 0.5. At the clean validation set the ensemble gives accuracy 0.880, F1 0.883, precision 0.870, recall 0.896 (confusion matrix: 601 true negatives, 95 false positives, 74 false negatives, 638 true positives). We compared 0.5 with the Youden-J and best-F1 thresholds (both 0.5195): mean F1 over the 23 variants is 0.8788 for 0.5 against 0.8792 for 0.5195, and 0.5 has the lower spread of F1 across variants (std 0.0065 against 0.0068). With no real difference in mean F1, the unmodified 0.5 is the safer choice under unknown degradations. The score that counts is AUC, which does not depend on the threshold.

**Robustness suite.** 23 deterministic variants of the validation images: clean; JPEG quality 90, 70, 50, 30; Gaussian blur sigma 1, 2; noise std 5, 10, 20; down-scale to 0.5 and 0.25 then back; brightness plus and minus 30; contrast 0.6 and 1.4; grayscale; centre crop 90 percent; rotation 5 degrees; horizontal flip; resize round-trips through 160 and 299; and one combination (JPEG 50, blur 1, noise 5). Per-member AUC (no TTA) on every variant:

| variant | ResNet-50 | ConvNeXt-Small | Swin-T |
| --- | --- | --- | --- |
| clean | 0.9027 | 0.9062 | 0.9018 |
| JPEG q90 / q70 | 0.9027 / 0.9017 | 0.9058 / 0.9054 | 0.9020 / 0.9009 |
| JPEG q50 / q30 | 0.9039 / 0.9030 | 0.9048 / 0.9032 | 0.8985 / 0.8967 |
| blur sigma 1 / 2 | 0.9022 / 0.8950 | 0.9060 / 0.9020 | 0.9017 / 0.8982 |
| noise std 5 / 10 / 20 | 0.9021 / 0.9003 / 0.8965 | 0.9060 / 0.9048 / 0.9015 | 0.9009 / 0.8993 / 0.8934 |
| down-scale 0.5 / 0.25 | 0.9005 / 0.8925 | 0.9052 / 0.8983 | 0.9011 / 0.8934 |
| brightness +30 / -30 | 0.8978 / 0.9020 | 0.9030 / 0.9061 | 0.8969 / 0.9010 |
| contrast 0.6 / 1.4 | 0.9004 / 0.9018 | 0.9042 / 0.9064 | 0.9009 / 0.9000 |
| grayscale | 0.9017 | 0.9027 | 0.8954 |
| centre crop 90 percent | 0.9010 | 0.9079 | 0.9035 |
| rotation 5 degrees | 0.9028 | 0.9043 | 0.9011 |
| horizontal flip | 0.9025 | 0.9039 | 0.9018 |
| resize 160 / 299 | 0.9020 / 0.9023 | 0.9058 / 0.9060 | 0.9014 / 0.9013 |
| combo JPEG 50, blur 1, noise 5 | 0.9019 | 0.9039 | 0.8996 |
| mean over 23 | 0.9008 | 0.9045 | 0.8996 |
| worst | 0.8925 | 0.8983 | 0.8934 |

![Per-variant validation AUC of the three members](report_robust_members.png)

The worst cases are the strongest down-scaling (0.25), the strongest noise (std 20) and blur sigma 2, each costing about 0.01 AUC. Nothing in the suite breaks a member. We report the fused ensemble's summary (clean, mean, worst, std) but did not tabulate its per-variant AUCs.

**Caveat on the robustness suite.** The training augmentation covers JPEG, blur, noise and down-scaling, which are also in the suite, so those rows are partly in-distribution for the model. Brightness, contrast, grayscale, crop and rotation are covered more loosely by colour jitter and the random crop and rotation. The organisers' degradations are not disclosed; the suite is our best guess, not a measurement of the hidden test variants.

## 7. Efficiency

Measured on a real Tesla T4 (Kaggle; Python 3.13, torch 2.11.0+cu128, torchvision 0.26.0, numpy 2.1.3, pillow 13.0.0). Script: `inference_script.py`, 3,000 test images, batch 128, flip TTA, fp16 autocast, all three members.

| run | wall time | time reported inside the script |
| --- | --- | --- |
| 1 (cold) | 93.1 s | 86.0 s |
| 2 (warm) | 55.7 s | 51.9 s |

Wall time includes interpreter start-up, bundle loading, model construction, image loading and CSV writing. Peak GPU memory (torch allocator) is 1,841 MiB. Both runs are far inside the 5-minute budget. We did not isolate why the first run is slower; likely contributors are a cold file cache and one-time CUDA and library initialisation, but that is not measured.

The T4 output matches the shipped `predictions.csv` (produced on an RTX 4060 Ti) with a maximum absolute confidence difference of 1.9e-4 and one label flip out of 3,000 (a sample sitting at the threshold). That is the usual fp16 kernel and batch-size effect; on the 4060 Ti we measured up to 1.7e-4 between batch sizes 32, 64 and 128 with one flip. On the 4060 Ti the whole run takes 26.8 s (batch 128), 24.9 s (batch 64), 23.4 s (batch 32), with two runs at the same batch size bit-identical.

Size: model bundle 402.6 MB (100.5M parameters in fp32), peak 1.8 GB GPU memory. The Kaggle environment used a newer Python, numpy and pillow than the pinned requirements (Python 3.10, numpy 2.2.6, pillow 12.3.0), and the script still ran unchanged.

## 8. Explainability: Grad-CAM

Implementation: our own code (`src/gradcam.py`), a forward hook and a gradient hook on the last stage of each member (last block of layer4 for ResNet-50, last stage of the features for ConvNeXt-Small, output of the final norm for Swin-T). The map is the ReLU of the gradient-weighted sum of target activations for the single fake logit, upsampled to 224x224 and scaled to 0..1 per image. Maps are 7x7 before upsampling, so they are coarse. Figures use 16 random validation images (seed 42) for each of true positive, true negative, false positive and false negative at threshold 0.5, with one figure per member, an ensemble-average figure, and a degradation figure. Region numbers below are enrichments: share of map mass in a region divided by the region's share of area (1.0 is uniform).

![Ensemble Grad-CAM, true positives (correctly detected fakes)](outputs/figures/gradcam_ensemble_tp.png|0.52)

Findings:

- **Fakes found.** All three members attend to the inner face, mostly the eye and upper-nose band (eye-band enrichment 2.4 for ResNet-50, 3.5 for ConvNeXt-Small, 2.5 for Swin-T, 2.7 for the ensemble) and the forehead and cheeks. The outer border ring is nearly ignored (0.06 to 0.33). Nothing in the maps points to a seam along the face contour or to the background.
- **Reals.** ResNet-50 and ConvNeXt-Small give weak, diffuse maps; Swin-T's are mostly empty. They show "no fake evidence found", not a positive reason for "real".
- **Errors.** False positives highlight the same inner-face region as true positives (eye-band enrichment 2.1 to 3.3), so those images carry artifacts the models treat as fake, or carry noisy labels. False negatives look like true negatives: weak attention on hairline, chin or background, often with profile, tilted, blurred or partly covered faces.
- **Under degradation.** On six images under JPEG 30, blur 2 and noise 10 the highlighted region stays the same. Correlation with the clean map for the ensemble is 0.98 (JPEG 30), 0.91 (blur 2) and 0.97 (noise 10); Swin-T's maps are almost unchanged (0.993 to 0.997) and blur moves the CNN maps most (0.90 to 0.94).

Caveat. Grad-CAM shows where the gradient is large, not what the model needs, and 7x7 maps cannot separate eyes from skin texture. The figures support "the decision uses the face interior and not the border", not a claim about a specific artifact.

![Ensemble Grad-CAM, false positives (real faces flagged as fake)](outputs/figures/gradcam_ensemble_fp.png|0.52)

## 9. TensorFlow export

The PyTorch weights were converted to TensorFlow SavedModels (PyTorch to ONNX, opset 17, then onnx2tf 1.28.8 with TensorFlow 2.17.1) and an inference script `inference_script_tf.py` runs the same ensemble: the same preprocessing, flip TTA, mean of logits and threshold, read from `meta.json` in the model folder.

- **Parity.** On 500 validation images (seeded shuffle, CPU), TensorFlow against PyTorch fp32: maximum confidence difference 4.2e-07, per-member logit differences at most 5e-06, AUC 0.923476 for both on this subset, zero label flips. Against the shipped GPU fp16 confidences the maximum difference is 1.0e-03 (AUC 0.923524).
- **Test set.** The TF script run on all 3,000 test images (CPU host, batch 64, TTA, 7 min 36 s while the host was heavily loaded) passes the format check and differs from `predictions.csv` by at most 1.9e-03 (mean 5.5e-05). We did not time the TF path on a T4, so the efficiency claim in section 7 applies to PyTorch only.
- **Limitation we want to be explicit about.** ResNet-50 and ConvNeXt-Small convert through `src/export_tf.py` with a dynamic batch dimension. Swin-T does not: onnx2tf fails in the first attention block in every variant we tried (dynamic and static batch, several flags). The Swin-T SavedModel that ships came from one earlier manual run (static batch 16, so the TF script pads the last batch) that the script cannot reproduce. Running `export_tf.py --only swin_t` is expected to fail. The Swin-T model is validated by the parity numbers above, but its conversion is not reproducible from the repository alone.
- **Packaging.** The SavedModels total 387 MB and are shipped in the `tf_model/` folder of the submission, not in the source archive.

## 10. Challenges and limitations

- **Unseen generators and distribution shift.** Validation images come from the same pool and generator mix as the training data. AUC 0.90 on that split says little about how a different generator or a harsher degradation will score. We did not test any out-of-distribution data because only the provided dataset may be used.
- **Label noise and near-duplicates.** 53 near-duplicate groups carry both labels (108 images). Dropping them helped a little (section 5). All backbones tried, from 4M to 50M parameters, plateau between 0.90 and 0.91 validation AUC, and fusion adds under 0.003. That pattern is what label noise or an irreducible hard subset would produce, but we have not proved it and cannot rule out a modelling limit shared by all ImageNet backbones.
- **Seed noise.** About 0.003 mean AUC between two seeds of the same configuration, so most single-run comparisons in section 5 and the choice of ConvNeXt-Small over ConvNeXt-Tiny are not statistically supported.
- **Selection on validation.** Epoch, member and threshold choices used the validation set; no separate untouched test split exists. There is no public leaderboard to calibrate against.
- **Robustness suite is our own.** See the caveat in section 6. Strong degradations outside it (heavy compression, very low resolution, unusual colour processing) are untested.
- **Grad-CAM resolution.** 7x7 maps; Swin-T's maps for confident reals are mostly empty by construction.
- **Hardware.** Training used an RTX 4060 Ti (local resources had no GPU), not a T4. The T4 timing is a real Kaggle T4 measurement of the inference script; the per-model img/s figures are from the 4060 Ti.
- **TF export** limits in section 9.
- **Compute and storage.** Our workstation had almost no free disk, which is why all training and large outputs lived on a remote host and on a separate drive.

## 11. Reproducibility

Everything below runs from the repository root; details and every flag are in the source README.

- **Seeds.** 42 for the split and for every training run (7 for the second ConvNeXt-Tiny seed experiment).
- **Hardware.** Training and ablations: one RTX 4060 Ti (16 GB), batch 64, about 9 + 20 + 13 minutes for the three final members. Timing: Tesla T4 (Kaggle).
- **Software.** Python 3.10, torch 2.11.0, torchvision 0.26.0, numpy 2.2.6, pillow 12.3.0 for training and inference on the GPU host (CUDA 12.8 and 13.0 builds gave identical predictions). The local requirements file pins CPU wheels used only for smoke tests. TF export and inference: TensorFlow 2.17.1, onnx 1.16.1, onnx2tf 1.28.8.
- **Order of scripts.** `inspect_data`, `find_duplicates`, `make_split`, `train` (once per member), `robust_eval`, `ensemble` (chooses members, fusion and threshold), `bundle`, then `inference_script.py`. Grad-CAM: `gradcam.py`; TF: `export_tf.py`.
- **Final member commands** (flags that differ from the defaults):

```
python src/train.py --arch resnet50 --epochs 12 --lr 1e-3 --backbone_lr_mult 1.0 --ema_decay 0.99 --conflict drop --out_dir outputs/runs/final_resnet50
python src/train.py --arch convnext_small --epochs 12 --lr 3e-4 --backbone_lr_mult 1.0 --ema_decay 0.99 --conflict drop --out_dir outputs/runs/e_cnxt_small
python src/train.py --arch swin_t --epochs 12 --lr 3e-4 --backbone_lr_mult 1.0 --ema_decay 0.99 --conflict drop --out_dir outputs/runs/final_swin_t
```

- **Determinism.** Two inference runs at the same batch size on the same machine are bit-identical; across batch sizes and GPUs the confidences differ by up to about 2e-4 (fp16), with at most one label flip in 3,000.
- **Logs.** Per-epoch logs and configs of every run are in the `logs/` folder of the submission.

## 12. Compliance statement

- Training used only the provided training set. The only external weights are torchvision ImageNet-pretrained backbones (ResNet-50, ConvNeXt-Small, Swin-T, plus EfficientNet and ConvNeXt-Tiny in ablations). No deepfake-specific detector, no extra face data.
- The test images were used for inference only: no training, no pseudo-labels, no normalisation statistics or thresholds computed from them. The test folder was counted and scored; validation (a held-out part of the training data) was used for all model selection.
- The inference script reproduces `predictions.csv` from the bundled weights, contains no hardcoded paths, and keeps the organisers' command-line arguments.
- The CSV header is `image_id,label,confidence`, with `image_id` equal to the file name without extension, `label` the thresholded class and `confidence` the ensemble P(fake).
