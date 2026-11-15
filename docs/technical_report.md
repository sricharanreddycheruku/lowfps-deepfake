# Deepfake face detection: technical report

Team lowfps - Deepfake ML Hackathon, IIIT Bangalore (Synergy '26)

Binary classifier on 224x224 aligned face JPEGs (0 = real, 1 = fake), scored by ROC-AUC on the confidence P(fake). The final system (round v20) is an ensemble of eight ImageNet-pretrained members: Swin-Small and ConvNeXt-Small with ImageNet-22k pretraining, MaxViT-T and ConvNeXt-Base, each at seeds 42 and 7. The members are refit on all labelled images (conflicting-label images dropped) with a fixed stopping epoch, and fused by the mean of logits with horizontal-flip test-time augmentation. Because the refit saw every labelled image, it has no held-out score. The figures we can defend come from five-fold cross-validation of the same four architectures and recipe: each image is scored by the models trained on the other four folds, so all 7,000 labelled images are held out exactly once. Pooled over the folds, the set reaches AUC 0.9080 clean (95% group-bootstrap interval 0.9010 to 0.9155), 0.9068 averaged over 22 degradation variants (worst variant 0.9018) and 0.8615 on 8 px pixelation. On the two folds where the previous final recipe (v11) was also cross-validated it is ahead on every summary: fold 0 clean 0.9074 against 0.9070, worst variant 0.9049 against 0.9002, pixelation 0.8677 against 0.8584; fold 4 clean 0.9038 against 0.8972, pixelation 0.8559 against 0.8364. On a Kaggle Tesla T4 the eight-member bundle runs the 3,000 test images in 202.1 s cold, under the 5-minute budget. Section 5 gives the pre-registered rule that selected this set, and section 7 the timing.

**Where to find each scoring item.** AUC: sections 4, 5, 6. Inference efficiency: section 7. Robustness: sections 4 (augmentation), 6 (per-variant results) and 10 (limits). Documentation: this report and the source README. Hyperparameters justified by ablations, including later rounds and the switching rule: section 5. Bonus items: Grad-CAM in section 8, diverse ensemble in section 3, TensorFlow export in section 9.

## 1. Problem statement and approach

The task gives 7,000 labelled training faces (3,478 real, 3,522 fake) and 3,000 unlabelled test faces, some of which are degraded in ways that are not disclosed. Final ranking depends on AUC, the time to score the test set on a T4 (under 5 minutes earns full marks), robustness to degraded variants and documentation.

Rationale for the approach:

- **Pretrained general backbones, fine-tuned end to end.** The rules allow ImageNet weights but not deepfake-specific detectors. Fine-tuning every layer (backbone learning rate equal to the head's) clearly beat a lower backbone rate in our ablation (section 5).
- **Diverse members.** Two modern convolutional networks (ConvNeXt-Small and ConvNeXt-Base), one shifted-window transformer (Swin-Small) and one hybrid of convolution and multi-axis attention (MaxViT-T); ConvNeXt-Small and Swin-Small start from ImageNet-22k pretraining. They make partly different errors, although on this data the errors are strongly correlated (section 6). The previous final system (v11) used ResNet-50, ConvNeXt-Small and Swin-T.
- **Degradation-aware training.** Random JPEG re-compression, blur, noise and down/up-scaling during training, so that degraded test variants look like more of the same rather than a new distribution.
- **Efficiency by design.** Networks of 30M to 88M parameters at 224 px, fp16 autocast, weights stored in fp16, flip TTA only, no multi-crop.

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

![Ensemble: eight members (four backbones at seeds 42 and 7), flip TTA, mean of logits, sigmoid](report_architecture.png)

Every member is an ImageNet-pretrained backbone (torchvision, or timm for the two ImageNet-22k models) with the classification layer replaced by Dropout(0.3) and a single-logit Linear layer. Inputs are resized to 224 (a no-op for this data), converted to tensors and normalised with ImageNet statistics. Training and inference use the same transform.

Members of the final system (v20). Train time is one refit run on all labelled images, 12 epochs, seed 42, on one RTX 4060 Ti.

| member | family | pretraining | params | batch | refit time |
| --- | --- | --- | --- | --- | --- |
| Swin-Small | shifted-window transformer | ImageNet-22k, then 1k (timm) | 48.8M | 48 | 15 min |
| ConvNeXt-Small | modern CNN | ImageNet-22k, then 1k (timm) | 49.5M | 48 | 12 min |
| MaxViT-T | CNN + multi-axis attention | ImageNet-1k (torchvision) | 30.4M | 48 | 17 min |
| ConvNeXt-Base | modern CNN | ImageNet-1k (torchvision) | 87.6M | 32 | 18 min |

Members of the previous final system (v11), fold-0 runs:

| member | family | params | head | best epoch / 12 | train time (RTX 4060 Ti) | fp16 img/s (RTX 4060 Ti, batch 128) |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet-50 | residual CNN | 23.5M | Dropout + Linear(2048, 1) | 10 | 9 min | 1,422 |
| ConvNeXt-Small | modern CNN | 49.5M | Dropout + Linear(768, 1) | 7 | 20 min | 706 |
| Swin-T | shifted-window transformer | 27.5M | Dropout + Linear(768, 1) | 12 | 13 min | 772 |

Fusion. Each member produces one logit per view (original and horizontally flipped). The ensemble logit is the unweighted mean over the 8 members and 2 views and the confidence is its sigmoid. This is computed per image and does not depend on any other test image. We tested weighted logits and probability averaging on the earlier bundles; both gained less than 0.002 AUC, so the simplest rule was kept. A rank average of member scores was compared offline on validation in a later round (section 5, "Later improvement rounds and the switching rule"); it is not used at inference, because ranks are taken over the whole score vector and would make each confidence depend on the other test images. The bundle holds the eight state dicts (the `fixed.pt` weights of the refit, stored in fp16), the fusion rule, normalisation constants, input size and decision threshold in one file of 867 MB (827 MiB). The inference script loads the fp16 tensors into fp32 models; against an fp32 copy of the same bundle the test confidences differ by at most 4.5e-4 with no label flip.

**Final system (round v20).** The deployed bundle holds eight members: the four architectures above, each trained twice with seeds 42 and 7 on all 6,892 labelled rows that remain after the 108 conflicting-label images are dropped from the 7,000. The recipe is the one cross-validated in section 5 (round v20): learning rate 3e-4 for head and backbone, EMA 0.99, label smoothing 0.05, strong augmentation with mosaic probability 0.5, paired-consistency training, and Self-Blended Images with probability 0.15 for the two ImageNet-22k members. The weights saved after epoch 9 of the 12-epoch cosine schedule are used for every member, with no selection on validation. Fusion is the unweighted mean of logits over 8 members and 2 views (432.6M parameters in total).

**Previous final system (round v11).** The v11 bundle held six members: the three architectures above, each trained twice with seeds 42 and 7 on all 6,892 labelled rows that remain after the 108 conflicting-label images are dropped from the 7,000 (the fold-0 members use about 80% of the labelled rows). Stopping epochs are fixed in advance rather than chosen on validation: 12 for ResNet-50 and Swin-T, 9 for ConvNeXt-Small, taken from the per-epoch mean score of 14 earlier runs per architecture; the cosine schedule still spans 12 epochs. The recipe is the v8 `p50` recipe (mosaic augmentation with probability 0.5, section 5). Fusion is the unweighted mean of logits over 6 members and 2 views, so each confidence depends only on its own image. Its bundle was fp32, 805 MB, about 201M parameters. The TensorFlow export (section 9) is of this system; the Grad-CAM figures (section 8) are of the final v20 bundle.

Why ConvNeXt-Small rather than ConvNeXt-Tiny (v11). It was the best single model on clean, mean and worst-case AUC and gave the best one-per-family ensemble, at a small cost in time. The margin over ConvNeXt-Tiny is of the same size as seed noise (section 6), so we do not claim it is a real improvement.

## 4. Training pipeline

- **Loss.** Binary cross-entropy on logits with label smoothing 0.05.
- **Optimiser.** AdamW, weight decay 0.05. v20 members: peak learning rate 3e-4 for head and backbone, batch 48 (32 for ConvNeXt-Base). v11 members: batch 64, peak learning rate 1e-3 for ResNet-50 and 3e-4 for ConvNeXt-Small and Swin-T.
- **Schedule.** One epoch of linear warm-up, then cosine decay to zero over 12 epochs.
- **EMA.** Exponential moving average of the weights, decay 0.99; the EMA weights are evaluated and shipped.
- **Precision.** fp16 autocast with gradient scaling on CUDA, channels-last memory format.
- **Checkpoint choice.** In the fold experiments, after each epoch the clean validation AUC and a cheap robust AUC (JPEG q50, blur sigma 1, noise std 10) are computed and the best mean of the two is logged. The cross-validation of section 5 and the final refit use the fixed epoch-9 weights instead, so no validation-based selection enters them.
- **Seed.** 42 for the split; the final members use seeds 42 and 7 (one copy of each architecture per seed).

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

The final recipe (`p50`, section 5) adds one step after the degradation block and before the colour jitter: a mosaic (nearest-neighbour down-scale and up-scale by a block size drawn from 4 to 16 px) with probability 0.5; `mosaic_p` and `mosaic_max` in the run's `config.json` are 0.5 and 16.

The v20 recipe adds two steps. **Self-Blended Images** (the two ImageNet-22k members only, `--sbi_p 0.15`): with probability 0.15 a real training image becomes a pseudo-fake, labelled fake, by blending a colour-, frequency- and geometry-altered copy of the image back into itself inside a soft, randomly warped elliptical face mask. It uses only the provided training reals. **Paired consistency** (`--paired_target_mix 0.5`): from the third epoch, half of each batch is also passed through a stronger degradation; the target for that extra view is the mean of the label and the model's detached prediction on the undegraded image, and its loss enters with weight 0.5 (total divided by 1.5).

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

### Later improvement rounds and the switching rule

After the shipped bundle was fixed we ran further rounds (v2 to v4: extra ConvNeXt-Small seeds, Swin-S, and EfficientNet-V2-S, RegNetY-16GF, MaxViT-T and ViT-B/16 as extra members) and compared candidate sets against the shipped one on the validation split with a paired bootstrap over images (1,000 resamples, seed 0). Gains are against the shipped set, in AUC; CI is the 95% interval. Source: `outputs/ablations/ens_v4_pool.json`.

| candidate set | members | clean | worst of 23 | clean gain [95% CI] | worst gain [95% CI] |
| --- | --- | --- | --- | --- | --- |
| shipped | 3 | 0.9068 | 0.9019 | - | - |
| v2c (ConvNeXt-S x3 seeds + Swin-S) | 4 | 0.9087 | 0.9054 | +0.0018 [-0.0011, 0.0050] | +0.0035 [-0.0007, 0.0072] |
| bigmix (shipped + v2/v3 seeds) | 9 | 0.9081 | 0.9047 | +0.0013 [-0.0010, 0.0037] | +0.0028 [-0.0004, 0.0064] |
| shipped + 4 new backbones | 7 | 0.9070 | 0.9020 | +0.0001 [-0.0023, 0.0025] | +0.0001 [-0.0029, 0.0032] |
| bigmix + 4 new backbones | 13 | 0.9078 | 0.9045 | +0.0010 [-0.0015, 0.0035] | +0.0026 [-0.0010, 0.0060] |
| shipped + EfficientNet-V2-S | 4 | 0.9063 | 0.9014 | -0.0005 [-0.0021, 0.0010] | -0.0005 [-0.0022, 0.0015] |
| shipped + RegNetY-16GF | 4 | 0.9084 | 0.9030 | +0.0015 [-0.0003, 0.0033] | +0.0011 [-0.0013, 0.0031] |
| shipped + MaxViT-T | 4 | 0.9071 | 0.9024 | +0.0002 [-0.0017, 0.0022] | +0.0005 [-0.0016, 0.0025] |
| shipped + ViT-B/16 | 4 | 0.9060 | 0.9015 | -0.0008 [-0.0025, 0.0008] | -0.0004 [-0.0022, 0.0016] |

Every confidence interval includes zero, so none of these sets is a demonstrated improvement over the shipped bundle. These rows were scored with rank-average fusion. That rule was used only in this round, to compare sets offline: on the largest set (bigmix + 4 new backbones) it gave mean AUC 0.9069 against 0.9064 for mean logit (+0.0005). No rank-fused set ships. The shipped rule is the mean of logits over 6 members and 2 views (section 3).

Two v4 attempts failed and are not in the table: MaxViT-T ran out of GPU memory twice while sharing the GPU and was rerun alone (it is in the table), and Swin-V2-T did not train at learning rates 3e-4 and 1e-4 (validation AUC about 0.69 after 5 epochs) and was dropped. Their logs are not shipped.

A candidate replaces the shipped bundle only if it gains at least 0.004 in both clean and worst-of-23 AUC against the shipped set, the lower 95% bootstrap bound of both gains is above zero, and the estimated T4 time for 3,000 images stays under 150 s. The rule was fixed before the experiments of the latest round and each candidate set is scored once. No candidate met it, so the shipped bundle is unchanged.

**Round v5 (pre-registered).** This round was written down before any job ran (`outputs/ablations/v5_prereg.json`: at most ten candidate sets, fusion weights fixed, each set scored once, no retuning). Add-on members enter at weight 0.2 (`0.8 * mean(shipped logits) + 0.2 * new logit`), swaps keep equal weights, fusion is the mean of logits with flip TTA. The ideas aimed at inputs or signals that differ from the RGB ImageNet members: noise-residual and DCT members (E1, E2), SAM and an early-learning-regularisation loss (E3, E4), structured dropout on the face (E5, E6), a centre-crop TTA view (E7), logistic stacking (E8) and a patch-level member (E9). Sources: `ens_v5_pool.json`, `stack_v5_prereg_pool.json`, `stack_v5.json`; T4 time is the 4060 Ti benchmark times 3.5.

| id | candidate | clean | worst of 23 | clean gain [95% CI] | worst gain [95% CI] | T4 est. | rule |
| --- | --- | --- | --- | --- | --- | --- | --- |
| - | shipped | 0.9065 | 0.9016 | - | - | 72 s | - |
| E1 | + SRM residual EfficientNet-B0 @0.2 | 0.9078 | 0.9026 | +0.0014 [-0.0022, 0.0053] | +0.0010 [-0.0019, 0.0038] | 80 s | fail |
| E2 | + DCT-domain CNN @0.2 | 0.9070 | 0.9023 | +0.0005 [-0.0013, 0.0023] | +0.0007 [-0.0008, 0.0019] | 73 s | fail |
| E3 | ConvNeXt-S trained with SAM (swap) | 0.9072 | 0.9027 | +0.0008 [-0.0008, 0.0024] | +0.0011 [-0.0004, 0.0029] | 72 s | fail |
| E4 | ConvNeXt-S trained with ELR (swap) | 0.9062 | 0.9009 | -0.0003 [-0.0022, 0.0018] | -0.0008 [-0.0031, 0.0019] | 72 s | fail |
| E5 | ConvNeXt-S with face-region cutout (swap) | 0.9065 | 0.9027 | 0.0000 [-0.0013, 0.0015] | +0.0010 [-0.0014, 0.0025] | 72 s | fail |
| E6 | ConvNeXt-S with GridMask (swap) | 0.9062 | 0.9018 | -0.0003 [-0.0016, 0.0010] | +0.0002 [-0.0019, 0.0014] | 72 s | fail |
| E7 | + centre-crop 90% view (crop90 TTA) | 0.9075 | 0.9041 | +0.0010 [-0.0005, 0.0025] | +0.0025 [0.0000, 0.0041] | 108 s | fail |
| E8 | logistic stacking, pre-registered pool (shipped + SRM + DCT + patch members, out-of-fold); governs | 0.9069 | 0.9022 | +0.0004 [-0.0023, 0.0033] | +0.0006 [-0.0020, 0.0030] | - | fail |
| E8* | logistic stacking, pool after dropping SRM and DCT (shipped + patch member, out-of-fold) | 0.9066 | 0.9017 | +0.0001 [-0.0024, 0.0025] | +0.0001 [-0.0023, 0.0024] | - | fail |
| E9 | + patch-level EfficientNet-B0 @0.2 | 0.9080 | 0.9032 | +0.0015 [-0.0010, 0.0040] | +0.0016 [-0.0008, 0.0040] | - | fail |

**Deviation from the registered E8 rule.** The registration said the stacking pool is the shipped three plus every v5 member finished by 10 November. We first scored a pool without `srm_b0` and `dct_b0` (row E8*), chosen after seeing that both were weak or fragile as single models and had failed E1 and E2. That removal was made after their validation results and departs from the registered rule, so we then scored the registered pool once (row E8). The registered row governs; both fail, and the conclusion does not depend on which is used. Erratum: the registered E8 command wrote the regularisation flag as `--C`, but `src/stack.py` takes `--c`; the command in `v5_prereg.json` is kept as written and the working commands are in the source README.

None of the nine candidates reaches +0.004 in both clean and worst-case AUC, and every clean interval includes zero, so the decision on 3 November was to ship the bundle unchanged (the union candidate E10 was not scored because nothing passed). The registration set the decision date at 12 November; the decision was taken earlier, on 3 November, because every registered job had finished and nothing more would have been scored. We disclose this as a timing deviation from the registration. The SRM member alone is fragile (clean 0.8639, worst 0.6754 under Gaussian noise with standard deviation 20; it is also weak under noise std 10 and the JPEG-blur-noise combination) and the DCT member alone is weak (0.6444). As single ConvNeXt-S models (flip TTA, `outputs/ablations/v5_single_cnxt.json`), against the baseline of the same seed the SAM runs changed clean AUC by +0.0013 (seed 42) and -0.0028 (seed 7) and the ELR runs by -0.0007 and -0.0012, which is seed noise or worse (the follow-up retrains of the other two members were therefore not triggered). A clean-AUC ceiling near 0.91 held across every backbone, loss, augmentation and input change we tried.

**Round v6 (pre-registered, negative).** This round was also written down before any job ran (`outputs/ablations/v6_prereg.json`). It tested whether stronger pretraining, synthetic fakes, cleaner labels or a five-fold ensemble could beat the shipped bundle. Each lever was first screened as a single ConvNeXt-Small-recipe model over two seeds on the same validation split. To pass, the two-seed mean had to beat the four-seed ConvNeXt-Small reference by +0.003 on both clean and worst-of-23 AUC (at least 0.9095 and 0.9032), with no seed below 0.9048 and 0.8983. The models are ImageNet-22k or 21k initialised backbones from timm (allowed: general image pretraining, not deepfake detectors), Self-Blended Images pseudo-fakes made from the training reals (p = 0.3), and confident-learning label cleaning. Cleaning used out-of-fold predictions on the 5,512 training candidates (out-of-fold AUC 0.9077) and removed 463 flagged images (238 real, 225 fake) from training only; validation labels were not touched.

| lever (two-seed mean) | clean | worst of 23 | rule |
| --- | --- | --- | --- |
| required | >= 0.9095 | >= 0.9032 | - |
| label cleaning | 0.9060 | 0.9011 | fail |
| Swin-S, ImageNet-22k | 0.9041 | 0.8967 | fail |
| ConvNeXt-S, ImageNet-22k | 0.9023 | 0.8925 | fail |
| Self-Blended Images | 0.9001 | 0.8933 | fail |
| EfficientNetV2-S, ImageNet-21k | 0.8996 | 0.8876 | fail |

No lever passed, so the final design defaulted to the baseline recipe: ConvNeXt-Small plus Swin-T, equal mean of logits, no flip. Five-fold training was then simulated on fold 0 with four inner-fold models per member, trained on folds 1 to 4 only, so that fold 0 stayed held out. The switching rule needed a point gain above zero on clean and worst AUC, a 95% lower bound above -0.002 for both, and a T4 estimate under 200 s. The registered fallback, scored once after the design failed, adds the eight inner-fold models to the shipped three. Sources: `ens_v6_pool_pool.json`, `ens_v6_fb_pool.json`.

| candidate | clean | mean | worst of 23 | clean gain [95% CI] | worst gain [95% CI] | rule |
| --- | --- | --- | --- | --- | --- | --- |
| shipped | 0.9065 | 0.9055 | 0.9016 | - | - | - |
| 4 ConvNeXt-S + 4 Swin-T inner-fold models | 0.9036 | 0.9025 | 0.8996 | -0.0029 [-0.0072, 0.0013] | -0.0021 [-0.0070, 0.0023] | fail |
| shipped + the 8 inner-fold models, no flip | 0.9050 | 0.9040 | 0.9016 | -0.0014 [-0.0046, 0.0018] | 0.0000 [-0.0043, 0.0031] | fail |

Both fail on point gain, so by the registered rule the bundle stayed unchanged at that point and full five-fold training was not run. One limit applies: each inner-fold model saw about 60% of the labelled data, against 80% for the shipped members, so this simulation understates a real five-fold set. A true five-fold ensemble may be slightly better than these rows, but we have no evidence that it beats the shipped bundle, and the rule was fixed in advance to stop us from running it on hope. Self-Blended Images were meant to help on unseen generators, which this validation split cannot measure, so that result says only that they cost in-distribution AUC here.

**Rounds v7 to v12 (pixelation, refit, cross-check).** All rules were written to `outputs/ablations/v{7,8,10,11,12}_prereg.json` before the corresponding jobs. Reference numbers are fused three-member sets on fold 0, flip TTA, scored with the default, extended and stress suites of `robust_eval.py`.

- **v7 (pass).** The extended suite showed one weak variant: an 8 px mosaic ("pixelate8") scored 0.7229 for the bundle of section 6, and 0.54 to 0.79 for every model tested. A random mosaic augmentation (`Mosaic` in `augment.py`, probability 0.15) lifted pixelate8 to 0.8322 with default-suite scores unchanged (0.9065 / 0.9054 / 0.9007 clean / mean / worst; clean gain 0.0000 [-0.0029, 0.0034]).
- **v8 (pass, winner `p50`).** Mosaic probability 0.3 and 0.5 and extra seeds of 0.15 were scored. Probability 0.5 was the only set to pass: pixelate8 0.8562, default suite 0.9070 / 0.9057 / 0.9002, clean gain +0.0005 [-0.0034, 0.0046], worst-variant gain -0.0005 [-0.0034, 0.0031], mean over the stress suite 0.8543 (v7: 0.8421), T4 estimate 72 s. The clean-AUC gain is not distinguishable from zero; the improvement is on the pixelation variants.
- **v9 (negative).** Heavy blur/downscale/noise augmentation, with and without mosaic: all six candidates failed (best stress mean 0.8549, required gain +0.02 not reached).
- **v10 (negative, no passer).** Tested a second seed of `p50` (six members) and heavy augmentation on top of mosaic 0.5 (`ph`, `ph30`, one and two seeds). Heavy augmentation lowered clean AUC (`ph`: 0.9013, gain -0.0057 [-0.0094, -0.0023]) and the other extended variants by up to 0.009; the six-member `p50x2` kept the numbers (clean 0.9075, mean 0.9059, worst 0.9010, pixelate8 0.8543) but gained only +0.0004 clean and +0.0009 worst with intervals spanning zero and doubled the estimated T4 time (143.5 s against 71.7 s), so it did not pass. The seed spread of `p50` on pixelate8 is 0.8562 against 0.8471.
- **v11 (refit, gates).** With no v10 winner, the refit uses the v8 `p50` recipe. Gate 1 checked on fold 0, for the trio of seed-42 members fused by mean probability without flip TTA (`robust_eval.py`; hence 0.9059 here against 0.9070 for the `p50` trio in section 5, which is scored as a mean of logits with flip TTA), that the fixed stopping epochs cost little against the best-epoch checkpoint of the same recipe: clean 0.9053 against 0.9059 (-0.0006), worst 0.8978 against 0.8988 (-0.0011), pixelate8 -0.0018, no other extended variant down by more than 0.001. Gate 2: all six refit members finished, final training losses within 0.05 of their fold-0 counterparts (largest difference -0.0112), and test predictions have Spearman 0.9846 with the previous bundle (a broken-model check only, never used to choose). Both passed; the T4 timing gate (cold run under 240 s) was then met with 88.6 s (section 7).
- **v12 (cross-check on folds 1 to 3; fold 4 not completed).** Fold 0 was used for every selection, so recipe A (v7, mosaic 0.15) and recipe B (the shipped recipe, mosaic 0.5) were retrained with fixed stopping epochs on the other folds and scored on each fold's held-out part, seed 42 first. Rule: keep B if the mean clean gain is at least -0.003 and the mean pixelate8 gain is above zero. Final status: The run stopped at the registered deadline (Sunday 8 November, 15:30) with seed 42 only: folds 1 to 3 were scored for both recipes, and on fold 4 recipe A was scored and recipe B finished training but was not scored, so the summary rests on three folds and one seed and no further seeds were run. B minus A, mean +/- standard deviation over folds, clean +0.0005 +/- 0.0013, mean-of-23 -0.0001 +/- 0.0009, worst-of-23 -0.0013 +/- 0.0019, pixelate8 +0.0133 +/- 0.0060, stress mean +0.0066 +/- 0.0022. In absolute terms recipe B scores clean 0.9079, mean 0.9062, worst 0.8976, pixelate8 0.8476, stress mean 0.8495; recipe A scores 0.9073, 0.9063, 0.8989, 0.8342 and 0.8429. The clean and mean-of-23 differences are within the spread, so the recipes are tied on those and B is better on the pixelation variants; the worst-of-23 is slightly lower (-0.0013 against a spread of 0.0019). The rule keeps B. These are three-fold, one-seed figures, with a standard deviation over only three values, for three-member sets trained on 80% of the data, so they are the cross-validated estimate for that setting, not for the refit.

**Rounds v13 to v19 (new recipe and backbones, five folds).** In the rounds above, "shipped" means the bundle shipped at that time. v13 screened label-noise losses on fold 0 (generalised cross entropy, soft relabelling of flagged rows). v14 and v15 added paired-consistency training (section 4). v16 to v18 tested ConvNeXt-Small and Swin-Small with ImageNet-22k pretraining and Self-Blended Images, MaxViT-T, ConvNeXt-Base and a self-supervised DINO ViT-S/16. These were run first on folds 0 and 4, and the shortlisted members were then run on all five folds with the recipe of section 4 and fixed epoch-9 weights. Each fold model was trained on the other four folds of the same group-aware split, so every labelled image has exactly one held-out score per architecture. The per-image logits for every degradation view are kept in `outputs/ablations/v14/fold{0..4}/`; the trained fold models were not kept.

**Round v20 (pre-registered, pass).** The rule was written to `outputs/ablations/v20/v20_prereg.json` before any pooled number was computed.

- **Sets.** Three fixed sets: A = ConvNeXt-Small-22k, Swin-Small-22k, MaxViT-T and ConvNeXt-Base; B = A without ConvNeXt-Base; C = A plus the DINO member. Fusion is the equal mean of logits with flip TTA. All pooled figures are flip-TTA scores, as at inference.
- **Selection.** The set with the highest (clean + mean over the 22 degraded views of the default suite) / 2 is selected.
- **Ship rule.** The selected set ships only if two conditions hold:
  - Its pooled clean AUC is at least 0.9052. This reference was fixed in the registration: it is the pooled out-of-fold clean AUC reported for a ten-model five-fold ResNet-50 and ConvNeXt-Small ensemble on the same data.
  - On folds 0 and 4, the folds with cached logits for the v11-recipe trio, it is no more than 0.003 below that trio in clean and in pixelate8 AUC, on each fold.

Scores: A 0.90742, B 0.90736, C 0.90664. A is selected by the rule; A and B differ by 0.00006, which is within noise. A's pooled clean AUC is 0.9080 (table in section 6), and the fold checks are below. Every condition passes, so set A was refit on all labelled rows.

| fold | system | clean | mean of 22 | worst of 22 | pixelate8 | stress mean |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | v11-recipe trio (ResNet-50, ConvNeXt-Small, Swin-T) | 0.9070 | 0.9056 | 0.9002 | 0.8584 | 0.8509 |
| 0 | set A | 0.9074 | 0.9072 | 0.9049 | 0.8677 | 0.8644 |
| 4 | v11-recipe trio | 0.8972 | 0.8960 | 0.8883 | 0.8364 | 0.8423 |
| 4 | set A | 0.9038 | 0.9023 | 0.8969 | 0.8559 | 0.8596 |

Caveats:
- The four architectures were shortlisted after looking at folds 0 and 4, so the pooled figure is mildly optimistic. Folds 1 to 3 played no part in the shortlist.
- On folds 0 and 4 the MaxViT-T logits come from the run `maxvit_t_consistency`, the only MaxViT-T run kept on those folds; on folds 1 to 3 they come from `maxvit_t`.
- Pooling over folds mixes five models whose score scales differ slightly. The mean of the per-fold clean AUCs of set A is 0.9085 (0.9074, 0.9043, 0.9121, 0.9151, 0.9038), close to the pooled 0.9080.

**Refit gates (v20).** The gates were written before any v20 test prediction existed:
- Every training log is finite, with falling loss. Final losses were 0.43 to 0.49.
- The test confidences have a Spearman correlation of at least 0.95 with the v11 bundle. The plan first said 0.97, but on folds 0 and 4 the healthy v11 trio and set A correlate at only 0.967 and 0.963 out of fold, so 0.97 would have rejected a sound model; the gate only has to catch a broken refit.
- A second inference run is byte-identical, and the format check passes.
- The Kaggle T4 cold run is under 240 s.

Results: Spearman 0.9724, label agreement with v11 0.9803, predicted fake rate 52.0% (v11: 52.3%). The rerun was byte-identical and the format check passed. The Kaggle T4 cold run took 202.1 s, under the 240 s gate (section 7). The test predictions were used only for these sanity gates, never to choose anything.

**Limits of this evidence.** The later rounds (v2 to v6) made more than twenty candidate comparisons on the same 1,408 validation images, so the multiple-comparisons problem is real: a candidate that had passed the rule would still have been optimistic, and a true gain of a few thousandths would not be detectable with an interval this wide (about +/-0.002 to 0.004). The validation split is also the only held-out data, which was used for epoch, member and threshold selection.


## 6. Results

**Final system (v20), five-fold out-of-fold.** Every labelled image is scored by the fold models trained without it, pooled over the 7,000 images. All figures use flip TTA and equal mean-of-logits fusion. "Mean of 22" and "worst of 22" cover the 22 degraded views of the default suite (the 23 variants below minus clean). Accuracy is at threshold 0.5. The 95% intervals come from 1,000 bootstrap draws that resample whole near-duplicate groups. Source: `outputs/ablations/v20/pooled_oof.json`, recomputed by `src/verify_results.py`.

| system (pooled out-of-fold, 7,000 images) | clean [95% CI] | accuracy | mean of 22 | worst of 22 | pixelate8 | extended mean | stress mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **set A (final: 4 architectures)** | **0.9080 [0.9010, 0.9155]** | **0.8716** | **0.9068** | **0.9018** | **0.8615** | **0.8965** | **0.8613** |
| set B (A without ConvNeXt-Base) | 0.9079 | 0.8707 | 0.9068 | 0.9016 | 0.8617 | 0.8966 | 0.8609 |
| set C (A plus DINO ViT-S/16) | 0.9073 | 0.8693 | 0.9060 | 0.9006 | 0.8594 | 0.8957 | 0.8595 |
| ConvNeXt-Small, ImageNet-22k | 0.9033 | 0.8609 | 0.9020 | 0.8962 | 0.8553 | 0.8887 | 0.8528 |
| Swin-Small, ImageNet-22k | 0.9035 | 0.8553 | 0.9015 | 0.8929 | 0.8291 | 0.8871 | 0.8437 |
| MaxViT-T | 0.9046 | 0.8697 | 0.9036 | 0.8997 | 0.8550 | 0.8920 | 0.8524 |
| ConvNeXt-Base | 0.8815 | 0.7897 | 0.8796 | 0.8717 | 0.8215 | 0.8634 | 0.8240 |

The worst default variant for set A is down-scaling to 0.25. The weakest case in the stress suite is still coarse pixelation (24 px: 0.7340). ConvNeXt-Base pools poorly on its own because its score scale differs between folds, yet it does not hurt the ensemble, and the registered rule kept it. The ensemble gains about 0.003 to 0.004 AUC over its best member, so the members' errors are again strongly correlated. The deployed bundle has two seeds per architecture, refit on all labelled rows; that has no held-out score, as explained in section 3.

**Earlier system (v11) on fold 0.** Validation set: 1,408 images, never used for training, with near-duplicates of training images excluded by the split. Model selection (best epoch, member choice, threshold) also used this set, so the numbers are mildly optimistic for the shipped weights. The tables from here to the end of this section describe the three-member bundle selected on fold 0. The v11 system (section 3) was later refit on all labelled images, which was a decision to trade measurability for data: fold-0 experiments gave about +0.004 (ConvNeXt-Small) and +0.007 (Swin-T) clean AUC per model going from 60% to 80% of the data, so we expect a small gain, roughly +0.003 to +0.005, but that is an extrapolation and not a measurement. To keep it honest, no refit model is scored on validation images (they are in its training set); the figures to quote are the fold-0 ones and the cross-validated ones in section 5. Fixed stopping epochs replaced best-epoch selection for this reason; the fold-0 cost of that change was 0.0006 clean AUC.

**Robustness of the v11 recipe.** Fold-0 results for the `p50` trio (three members, flip TTA): default suite clean 0.9070, mean 0.9057, worst 0.9002. Extended and stress variants against the bundle of the table below: pixelate8 0.8562 (from 0.7229; 0.8322 with mosaic probability 0.15), mean over the stress suite 0.8543. The stress suite is where the weak spots remain: for the v7 set, pixelate24 0.6898, pixelate12 0.7828, downscale 0.125 0.8180, blur sigma 4 0.8474 and heavy_combo 0.8969; JPEG quality 10 and noise std 40 cost less (0.8986 and 0.8962). Coarser mosaics than 8 px remain the weakest case and were not separately targeted.

**Ensemble versus members** (flip TTA on; AUC clean / mean over 23 variants / worst variant / std over variants):

| model | clean | mean | worst | std |
| --- | --- | --- | --- | --- |
| ResNet-50 | 0.9036 | 0.9023 | 0.8958 | 0.0022 |
| ConvNeXt-Small | 0.9060 | 0.9047 | 0.8992 | 0.0018 |
| Swin-T | 0.9025 | 0.9010 | 0.8958 | 0.0024 |
| **Ensemble (fold-0 bundle)** | **0.9065** | **0.9055** | **0.9016** | **0.0015** |
| previous triple (ResNet-50 + ConvNeXt-Tiny + Swin-T) | 0.9052 | 0.9044 | 0.9011 | - |

The ensemble beats the best single model by +0.0008 mean and +0.0024 worst-case AUC. The gain is small, and a single ConvNeXt-Small with TTA is nearly as good. The members' errors are strongly correlated, so fusion cannot remove errors they all make. The most plausible reason is that part of the remaining error is label noise and hard cases (section 10), but we have not shown that directly. Flip TTA changed mean AUC by at most +0.0015 for each fold-0 bundle member and lowered neither mean nor worst case. Larger candidate ensembles tried later did not beat it with confidence (section 5, "Later improvement rounds and the switching rule").

**Threshold, accuracy and F1.** The decision threshold is fixed at 0.5. At the clean validation set the ensemble gives accuracy 0.880, F1 0.883, precision 0.870, recall 0.896 (confusion matrix: 601 true negatives, 95 false positives, 74 false negatives, 638 true positives). We compared 0.5 with the Youden-J and best-F1 thresholds (both 0.5195): mean F1 over the 23 variants is 0.8788 for 0.5 against 0.8792 for 0.5195, and 0.5 has the lower spread of F1 across variants (std 0.0065 against 0.0068). With no real difference in mean F1, the unmodified 0.5 is the safer choice under unknown degradations. The score that counts is AUC, which does not depend on the threshold.

**Robustness suite.** 23 deterministic variants of the validation images: clean; JPEG quality 90, 70, 50, 30; Gaussian blur sigma 1, 2; noise std 5, 10, 20; down-scale to 0.5 and 0.25 then back; brightness plus and minus 30; contrast 0.6 and 1.4; grayscale; centre crop 90 percent; rotation 5 degrees; horizontal flip; resize round-trips through 160 and 299; and one combination (JPEG 50, blur 1, noise 5). AUC on every variant (members without TTA; ensemble with flip TTA, from `outputs/ablations/ens_v5_shipped_per_variant.csv`):

| variant | ResNet-50 | ConvNeXt-Small | Swin-T | Ensemble (TTA) |
| --- | --- | --- | --- | --- |
| clean | 0.9027 | 0.9062 | 0.9018 | 0.9065 |
| JPEG q90 / q70 | 0.9027 / 0.9017 | 0.9058 / 0.9054 | 0.9020 / 0.9009 | 0.9063 / 0.9059 |
| JPEG q50 / q30 | 0.9039 / 0.9030 | 0.9048 / 0.9032 | 0.8985 / 0.8967 | 0.9061 / 0.9048 |
| blur sigma 1 / 2 | 0.9022 / 0.8950 | 0.9060 / 0.9020 | 0.9017 / 0.8982 | 0.9066 / 0.9036 |
| noise std 5 / 10 / 20 | 0.9021 / 0.9003 / 0.8965 | 0.9060 / 0.9048 / 0.9015 | 0.9009 / 0.8993 / 0.8934 | 0.9057 / 0.9055 / 0.9038 |
| down-scale 0.5 / 0.25 | 0.9005 / 0.8925 | 0.9052 / 0.8983 | 0.9011 / 0.8934 | 0.9059 / 0.9016 |
| brightness +30 / -30 | 0.8978 / 0.9020 | 0.9030 / 0.9061 | 0.8969 / 0.9010 | 0.9033 / 0.9062 |
| contrast 0.6 / 1.4 | 0.9004 / 0.9018 | 0.9042 / 0.9064 | 0.9009 / 0.9000 | 0.9047 / 0.9065 |
| grayscale | 0.9017 | 0.9027 | 0.8954 | 0.9037 |
| centre crop 90 percent | 0.9010 | 0.9079 | 0.9035 | 0.9087 |
| rotation 5 degrees | 0.9028 | 0.9043 | 0.9011 | 0.9066 |
| horizontal flip | 0.9025 | 0.9039 | 0.9018 | 0.9065 |
| resize 160 / 299 | 0.9020 / 0.9023 | 0.9058 / 0.9060 | 0.9014 / 0.9013 | 0.9063 / 0.9063 |
| combo JPEG 50, blur 1, noise 5 | 0.9019 | 0.9039 | 0.8996 | 0.9058 |
| mean over 23 | 0.9008 | 0.9045 | 0.8996 | 0.9055 |
| worst | 0.8925 | 0.8983 | 0.8934 | 0.9016 |

![Per-variant validation AUC of the three members](report_robust_members.png)

The worst cases are the strongest down-scaling (0.25), the strongest noise (std 20) and blur sigma 2, each costing about 0.01 AUC. Nothing in the suite breaks a member. The fused ensemble is the flattest column: its worst variant (0.9016, down-scale 0.25) is 0.0049 below clean, against 0.0079 to 0.0102 for the members.

**Caveat on the robustness suite.** The training augmentation covers JPEG, blur, noise and down-scaling, which are also in the suite, so those rows are partly in-distribution for the model. Brightness, contrast, grayscale, crop and rotation are covered more loosely by colour jitter and the random crop and rotation. The organisers' degradations are not disclosed; the suite is our best guess, not a measurement of the hidden test variants.

## 7. Efficiency

**Final eight-member bundle (v20).** The bundle runs 8 members x 2 views = 16 forward passes per image.
- **Training host (RTX 4060 Ti).** The shipped script ran on all 3,000 test images in 101.1 s and 101.0 s (batch 128), with peak GPU memory 3,756 MiB. The two outputs were byte-identical and passed the format check.
- **Tesla T4.** Timed on a Kaggle Tesla T4 (torch 2.11.0+cu128, timm 1.0.29) with the shipped inference script, 3,000 images of 224x224, batch 128, flip TTA: the cold run took 202.1 s wall (194.8 s inside the script) and a second run 205.0 s wall (201.0 s), peak GPU memory 3,756 MiB, return code 0 both times. The registered gate (cold run under 240 s) and the 5-minute budget are met. As for the six-member timing below, the bundle held randomly initialised fp16 weights of the same eight architectures in the same file format, and the images were random-noise JPEGs, because run time does not depend on the weight values; the predictions of that run are not used for anything.
- **fp16 storage.** The weights are stored in fp16, which makes the bundle 867 MB instead of 1.73 GB in fp32. The script loads them into fp32 models and runs under fp16 autocast. Against the fp32 copy of the bundle, the test confidences differ by at most 4.5e-4 with no label flip.

**Previous six-member bundle (v11).** It ran twice as many members as the three-member bundle (six forward passes per view, 12 per image). The six-member bundle was timed on a Kaggle Tesla T4 (torch 2.11.0+cu128) with the shipped inference script, 3,000 images of 224x224, batch 128, flip TTA: the cold run took 88.6 s wall (81.2 s inside the script) and a second run 86.9 s wall (82.9 s), peak GPU memory 2,229 MiB, return code 0 both times. The registered gate (cold run under 240 s) and the 5-minute budget are met. For this timing the bundle held randomly initialised weights of the same six architectures in the same file format (805 MB), and the images were random-noise JPEGs, because run time does not depend on the weight values; the predictions of that run are not used for anything. The one host run took 305 s with a peak of 2,227 MiB GPU memory, but the host GPU was shared with other users, so that figure says nothing about a T4. On the training host the v11 six-member script ran on all 3,000 test images, passed the format check, and a second run was identical to the first (maximum confidence difference 0).

The measurements below are for the earlier three-member bundle. Measured on a real Tesla T4 (Kaggle; Python 3.13, torch 2.11.0+cu128, torchvision 0.26.0, numpy 2.1.3, pillow 13.0.0). Script: `inference_script.py`, 3,000 test images, batch 128, flip TTA, fp16 autocast, all three members.

| run | wall time | time reported inside the script |
| --- | --- | --- |
| 1 (cold) | 93.1 s | 86.0 s |
| 2 (warm) | 55.7 s | 51.9 s |

Wall time includes interpreter start-up, bundle loading, model construction, image loading and CSV writing. Peak GPU memory (torch allocator) is 1,841 MiB. Both runs are far inside the 5-minute budget. We did not isolate why the first run is slower; likely contributors are a cold file cache and one-time CUDA and library initialisation, but that is not measured.

The T4 output matches that bundle's `predictions.csv` (produced on an RTX 4060 Ti) with a maximum absolute confidence difference of 1.9e-4 and one label flip out of 3,000 (a sample sitting at the threshold). That is the usual fp16 kernel and batch-size effect; on the 4060 Ti we measured up to 1.7e-4 between batch sizes 32, 64 and 128 with one flip. On the 4060 Ti the whole run takes 26.8 s (batch 128), 24.9 s (batch 64), 23.4 s (batch 32), with two runs at the same batch size bit-identical.

Size: three-member bundle 402.6 MB (100.5M parameters in fp32), peak 1.8 GB GPU memory; the six-member bundle is 805 MB with a peak of 2,229 MiB on a T4. The Kaggle environment used a newer Python, numpy and pillow than the pinned requirements (Python 3.10, numpy 2.2.6, pillow 12.3.0), and the script still ran unchanged.

## 8. Explainability: Grad-CAM

**Scope.** The maps in this section are of the shipped v20 bundle itself: its four architectures (ConvNeXt-Small and Swin-Small with ImageNet-22k pretraining, MaxViT-T, ConvNeXt-Base). Because the bundle members were refit on every labelled row, the validation images shown here were part of their training set; the maps show where the deployed model attends, not held-out behaviour. The v20 fold-0 models that were held out were not kept after scoring, so the shipped refit weights are the only v20 weights available to map; the attention pattern matches what the earlier held-out models showed.

Implementation: our own code (`src/gradcam.py`), a forward hook and a gradient hook on the last stage of each member (last stage of the features for ConvNeXt-Base, the last timm stage for ConvNeXt-Small, the last block of MaxViT-T, and the output of the final norm for Swin-Small, a channels-last 7x7xC tensor permuted to channels-first). The map is the ReLU of the gradient-weighted sum of target activations for the single fake logit, upsampled to 224x224 and scaled to 0..1 per image. Maps are 7x7 before upsampling, so they are coarse. The per-architecture figures show one representative member of each backbone; the ensemble figures average the normalised maps of all eight bundle members (four architectures, seeds 42 and 7). Figures use 16 random validation images (seed 42) for each of true positive, true negative, false positive and false negative at threshold 0.5, with one figure per member, an ensemble-average figure, and a degradation figure. Region numbers below are enrichments: share of map mass in a region divided by the region's share of area (1.0 is uniform).

![Ensemble Grad-CAM, true positives (correctly detected fakes)](outputs/figures/gradcam_ensemble_tp.png|0.52)

Findings:

- **Fakes found.** All four members attend to the inner face, mostly the eye and upper-nose band (eye-band enrichment 2.4 for ConvNeXt-Small, 1.9 for Swin-Small, 1.3 for MaxViT-T, 3.7 for ConvNeXt-Base, 2.3 for the ensemble) and the forehead and cheeks. The outer border ring is nearly ignored (0.14 to 0.72, ensemble 0.42). Nothing in the maps points to a seam along the face contour or to the background.
- **Reals.** The members give weak, diffuse maps with no dominant face blob (ensemble eye band 0.42, border ring 1.24; MaxViT-T leans on the border, 1.6). They show "no fake evidence found", not a positive reason for "real".
- **Errors.** False positives highlight the same inner-face region as true positives (eye-band enrichment 1.4 to 2.3, ensemble 1.8), so those images carry artifacts the models treat as fake, or carry noisy labels. False negatives look like true negatives (ensemble eye band 0.57, border ring 1.08): weak attention on hairline, chin or background, often with profile, tilted, blurred or partly covered faces.
- **Under degradation.** On six images under JPEG 30, blur 2 and noise 10 the highlighted region stays the same. Correlation with the clean map for the ensemble is 0.99 in every condition; MaxViT-T's maps move most (0.92 to 0.94) and the ConvNeXt and Swin maps least (0.98 to 0.99). The ensemble's eye-band enrichment stays near 1.5 across all four conditions.

Caveat. Grad-CAM shows where the gradient is large, not what the model needs, and 7x7 maps cannot separate eyes from skin texture. The figures support "the decision uses the face interior and not the border", not a claim about a specific artifact.

![Ensemble Grad-CAM, false positives (real faces flagged as fake)](outputs/figures/gradcam_ensemble_fp.png|0.52)

## 9. TensorFlow export

**Scope.** The TensorFlow export is of the v11 six-member system, not of the v20 bundle. `inference_script_tf.py` with `tf_model/` reproduces the v11 predictions, not the v20 `predictions.csv`, and it is shipped for the TensorFlow bonus only. We did not attempt a conversion of the v20 members in the time left. Their two transformer families (Swin-Small and MaxViT-T) are the kind that onnx2tf failed on for Swin-T, as described below.

The PyTorch weights were converted to TensorFlow SavedModels (PyTorch to ONNX, opset 17, then onnx2tf 1.28.8 with TensorFlow 2.17.1) and an inference script `inference_script_tf.py` runs the same ensemble: the same preprocessing, flip TTA, mean of logits and threshold, read from `meta.json` in the model folder.

- **Final six-member export (v11).** All six members converted (`failed_members` empty; the Swin-T rebuild above is used for both Swin-T seeds, and members of a repeated architecture are stored in separate folders by member name). Parity on 500 validation images against PyTorch fp32 on CPU, same preprocessing and mean-logit fusion with flip TTA: maximum confidence difference 3.0e-07 (mean 4.2e-08), per-member logit differences at most 5.5e-06, zero label flips, equal AUC to three decimals. These images are in the refit training set, so the AUC on them (0.944) says nothing about generalisation; only the agreement between the two frameworks is meaningful. TensorFlow inference on all 3,000 test images matches the v11 PyTorch predictions with a maximum confidence difference of 6.8e-04 and zero label flips. TensorFlow timing of this bundle on a T4 or CPU has not been recorded.
- **Parity of the earlier three-member export.** On 500 validation images (seeded shuffle, CPU), TensorFlow against PyTorch fp32: maximum confidence difference 4.2e-07, per-member logit differences at most 3.9e-06, AUC 0.923476 for both on this subset, zero label flips (`outputs/eval/tf_parity.json`). This is a fresh export of all three members from an empty work directory with `src/export_tf.py`, followed by `--parity`; it was superseded by the six-member export above.
- **Test run of the earlier three-member export.** That `tf_model/` was produced by `src/export_tf.py` from the three-member `model_weights.pt`, not hand-converted. `inference_script_tf.py` on all 3,000 test images (24-core CPU host without GPU, TensorFlow 2.17.1, batch 64, flip TTA) took 5 min 49 s wall (peak memory 5.2 GB; `outputs/eval/tf_test_run.json`), passes `src/check_predictions.py` (3,000 rows, 51.6% predicted fake), and differs from `predictions.csv` by at most 1.9e-03 in confidence (mean 5.5e-05) with one label flip out of 3,000 (an image whose confidence lies within that difference of 0.5; `predictions.csv` comes from the GPU fp16 path). The six-member export was checked against the v11 predictions as described above (6.8e-04, zero flips) but not timed. We did not time the TF path on a T4, so the efficiency claim in section 7 applies to PyTorch only.
- **Swin-T conversion.** ResNet-50 and ConvNeXt-Small convert through ONNX and onnx2tf (dynamic batch). onnx2tf cannot convert Swin-T's window attention: it fails in the attention blocks in every variant we tried (dynamic and static batch, several flags, a rewrite with constant masks and an auto-generated replacement file). `src/export_tf.py` therefore rebuilds Swin-T directly in TensorFlow ops from the PyTorch weights. The relative position bias and the shifted-window masks are computed once and stored as constants (a rewritten PyTorch module that matches torchvision's logits exactly on random input checks this step), and the SavedModel has a fixed batch of 16, so the TF script pads the last batch. `python src/export_tf.py --only swin_t` builds it in seconds; the full six-member export and the parity runs above were done on a CPU host with TensorFlow 2.17.1, onnx 1.16.1 and onnx2tf 1.28.8.
- **Packaging.** The six SavedModels total about 773 MB, measured on the host with `du -sh` on the `tf_model/` folder (the earlier three-member export was 387 MB) and are shipped in the `tf_model/` folder of the submission, not in the source archive.

## 10. Challenges and limitations

- **Unseen generators and distribution shift.** Validation images come from the same pool and generator mix as the training data. AUC 0.90 on that split says little about how a different generator or a harsher degradation will score. We did not test any out-of-distribution data because only the provided dataset may be used.
- **Label noise and near-duplicates.** 53 near-duplicate groups carry both labels (108 images). Dropping them helped a little (section 5). The RGB ImageNet backbones tried, from 4M to 88M parameters (`outputs/models_params.json`), plateau between 0.90 and 0.91 validation AUC, and fusion adds under 0.003. The SRM-residual and DCT-domain members scored lower (0.644 to 0.86 clean). That pattern is what label noise or an irreducible hard subset would produce, but we have not proved it and cannot rule out a modelling limit shared by all ImageNet backbones.
- **Seed noise.** About 0.003 mean AUC between two seeds of the same configuration, so most single-run comparisons in section 5 and the choice of ConvNeXt-Small over ConvNeXt-Tiny are not statistically supported.
- **Selection on validation.** Epoch, member and threshold choices used the validation set; no separate untouched test split exists. The later rounds compared many candidate sets on the same images (multiple comparisons; section 5, "Later improvement rounds and the switching rule"), which is why a fixed switching rule with bootstrap intervals was used. There is no public leaderboard to calibrate against.
- **Robustness suite is our own.** See the caveat in section 6. Strong degradations outside it (heavy compression, very low resolution, unusual colour processing) are untested.
- **Grad-CAM resolution.** 7x7 maps; the maps for confident reals are weak and diffuse by construction. The bundle members were refit on all labelled rows, so the images mapped in section 8 were in their training set.
- **Hardware.** Training used remote RTX 4060 Ti hosts, not a T4. The three-member and six-member bundles were timed on a real T4 (Kaggle) as described in section 7, and so was the final eight-member bundle. The per-model img/s figures are from the 4060 Ti.
- **No held-out score for the final weights.** The refit uses all labelled images, so the final AUC cannot be measured on our data. The quoted figures come from five-fold models, each trained on 80% of it and with one seed per architecture. The expected gain from the extra data and the second seed is not measured.
- **Shortlist seen on two folds.** The v20 architectures were chosen after looking at folds 0 and 4, so the pooled out-of-fold AUC is mildly optimistic (section 5).
- **TensorFlow export lags the final bundle.** The TensorFlow export covers the v11 members, not the v20 bundle (section 9); the Grad-CAM figures were regenerated on the v20 bundle (section 8).
- **Seed and fold noise.** The v20 cross-validation is one seed per architecture on five folds; the per-fold clean AUC of set A ranges from 0.9038 to 0.9151. The earlier cross-validation of the v11 recipe (v12) is one seed on three folds (it was stopped by the deadline before fold 4 was scored for recipe B and before any further seed); clean-AUC differences of a few thousandths between recipes are within its spread.
- **TF export** limits in section 9.
- **Compute and storage.** Our workstation had almost no free disk, which is why all training and large outputs lived on a remote host and on a separate drive.

## 11. Reproducibility

Everything below runs from the repository root; details and every flag are in the source README.

- **Seeds.** 42 for the split and for the fold runs; the final eight members use seeds 42 and 7.
- **Hardware.** Training and ablations: RTX 4060 Ti (16 GB) hosts. Each v20 refit member took 12 to 19 minutes for 12 epochs on one GPU (`train_seconds` in its `config.json`). v11: batch 64, about 9 + 20 + 13 minutes for the three fold-0 members; the refit members trained on 6,892 rows take about 40 to 80 minutes each when the GPU is shared and throttled. Timing: Tesla T4 (Kaggle).
- **Software.** Python 3.10, torch 2.11.0, torchvision 0.26.0, timm 1.0.31, numpy 2.2.6, pillow 12.3.0 for training and inference on the GPU host (CUDA 12.8 and 13.0 builds gave identical predictions). The seed-7 v20 members were trained on a second host with torch 2.7.1 and the same code. The local requirements file pins CPU wheels used only for smoke tests. TF export and inference: TensorFlow 2.17.1, onnx 1.16.1, onnx2tf 1.28.8 (plus ai-edge-litert for the export).
- **Order of scripts.** `inspect_data`, `find_duplicates`, `make_split`, `train` (once per member), `robust_eval`, `ensemble` (chooses members, fusion and threshold), `bundle`, then `inference_script.py`. Grad-CAM: `gradcam.py`; TF: `export_tf.py`. Candidate-set comparisons of the later rounds: `ensemble_pool.py` (paired bootstrap, per-variant AUC) and `stack.py`; their commands and the v5 run list are in the source README and `outputs/ablations/v5_prereg.json`.
- **Final refit members (v20).** For each seed in 42 and 7, and each of `--arch swin_small_22k --bs 48 --sbi_p 0.15`, `--arch convnext_small_22k --bs 48 --sbi_p 0.15`, `--arch maxvit_t --bs 48` and `--arch convnext_base --bs 32`:

```
python src/train.py <arch flags> --epochs 12 --save_at 9 --lr 3e-4 --ema_decay 0.99 --backbone_lr_mult 1.0 --label_smoothing 0.05 --aug_level strong --conflict drop --mosaic_p 0.5 --mosaic_max 16 --paired_target_mix 0.5 --all_data --seed <seed> --split_csv data/splits/split_v1.csv --out_dir runs/<name>_s<seed>
python src/bundle.py --chosen outputs/ablations/ensemble_chosen_v20.json --out model_weights.pt --fp16
```

The checkpoint used is `fixed.pt`. The bundle has eight members, mean-logit fusion, flip TTA and threshold 0.5. The pooled out-of-fold table of section 6 is recomputed from the saved logits by `python src/verify_results.py`, and `python -m unittest discover -s tests` checks the split, the bundle and the shipped predictions.
- **Previous refit members (v11).** Per architecture and seed, the fold-0 commands below plus `--mosaic_p 0.5 --mosaic_max 16 --all_data --save_at <12 or 9> --seed <42 or 7>`; the checkpoint used is `fixed.pt`. Bundle: six members, mean-logit fusion, flip TTA, threshold 0.5.
- **Fold-0 member commands (v11)** (flags that differ from the defaults):

```
python src/train.py --arch resnet50 --epochs 12 --lr 1e-3 --backbone_lr_mult 1.0 --ema_decay 0.99 --conflict drop --out_dir outputs/runs/final_resnet50
python src/train.py --arch convnext_small --epochs 12 --lr 3e-4 --backbone_lr_mult 1.0 --ema_decay 0.99 --conflict drop --out_dir outputs/runs/e_cnxt_small
python src/train.py --arch swin_t --epochs 12 --lr 3e-4 --backbone_lr_mult 1.0 --ema_decay 0.99 --conflict drop --out_dir outputs/runs/final_swin_t
```

- **Determinism.** Two inference runs at the same batch size on the same machine are bit-identical; across batch sizes and GPUs the confidences differ by up to about 2e-4 (fp16), with at most one label flip in 3,000.
- **Logs.** Per-epoch logs (`log.csv`) and configs of every completed training run, including the v2 to v6 rounds, are in the `logs/runs/` folder of the submission, with `ablations.csv` and the TF parity file. The aborted v4 attempts are described in section 5 only. The v4 candidate-set results exist only as the summary `outputs/ablations/ens_v4_pool.json`; the four trained v4 models have their own logs and per-variant evaluation files.

## 12. Compliance statement

- Training used only the provided training set. The external weights are general ImageNet-pretrained backbones, none trained for deepfake detection:
  - torchvision: MaxViT-T and ConvNeXt-Base in the final system; ResNet-50, ConvNeXt-Small and Swin-T in v11; EfficientNet and ConvNeXt-Tiny in ablations.
  - timm: ConvNeXt-Small and Swin-Small pretrained on ImageNet-22k and fine-tuned on ImageNet-1k, in the final system; EfficientNetV2-S-21k in an ablation.
  - A self-supervised ImageNet DINO ViT-S/16, in an ablation only.

  We used no deepfake-specific detector and no extra face data. The Self-Blended Images pseudo-fakes are made on the fly from the provided training reals only.
- The test images were used for inference only: no training, no pseudo-labels, no normalisation statistics or thresholds computed from them. The test folder was counted and scored; validation (a held-out part of the training data) was used for all model selection.
- The inference script reproduces `predictions.csv` from the bundled weights, contains no hardcoded paths, and keeps the organisers' command-line arguments.
- The CSV header is `image_id,label,confidence`, with `image_id` equal to the file name without extension, `label` the thresholded class and `confidence` the ensemble P(fake).
