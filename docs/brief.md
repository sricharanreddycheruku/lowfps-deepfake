# Deepfake face detection: brief report

Team lowfps - Deepfake ML Hackathon, IIIT Bangalore (Synergy '26). Full details: `technical_report.pdf`.

## Architecture

![Ensemble of six members (three ImageNet-pretrained backbones at seeds 42 and 7) with flip TTA and logit averaging](report_architecture.png|0.9)

Six members: three torchvision backbones fine-tuned end to end from ImageNet weights, each trained at two seeds (42 and 7), each with a Dropout(0.3) plus single-logit head: ResNet-50 (23.5M parameters), ConvNeXt-Small (49.5M) and Swin-T (27.5M), all at 224 px. The confidence is the sigmoid of the mean logit over the 6 members and 2 views (original and horizontal flip). The decision threshold is 0.5. The members come from different families (residual CNN, modern CNN, shifted-window transformer); their errors are still strongly correlated, so the ensemble gain is small (below). No deepfake-specific detector and no external data were used.

## Training and augmentation

BCE with logits and label smoothing 0.05; AdamW, weight decay 0.05, batch 64, learning rate 1e-3 (ResNet-50) or 3e-4 (ConvNeXt-Small, Swin-T) for head and backbone alike; one epoch of warm-up then cosine decay over 12 epochs; EMA of weights (decay 0.99); fp16 autocast. For model selection the checkpoint was chosen on the mean of clean validation AUC and a robust AUC (JPEG q50, blur, noise). The shipped members are refit on all labelled images (conflicting-label images dropped) and stop at fixed epochs (12, 9 for ConvNeXt-Small, 12) instead; this costs 0.0006 clean AUC against best-epoch selection on fold 0.

Augmentation: random resized crop (scale 0.8 to 1), horizontal flip, rotation up to 10 degrees, then in random order JPEG re-encode quality 30 to 95 (p 0.5), Gaussian blur sigma up to 2 (p 0.3), Gaussian noise std 2 to 20 (p 0.3), down/up-scale 0.25 to 0.75 (p 0.3) and sharpen (p 0.2); colour jitter, grayscale (p 0.1) and random erasing (p 0.25). A random mosaic (pixelation) augmentation with probability 0.5 was added in later rounds; it raised the 8 px pixelation variant from 0.7229 to 0.8562 AUC.

## Validation strategy

Near-duplicate images (perceptual hash, Hamming distance 6 or less; 279 groups) are kept together in a stratified group split: 5,592 training and 1,408 validation images with no group overlap. 53 groups carry both labels (108 images); these are dropped from training in the final recipe. Robustness is measured on 23 deterministic degradation variants of the validation images (JPEG 90 to 30, blur, noise, down-scaling, brightness, contrast, grayscale, crop, rotation, flip, resize round-trips and one combination). Test images were used for inference only.

## Headline numbers (validation fold 0, flip TTA)

The shipped refit has seen every labelled image, so it has no held-out score. The figures are for models trained on 80% of the data and scored on the other 20%.

| model | clean AUC | mean AUC over 23 variants | worst variant |
| --- | --- | --- | --- |
| ResNet-50 (earlier recipe) | 0.9036 | 0.9023 | 0.8958 |
| ConvNeXt-Small (earlier recipe) | 0.9060 | 0.9047 | 0.8992 |
| Swin-T (earlier recipe) | 0.9025 | 0.9010 | 0.8958 |
| Earlier three-member ensemble | 0.9065 | 0.9055 | 0.9016 |
| **Six-member set, shipped recipe (best-epoch checkpoints)** | **0.9075** | **0.9059** | **0.9010** |

The six-member row uses best-epoch checkpoints; fixed stopping cost 0.0006 clean on fold 0.

At threshold 0.5 the earlier three-member ensemble has accuracy 0.880 and F1 0.883 on clean validation. Its gain over the best single model is +0.0008 mean AUC, which is below the seed-to-seed noise we measured (about 0.003); the clearer benefit is a better worst case (+0.0024) and a smaller spread across variants. The shipped recipe differs from the earlier one by the mosaic augmentation and the refit. On folds 1 to 3 (one seed), compared against the v7 mosaic-0.15 recipe, both with fixed stopping on 80% of the data (the refit itself was not tested), it is tied on clean AUC (+0.0005, sd 0.0013) and better on pixelation (+0.0133, sd 0.0060), with a slightly lower worst case (-0.0013, sd 0.0019). Validation was also used to select epochs, members and threshold, so these figures are mildly optimistic, and the hidden test degradations may differ from our suite. Fusion is a per-image mean of logits, independent of other test images. Larger or rank-fused candidate sets tried in later rounds gained at most +0.0018 clean and +0.0035 worst-case AUC with bootstrap intervals that include zero. A pre-registered round (v5: noise-residual and DCT members, SAM, ELR, face cutout, GridMask, centre-crop TTA, stacking, a patch-level member) gained at most +0.0015 clean and +0.0025 worst case, also with intervals that include zero, so none passed the switching rule and the bundle was not changed (technical report, section 5, "Later improvement rounds and the switching rule"). Reusing the same validation images for many comparisons is a limitation.

Two pre-registered rounds tried to beat the earlier bundle and failed the fixed rule. The second (v6) tested ImageNet-22k backbones, Self-Blended Images, confident-learning label cleaning and a simulated five-fold ensemble (best clean gain -0.0014).

Most useful ablation findings (EfficientNet-B0, 20 epochs): a backbone learning rate equal to the head's raised the worst-variant AUC from 0.790 to 0.884; dropping conflicting-label images and a learning rate of 1e-3 gave mean 0.9014 against 0.8711 for the first recipe; CutMix, MixUp and 288 px input did not help.

## Efficiency (Tesla T4, Kaggle, PyTorch 2.11.0, fp16, batch 128, 3,000 images, flip TTA)

The six-member bundle was timed on a Kaggle T4: 88.6 s cold wall (81.2 s in the script) for 3,000 images, batch 128, flip TTA, peak GPU memory 2,229 MiB, against the 5-minute budget (random weights of the same architectures and noise JPEGs were used, as run time does not depend on weight values). A run on a shared host GPU took 305 s, which is not a T4 figure. The table below is for the earlier three-member bundle.

| run | wall time | in script |
| --- | --- | --- |
| cold | 93.1 s | 86.0 s |
| warm | 55.7 s | 51.9 s |

Peak GPU memory 1,841 MiB; weights bundle 402.6 MB (the six-member bundle is 805 MB). The output matches that bundle's `predictions.csv` to a maximum confidence difference of 1.9e-4 with one label flip in 3,000 (fp16 kernel differences at the threshold).

## Bonus items and limits

Grad-CAM (own implementation, run on the fold-0 members because the refit models saw every validation image) shows all members attend to the inner face, mainly the eye and upper-nose region, and not the border; maps stay stable under JPEG, blur and noise. A TensorFlow export of the six-member bundle matches PyTorch to 3.0e-07 in confidence on 500 validation images and is reproducible from `src/export_tf.py` (Swin-T is rebuilt directly in TensorFlow ops). Main limits: validation comes from the training distribution, so generalisation to unseen generators is untested; labels contain conflicting near-duplicates; most single-run comparisons are within seed noise.
