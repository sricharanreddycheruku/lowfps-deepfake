# Deepfake face detection: brief report

Team lowfps - Deepfake ML Hackathon, IIIT Bangalore (Synergy '26). Full details: `technical_report.pdf`.

## Architecture

![Ensemble of three ImageNet-pretrained backbones with flip TTA and logit averaging](report_architecture.png|0.9)

Three torchvision backbones fine-tuned end to end from ImageNet weights, each with a Dropout(0.3) plus single-logit head: ResNet-50 (23.5M parameters), ConvNeXt-Small (49.5M) and Swin-T (27.5M), all at 224 px. The confidence is the sigmoid of the mean logit over the 3 members and 2 views (original and horizontal flip). The decision threshold is 0.5. The members come from different families (residual CNN, modern CNN, shifted-window transformer); their errors are still strongly correlated, so the ensemble gain is small (below). No deepfake-specific detector and no external data were used.

## Training and augmentation

BCE with logits and label smoothing 0.05; AdamW, weight decay 0.05, batch 64, learning rate 1e-3 (ResNet-50) or 3e-4 (ConvNeXt-Small, Swin-T) for head and backbone alike; one epoch of warm-up then cosine decay over 12 epochs; EMA of weights (decay 0.99); fp16 autocast; seed 42. The checkpoint is chosen on the mean of clean validation AUC and a robust AUC (JPEG q50, blur, noise).

Augmentation: random resized crop (scale 0.8 to 1), horizontal flip, rotation up to 10 degrees, then in random order JPEG re-encode quality 30 to 95 (p 0.5), Gaussian blur sigma up to 2 (p 0.3), Gaussian noise std 2 to 20 (p 0.3), down/up-scale 0.25 to 0.75 (p 0.3) and sharpen (p 0.2); colour jitter, grayscale (p 0.1) and random erasing (p 0.25).

## Validation strategy

Near-duplicate images (perceptual hash, Hamming distance 6 or less; 279 groups) are kept together in a stratified group split: 5,592 training and 1,408 validation images with no group overlap. 53 groups carry both labels (108 images); these are dropped from training in the final recipe. Robustness is measured on 23 deterministic degradation variants of the validation images (JPEG 90 to 30, blur, noise, down-scaling, brightness, contrast, grayscale, crop, rotation, flip, resize round-trips and one combination). Test images were used for inference only.

## Headline numbers (validation, flip TTA)

| model | clean AUC | mean AUC over 23 variants | worst variant |
| --- | --- | --- | --- |
| ResNet-50 | 0.9036 | 0.9023 | 0.8958 |
| ConvNeXt-Small | 0.9060 | 0.9047 | 0.8992 |
| Swin-T | 0.9025 | 0.9010 | 0.8958 |
| **Ensemble (shipped)** | **0.9065** | **0.9055** | **0.9016** |

At threshold 0.5 the ensemble has accuracy 0.880 and F1 0.883 on clean validation. The gain over the best single model is +0.0008 mean AUC, which is below the seed-to-seed noise we measured (about 0.003); the clearer benefit is a better worst case (+0.0024) and a smaller spread across variants. Validation was also used to select epochs, members and threshold, so these figures are mildly optimistic, and the hidden test degradations may differ from our suite.

Most useful ablation findings (EfficientNet-B0, 20 epochs): a backbone learning rate equal to the head's raised the worst-variant AUC from 0.790 to 0.884; dropping conflicting-label images and a learning rate of 1e-3 gave mean 0.9014 against 0.8895 for the first setting; CutMix, MixUp and 288 px input did not help.

## Efficiency (Tesla T4, Kaggle, PyTorch 2.11.0, fp16, batch 128, 3,000 images, flip TTA)

| run | wall time | in script |
| --- | --- | --- |
| cold | 93.1 s | 86.0 s |
| warm | 55.7 s | 51.9 s |

Peak GPU memory 1,841 MiB; weights bundle 402.6 MB. The output matches the shipped `predictions.csv` to a maximum confidence difference of 1.9e-4 with one label flip in 3,000 (fp16 kernel differences at the threshold).

## Bonus items and limits

Grad-CAM (own implementation) shows all members attend to the inner face, mainly the eye and upper-nose region, and not the border; maps stay stable under JPEG, blur and noise. A TensorFlow export matches PyTorch to 4.2e-07 in confidence on 500 validation images; the Swin-T SavedModel came from a manual conversion that the export script cannot reproduce. Main limits: validation comes from the training distribution, so generalisation to unseen generators is untested; labels contain conflicting near-duplicates; most single-run comparisons are within seed noise.
