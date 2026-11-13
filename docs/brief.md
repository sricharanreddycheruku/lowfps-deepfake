# Deepfake face detection: brief report

Team lowfps - Deepfake ML Hackathon, IIIT Bangalore (Synergy '26). Full details: `technical_report.pdf`.

## Architecture

![Ensemble of eight members (four ImageNet-pretrained backbones at seeds 42 and 7) with flip TTA and logit averaging](report_architecture.png|0.9)

The ensemble has eight members: four backbones fine-tuned end to end from ImageNet weights, each trained at seeds 42 and 7. Every member has a Dropout(0.3) plus single-logit head and runs at 224 px. The backbones are:
- Swin-Small (48.8M parameters) and ConvNeXt-Small (49.5M), both with ImageNet-22k pretraining (timm);
- MaxViT-T (30.4M) and ConvNeXt-Base (87.6M), from torchvision.

The confidence is the sigmoid of the mean logit over 8 members and 2 views (original and horizontal flip), and the decision threshold is 0.5. The members come from three families (modern CNN, shifted-window transformer, convolution plus multi-axis attention). Their errors are still strongly correlated, so the ensemble gain over the best member is about 0.003 to 0.004 AUC. No deepfake-specific detector and no external data were used. Weights are stored in fp16 (867 MB).

## Training and augmentation

- **Loss and optimiser.** BCE with logits and label smoothing 0.05. AdamW, weight decay 0.05, learning rate 3e-4 for head and backbone alike, batch 48 (32 for ConvNeXt-Base). One warm-up epoch, then cosine decay over 12 epochs. EMA of weights (decay 0.99), fp16 autocast.
- **Final weights.** The members are refit on all labelled images, with the 108 conflicting-label images dropped. The weights after epoch 9 are used, with no selection on validation.
- **Augmentation.** Random resized crop, flip and rotation. Then, in random order: JPEG quality 30 to 95, blur, Gaussian noise, down/up-scaling and sharpening. Also colour jitter, grayscale, random erasing, and a random mosaic (pixelation) with probability 0.5.
- **Two additions in the final recipe:**
  - *Self-Blended Images* (probability 0.15, the two ImageNet-22k members only): a training real is blended with an altered copy of itself inside a soft, warped face mask and labelled fake.
  - *Paired consistency*: from epoch 3, half of each batch is also given a stronger degradation, with a target pulled halfway towards the model's own prediction on the undegraded image.

## Validation strategy

Near-duplicate images (perceptual hash, Hamming distance 6 or less) are kept together in a stratified group split. Five folds are built this way, so no near-duplicate group straddles training and validation. Each architecture was trained five times, each time on four folds, and scored on the fold it did not see. Pooled, this gives one held-out score for each of the 7,000 labelled images. Robustness is measured on 22 deterministic degradations of the validation images: JPEG 90 to 30, blur, noise, down-scaling, brightness, contrast, grayscale, crop, rotation, flip, resize round-trips and one combination. Further extended and stress suites include pixelation. The rule for choosing and shipping the ensemble was written down before the pooled numbers were computed. Test images were used for inference only.

## Headline numbers (five-fold out-of-fold, 7,000 images, flip TTA)

| model | clean AUC [95% CI] | mean of 22 | worst | pixelate 8 px | accuracy |
| --- | --- | --- | --- | --- | --- |
| ConvNeXt-Small, ImageNet-22k | 0.9033 | 0.9020 | 0.8962 | 0.8553 | 0.861 |
| Swin-Small, ImageNet-22k | 0.9035 | 0.9015 | 0.8929 | 0.8291 | 0.855 |
| MaxViT-T | 0.9046 | 0.9036 | 0.8997 | 0.8550 | 0.870 |
| ConvNeXt-Base | 0.8815 | 0.8796 | 0.8717 | 0.8215 | 0.790 |
| **Four-member ensemble (final recipe)** | **0.9080 [0.9010, 0.9155]** | **0.9068** | **0.9018** | **0.8615** | **0.872** |

- **Against the previous recipe.** On the two folds where the previous system (v11: ResNet-50, ConvNeXt-Small, Swin-T) was also cross-validated, the new set leads on every summary. Fold 0: clean 0.9074 against 0.9070, worst 0.9049 against 0.9002, pixelation 0.8677 against 0.8584. Fold 4: clean 0.9038 against 0.8972, pixelation 0.8559 against 0.8364.
- **Held-out score of the shipped weights.** The deployed bundle has two seeds per architecture and was refit on all labelled data, so it has no held-out score.
- **Caveats.** The architectures were shortlisted after looking at folds 0 and 4, so the pooled figure is mildly optimistic. The hidden test degradations may differ from our suite. The weakest case is coarse pixelation (24 px: 0.734).
- **Earlier rounds.** Many candidates were tested on fold 0 against fixed switching rules with bootstrap intervals: noise-residual and DCT members, SAM, ELR, cutout, stacking, label cleaning and heavy augmentation. Most gave no reliable gain. Mosaic augmentation was the clearest win, raising 8 px pixelation from 0.72 to 0.86.

## Efficiency

On the RTX 4060 Ti the shipped script takes 101 s for the 3,000 test images (batch 128, flip TTA, fp16 autocast), with peak GPU memory 3.7 GB. Two runs give byte-identical output. On a Kaggle Tesla T4 it takes 202.1 s cold (194.8 s inside the script) and 205.0 s on a second run, peak GPU memory 3.7 GB, under the 5-minute budget. fp16 storage changes the confidences by at most 4.5e-4 against fp32 weights, with no label flip.

## Bonus items and limits

- **Grad-CAM and TensorFlow export.** Both cover the previous six-member system (v11), not the final bundle.
  - Grad-CAM uses our own implementation. All members attend to the inner face, mainly the eye and upper-nose region, not the border, and the maps stay stable under JPEG, blur and noise.
  - The TensorFlow export matches PyTorch to 3.0e-07 in confidence on 500 validation images. Swin-T is rebuilt directly in TensorFlow ops.
- **Main limits.** Validation comes from the training distribution, so generalisation to unseen generators is untested. The labels contain conflicting near-duplicates. The final weights have no held-out score.
