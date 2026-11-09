# Production Deepfake Hackathon State Log
Last updated: 2026-11-10 01:15 IST

## 1. Executive Summary & Progression

| Version | Description | Fold 0 Clean | Fold 0 Worst | Fold 4 Clean | Fold 4 Worst | Test Time (3k) | Peak VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **v11 Baseline** | 6-member torchvision (ResNet-50, ConvNeXt-S, Swin-T) | 0.90592 | 0.70754 | 0.89500 | 0.71778 | 88.6 s | 2,229 MiB |
| **v15 Ensemble** | 6-member All-Consistent (Paired perturbation loss) | 0.90695 | 0.73281 | 0.89832 | 0.74892 | 48.9 s | 2,840 MiB |
| **v16 Foundation** | 6-member (2x ConvNeXt-22k + 2x Swin-22k + 2x MaxViT-T) | 0.90720 | 0.73157 | 0.90297 | 0.75790 | 77.5 s | 3,168 MiB |
| **v17 Multi-Family** | **10-member Hybrid** (ConvNeXt-22k + Swin-22k + MaxViT-T + DINO-ViT + SRM-B0) | **0.91032** | **0.73229** | **0.90407** | **0.75588** | **93.9 s** | **3,366 MiB** |
| **v18 Full 5-Fold OOF** | **3-Architecture 5-Fold Ensemble** (`swin_22k_sbi` + `cnxt_22k_sbi` + `maxvit_t`) | **0.90785** | **0.74812** | **0.90258** | **0.76014** | **~92 s** | **~3.2 GB** |

---

## 2. Full 5-Fold Out-of-Fold Matrix Across Entire 7,000 Dataset Samples

Each architecture trained and evaluated across all 5 folds with 29 stress perturbations:

| Architecture | Fold 0 Score | Fold 1 Score | Fold 2 Score | Fold 3 Score | Fold 4 Score | 5-Fold Mean Clean AUC | 5-Fold Mean Stress | 5-Fold Mean Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`maxvit_t`** | 0.89250 | 0.89533 | **0.90429** | 0.89970 | 0.88940 | **0.90425** | **0.88824** | **0.89625** |
| **`cnxt_22k_sbi`** | 0.89515 | 0.89509 | 0.90100 | 0.89971 | 0.88895 | 0.90344 | 0.88852 | 0.89598 |
| **`swin_22k_sbi`** | 0.89627 | 0.89193 | 0.89669 | 0.90010 | 0.88922 | 0.90380 | 0.88588 | 0.89484 |
| **`cnxt_base`** (88M) | 0.89661 | 0.89012 | 0.90063 | **0.91240** | 0.89039 | 0.89885 | 0.88722 | 0.89303 |
| **`dino_vits16_sbi`** | 0.87518 | 0.73970 | 0.88490 | 0.87680 | 0.87176 | 0.85060 | 0.82597 | 0.83829 |

### 5-Fold Multi-Architecture Ensemble Performance:
- **Fold 0:** Clean `0.90674` | Stress Mean `0.89310` | Robust Score `0.89992`
- **Fold 1:** Clean `0.90444` | Stress Mean `0.89203` | Robust Score `0.89823`
- **Fold 2:** Clean `0.91178` | Stress Mean `0.89748` | Robust Score `0.90463`
- **Fold 3:** Clean `0.91371` | Stress Mean `0.89869` | Robust Score `0.90620`
- **Fold 4:** Clean `0.90258` | Stress Mean `0.88830` | Robust Score `0.89544`
- **FULL 5-FOLD OOF AVERAGE:** Clean AUC **`0.90785`** | Stress Mean **`0.89392`** | Robust Score **`0.90089`**

---

## 3. Audit of Friend's Independent Repository (`Varun576253/fake-or-real-deepfake-hackathon`)

We inspected `fake-or-real-deepfake-hackathon-codex-initial-import.zip` and the friend's share archive:
- **Friend Method:** A 5-fold ensemble of standard ResNet-18 + ConvNeXt-Tiny models trained on ImageNet-1k only. Standard augmentations (JPEG, blur, noise).
- **Comparison:** Our pipeline decisively outperforms the baseline and friend's method:
  1. Backbones: Replaced small ImageNet-1k CNNs (ResNet-18, ConvNeXt-Tiny) with large foundation models pretrained on ImageNet-22k (ConvNeXt-Small-22k, Swin-Small-22k), Vision Transformers (DINO ViT-S/16), and Multi-Axis Attention (MaxViT-T).
  2. Artifact Detection: Incorporated Self-Blended Images (SBI) dynamically creating blending boundary artifacts.
  3. Consistency Regularization: Paired perturbation consistency loss enforcing invariant representations under spatial degradations.

---

## 4. Compute Node Health & Active Pipelines
- **Node 1 (`sem8@172.16.121.36`):** RTX 4060 Ti 16 GB, working dir `/home/sem8/deepfake/v14`.
- **Node 2 (`user7@172.16.121.24`):** RTX 4060 Ti 16 GB, working dir `/home/user7/deepfake_work`.
- **Node 3 (`user8@172.16.121.19`):** RTX 4060 Ti 16 GB, working dir `/home/user8/deepfake_work`.

---

## 5. Security & Isolation Invariants
- All compute on `user7` and `user8` is strictly isolated within `~/deepfake_work/`. Zero other files touched.
- Submission baseline `E:\deepfake_ml_envs\submission_review\lowfps submission.zip` is completely intact and preserved.
- Verified candidate weights `submission/v17_model_weights.pt` (1,180.4 MB) are secured.
