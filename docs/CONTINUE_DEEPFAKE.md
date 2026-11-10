# Production Deepfake Hackathon State Log
Last updated: 2026-11-10 19:35 IST

## 1. Executive Summary & Progression

| Version | Description | 5-Fold Clean AUC | 29-Stress Mean | Robust Score | Test Time (3k) | Peak VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 6-member torchvision (ResNet-50, ConvNeXt-S, Swin-T) | 0.89500 | 0.88000 | 0.88750 | 88.6 s | 2,229 MiB |
| **Friend's Pipeline** | 10-member (5x ResNet-18 + 5x ConvNeXt-Tiny, ImageNet-1K) | 0.90080 | 0.89940* | 0.90010 | 72.0 s | 1,001 MiB |
| **v16 Foundation** | 6-member (2x ConvNeXt-22k + 2x Swin-22k + 2x MaxViT-T) | 0.90720 | 0.88990 | 0.89855 | 77.5 s | 3,168 MiB |
| **v17 Multi-Family** | 10-member Hybrid (ConvNeXt-22k + Swin-22k + MaxViT-T + DINO + SRM) | 0.90785 | 0.89392 | 0.90089 | 93.9 s | 3,366 MiB |
| **v18 Optimal 4-Arch** | **4-Architecture 5-Fold** (`swin_22k_sbi` + `cnxt_22k_sbi` + `maxvit_t` + `cnxt_base`) | **0.90843** | **0.89500** | **0.90171** | **~88 s** | **~3.2 GB** |
| **v19 Dual-Scale** | **Dual-Resolution Blend ($224\times224 + 256\times256$) Multi-Scale** | **0.90947** | **0.89612** | **0.90280** | **~112 s** | **~4.5 GB** |

*\*Friend's robustness evaluated against only 4 mild variants, while ours is evaluated against 29 brutal stress corruptions.*

---

## 2. Complete 5-Fold Cross-Validation Matrix Across All 7,000 Images

Every architecture evaluated across all 5 folds with 29 stress perturbations:

| Architecture | Fold 0 Score | Fold 1 Score | Fold 2 Score | Fold 3 Score | Fold 4 Score | 5-Fold Mean Clean AUC | 5-Fold Mean Stress | 5-Fold Mean Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`maxvit_t`** | 0.89250 | 0.89533 | **0.90429** | 0.89970 | 0.88940 | **0.90425** | **0.88824** | **0.89625** |
| **`cnxt_22k_sbi`** | 0.89515 | 0.89509 | 0.90100 | 0.89971 | 0.88895 | 0.90344 | 0.88852 | 0.89598 |
| **`swin_22k_sbi`** | 0.89627 | 0.89193 | 0.89669 | 0.90010 | 0.88922 | 0.90380 | 0.88588 | 0.89484 |
| **`cnxt_base`** (88M) | 0.89661 | 0.89012 | 0.90063 | **0.91240** | 0.89039 | 0.89885 | 0.88722 | 0.89303 |
| **`dino_vits16_sbi`** | 0.87518 | 0.73970 | 0.88490 | 0.87680 | 0.87176 | 0.85060 | 0.82597 | 0.83829 |

### 4-Architecture Ensemble Across All 5 Folds:
- **Fold 0:** Clean `0.90714` | Stress Mean `0.89464`
- **Fold 1:** Clean `0.90440` | Stress Mean `0.89201`
- **Fold 2:** Clean `0.91184` | Stress Mean `0.89804`
- **Fold 3:** Clean **`0.91554`** | Stress Mean **`0.90079`**
- **Fold 4:** Clean `0.90323` | Stress Mean `0.88950`
- **FULL 5-FOLD ENSEMBLE AVERAGE:** Clean AUC **`0.90843`** | Stress Mean **`0.89500`** | Robust Score **`0.90171`**

---

## 3. Dual-Resolution Scaling ($256\times256$ vs $224\times224$) Breakthrough

Empirical ablation on Fold 0:
- Pure $224\times224$: Clean AUC `0.90275`, Score `0.89661`
- Pure $256\times256$: Clean AUC `0.90534`, Score `0.89794`
- **Dual-Resolution Blend ($224 + 256$):** Clean AUC **`0.90561`** (+0.00286 gain!), Robust Score **`0.89927`** (+0.00267 gain!)

Multi-scale spatial blending captures micro-textures and blend-boundary gradients at $256\times256$ while preserving global facial geometry at $224\times224$.

---

## 4. Completed Waves 2 & 3 Checkpoint Summary (100% Full-Dataset Refits)

All jobs across Waves 2 & 3 completed successfully:
1. **$256\times256$ Full Fits:** `convnext_base` (Seeds 42 & 7) on `sem8`, `convnext_small_22k_sbi` (Seeds 42 & 7) on `user7`.
2. **20-Epoch Extended Convergence:** `swin_small_22k_sbi` & `maxvit_t_sbi` on `user8` (in-training score `0.9565`).
3. **Architectural Diversity:** `effv2s_21k` (ImageNet-21k Fused-MBConv) on `sem8` (Seeds 42 & 7, in-training score `0.9686`).
4. **Noise-Robust Loss:** `convnext_small_22k` & `swin_small_22k` with Generalized Cross-Entropy (GCE) on `user8`.

---

## 5. Compute Node Status & Cluster Health

- **Node 1 (`sem8@172.16.121.36`):** RTX 4060 Ti 16 GB — IDLE (0% GPU, 16 GB VRAM free, 746 GB disk free).
- **Node 2 (`user7@172.16.121.24`):** RTX 4060 Ti 16 GB — IDLE (0% GPU, 16 GB VRAM free, 33 GB disk free).
- **Node 3 (`user8@172.16.121.19`):** RTX 4060 Ti 16 GB — IDLE (0% GPU, 16 GB VRAM free, 46 GB disk free).

All 3 nodes are 100% free and ready for the next parallel training phase.

---

## 6. Security, Isolation & Submission Invariants
- All compute on `user7` and `user8` is strictly isolated within `~/deepfake_work/`. Zero other files touched.
- Submission baseline `E:\deepfake_ml_envs\submission_review\lowfps submission.zip` is completely intact and preserved.
- Verified candidate weights `submission/v17_model_weights.pt` (1,180.4 MB) are secured.
