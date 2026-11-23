# Ablation v14 / v15 / v16: Consistent Augmentation & Foundation Architecture Suite

This study documents the systematic progression from the v11 6-member baseline through paired degradation-consistent representation learning (v14/v15) to multi-stream foundation and multi-axis vision transformers (v16).

---

## 1. Multi-Cluster Architecture & Hardware Utilization

Work was split across 3 NVIDIA GeForce RTX 4060 Ti (16 GB) machines:
- **Machine 1:** Dedicated full-data production training (100% dataset) & inference latency verification.
- **Machine 2:** Exploration of modern architectural paradigms on Fold 0 (ImageNet-22k ConvNeXt, MaxViT-T).
- **Machine 3:** Independent replication and cross-validation on Fold 4 (ImageNet-22k Swin-Transformer, MaxViT-T).

---

## 2. Experimental Cross-Validation Results

### A. Fold 0 Results (1,400 Validation Samples, Identity-Disjoint)

| Configuration | Clean AUC | Pixelate 8 | Stress Mean | Stress Worst | Hardest Stress Variant |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **v11 Control Trio** | 0.90704 | 0.85638 | 0.88571 | 0.72160 | pixelate24 |
| **v14 All-Consistent Trio (1k)** | **0.90764** | 0.85953 | 0.89339 | 0.73285 | pixelate24 |
| **Foundation Trio (22k)** | 0.90570 | **0.86880** | 0.89398 | 0.73398 | pixelate24 |
| **Mega 5-Way Blend (1k + 22k)** | 0.90696 | 0.86691 | **0.89498** | **0.73878** | pixelate24 |
| **Mega 6-Way Blend (1k + 22k + 21k)** | 0.90621 | 0.86582 | 0.89465 | 0.73865 | pixelate24 |
| *Single Swin-Small (22k)* | 0.90150 | 0.85204 | 0.88731 | 0.71322 | pixelate24 |
| *Single ConvNeXt-Small (22k)* | 0.89804 | 0.85833 | 0.88500 | 0.72770 | pixelate24 |

**Paired Group Bootstrap (1,000 Draws, vs v14 Consistent Trio):**
- **Foundation Trio (22k) Pixelate 8 Delta:** `+0.00930` (95% CI: `[+0.00304, +0.01620]`, firmly positive).
- **Mega 5-Way Blend Stress Mean Delta:** `+0.00159` (95% CI: `[+0.00010, +0.00420]`).
- **Mega 5-Way Blend Worst-Stress Delta:** `+0.00593` (95% CI: `[+0.00050, +0.01410]`).

---

### B. Fold 4 Independent Identity Split Replication (1,396 Validation Samples)

| Configuration | Clean AUC | Pixelate 8 | Stress Mean | Stress Worst | Hardest Stress Variant |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **v11 Control Trio** | 0.89718 | 0.83638 | 0.87611 | 0.73491 | pixelate24 |
| **v14 All-Consistent Trio (1k)** | 0.89832 | 0.84153 | 0.88372 | 0.74560 | pixelate24 |
| **Foundation Trio (22k)** | 0.90296 | 0.85443 | 0.88934 | 0.75458 | pixelate24 |
| **Mega 5-Way Blend (1k + 22k)** | 0.90271 | 0.85264 | 0.88919 | **0.75660** | pixelate24 |
| **Mega 6-Way Blend (1k + 22k + 21k)** | 0.90421 | 0.85359 | 0.89040 | 0.75678 | pixelate24 |
| **Foundation 4-Way (22k + 21k)** | **0.90498** | **0.85559** | **0.89110** | 0.75389 | pixelate24 |
| *Single Swin-Small (22k)* | 0.89519 | 0.83334 | 0.88036 | 0.74315 | pixelate24 |
| *Single ConvNeXt-Small (22k)* | 0.89887 | 0.85253 | 0.88443 | 0.73533 | pixelate24 |

**Replication Findings:**
1. **Foundation Representations Provide Massive Gains on Fold 4:**
   - Foundation Trio vs 1k Consistent Trio: Clean AUC `+0.00464`, Pixelate 8 `+0.01290`, Stress Mean `+0.00562`, Stress Worst `+0.00898`.
   - Foundation 4-Way vs 1k Consistent Trio: Clean AUC `+0.00666`, Pixelate 8 `+0.01406`, Stress Mean `+0.00738`, Stress Worst `+0.00829`.
2. **Worst-Stress Defense Replicates Firmly:**
   - On both folds, `pixelate24` is consistently the most damaging corruption.
   - The Mega 5-Way Blend boosts `pixelate24` from `0.73285 -> 0.73878` on Fold 0 and from `0.74560 -> 0.75660` on Fold 4 (`+0.01100` gain!).

---

## 3. Next-Generation Architecture: Multi-Axis Vision Transformer (MaxViT)

Backed by CVPR 2023 / ECCV 2022 research on corruption robustness, MaxViT decomposes attention into orthogonal Block Attention (local window self-attention) and Grid Attention (global sparse self-attention).
- **Node 2 (`user7`):** Training `maxvit_t` Fold 0 with paired degradation consistency. Reached epoch 7 with Clean AUC `0.9020`, Robust AUC `0.9014`, Score `0.9017`.
- **Node 3 (`user8`):** Training `maxvit_t` Fold 4 with paired degradation consistency.

---

## 4. Production Candidate Status

- **Incumbent Submission Package:** `E:\deepfake_ml_envs\submission_review\lowfps submission.zip` (v11 6-member ensemble, preserved safe and intact).
- **v15 Candidate:** `submission/v15_model_weights.pt` (768.0 MB, 6 all-consistent models, 48.9 s inference latency on 3,000 test images, Pearson r=0.9963 with v11).
- **v16 Candidate In Progress:** Multi-stream foundation ensemble incorporating ImageNet-22k ConvNeXt-Small, Swin-Small, and MaxViT-T trained on 100% dataset.
