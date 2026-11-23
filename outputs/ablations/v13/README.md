# Fold 0 label-noise experiments

The comparison holds architecture, seed, optimizer, mosaic augmentation and fixed
stopping epoch constant. All models are ImageNet-initialized ResNet-50 with the
v11 training recipe. Fold 0 is used as a screening set. It has been used in prior
experiments, so a promising gain needs independent confirmation on other folds.

- Control: binary cross entropy (BCE) with label smoothing 0.05.
- GCE: the first two epochs use BCE, then the rest use generalized cross entropy
  with q=0.7. The bounded loss gives very confidently contradicted labels less
  influence on the gradient.
- Soft relabel: BCE with the same smoothing and a modified training target for
  the 463 rows flagged in the previous inner-fold analysis. Each changed target
  is 25% original label and 75% probability from a model trained on other
  groups in the outer training partition. The soft split changes no rows in
  validation and retains all 5,512 eligible training examples.

Evaluate original and horizontal flip and average their logits for each image.
Report clean, mean and worst AUC over the existing 23-variant suite, plus
pixelate8 AUC. Compare to the matched control with 1,000 paired bootstrap draws
that resample whole near-duplicate groups. The default-suite switching rule
for the shipped ensemble remains the one in docs/technical_report.md (section 5). A single-model
screen can identify candidates for that comparison.

From the project root, run the split generation with
`python outputs/ablations/v13/make_soft_split.py`. The shell scripts capture
the host training commands. The evaluator reads the corresponding fixed
checkpoints and writes a comparison JSON and per-image validation logits.

## Outcome

| ResNet-50 training method | Clean AUC | Mean of 23 | Worst of 23 | Pixelate8 |
| --- | ---: | ---: | ---: | ---: |
| BCE control | 0.90322 | 0.90111 | 0.89343 | 0.84783 |
| GCE after two BCE epochs | 0.89857 | 0.89681 | 0.88643 | 0.82244 |
| BCE with soft relabeling | 0.90272 | 0.90166 | 0.89617 | 0.84717 |

For the ensemble substitution, keep the archived v8 seed-42 ConvNeXt-Small
and Swin-T logits fixed, and replace only the ResNet-50 logits. The retrained
BCE control reproduced the archived trio's AUC on all reported summaries.
The GCE swap lowered clean and worst AUC by 0.00294 and 0.00600. The soft-label
swap gained 0.00005 clean (95% paired interval -0.00172 to +0.00178) and
0.00105 worst (interval -0.00172 to +0.00312). Both were screened on fold 0
with one training seed. The six-member v11 submission remains the selected
model.

The ensemble check is reproducible with
`python outputs/ablations/v13/compare_trio.py <archived-v8-cache-directory>`
from the project root. The three JSON files contain the unrounded scores and
paired intervals. See Zhang and Sabuncu, *Generalized Cross Entropy Loss for
Training Deep Neural Networks with Noisy Labels* (NeurIPS 2018) for GCE.
