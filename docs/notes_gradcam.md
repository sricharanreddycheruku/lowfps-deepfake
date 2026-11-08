# Grad-CAM notes

Implementation: `src/gradcam.py` (own code, forward hook plus a tensor hook on the
target layer, no Grad-CAM library). The heat map is the ReLU of the gradient-weighted
sum of the target activations for the single "fake" logit, bilinearly upsampled to
224x224 and min-max scaled per image. Target layers: last block of `layer4` for
ResNet-50, last stage of `features` for ConvNeXt-Small, and the output of the final
norm of Swin-T (a channels-last 7x7x768 tensor, permuted to channels-first). The maps
are therefore 7x7 before upsampling and coarse by construction.

Images come from the validation split (1408 images, never trained on), classified by
the fused ensemble at the bundle threshold 0.5: 16 random (seed 42) images each for true
positives, true negatives, false positives and false negatives. Which images were drawn
is listed in `outputs/figures/gradcam_cases.csv`. Single forward pass, no flip TTA.

## Figures (`outputs/figures/`, copies in `outputs/bonus/`)

| file | content |
| --- | --- |
| `gradcam_<arch>_{tp,tn,fp,fn}.png` | 4x4 grids per member, arch in resnet50, convnext_small, swin_t |
| `gradcam_<arch>_robust.png` | 6 images (3 fake, 3 real) clean / JPEG 30 / blur 2 / noise 10 |
| `gradcam_ensemble_{tp,tn,fp,fn}.png` | mean of the three members' normalised maps, same images |
| `gradcam_ensemble_robust.png` | ensemble-averaged maps for the robustness images |
| `gradcam_stats.json` | share of map mass in four face regions per case and condition |

The region statistics split the 224x224 frame into a border ring (outer 15 percent),
an eye band (rows 25-48 percent), a mouth band (rows 62-85 percent) and the rest of the
centre. The numbers below are enrichments: share of map mass in the region divided by
the region's share of the area (1.0 means evenly spread, above 1 means the model
attends there more than its size suggests).

## Observations

**True positives (fakes found).** All three members light up the inner face, most
strongly the eye and upper nose region (eye-band enrichment ResNet-50 2.4, ConvNeXt 3.5,
Swin-T 2.5, ensemble 2.7) and the adjacent forehead and cheeks (rest of the centre about
2). The border ring is almost ignored (0.06 to 0.33). The mouth band is used by
ResNet-50 (1.1) but little by ConvNeXt (0.2) and Swin-T (0.8). Examples with glasses show
the map following the glasses frame and eyes. Nothing in the maps points to a seam along
the face contour; the evidence the models use sits inside the face, at a texture and
detail level that the 7x7 map cannot resolve any finer.

**True negatives (reals).** ResNet-50 and ConvNeXt give weak, diffuse maps with no
dominant face blob (border ring 1.1 to 1.2, eye band 0.4 to 0.6, mouth band 1.2 to 1.35):
the gradient of the fake logit has little to point at, so the per-image scaling blows up
small background, neck, hair or chin responses. Swin-T's maps are mostly empty for
confident reals, because the ReLU removes everything when no region argues for "fake".
These grids should be read as "no fake evidence found", not as a positive explanation of
why the image is real.

**False positives (real predicted fake).** They look like the true positives: eye-band
enrichment 2.1 (ResNet-50), 3.3 (ConvNeXt), 2.4 (Swin-T), border ring 0.1 to 0.45. The
models fire on the same face region and are simply wrong there. No single image trait
stands out: the grids mix sharp frontal portraits with glasses, pixelated low-resolution
faces and crowded scenes where the map sits on a neighbouring face. Since the training
labels contain conflicting near-duplicates, some of these may be label noise rather than
model error.

**False negatives (fakes missed).** The maps resemble the true-negative ones (eye-band
enrichment 0.4 to 0.8, border ring 1.06 for ResNet-50, ConvNeXt and the ensemble): the
face interior is not highlighted; weak attention lands on the hairline, chin, neck or
background. Many of these faces are in profile, tilted down, motion-blurred or partly
covered, so the face region produced no strong fake evidence.

**Under JPEG 30, blur 2 and noise 10.** The same 6 images keep the same highlighted
region. Correlation with the clean map (mean over the 6 images, JPEG 30 / blur 2 / noise
10): ResNet-50 0.97 / 0.94 / 0.96, ConvNeXt 0.97 / 0.90 / 0.98, Swin-T 0.997 / 0.995 /
0.993, ensemble 0.98 / 0.91 / 0.97. The ensemble's eye-band enrichment stays at
1.9 to 2.0 in all four conditions. Blur 2 moves the CNN maps the most; Swin-T's maps are
almost unchanged, which fits its small spread of AUC across the degradation suite. The
Swin-T probabilities of the three shown fakes drop by at most 0.04. On the real images the degradation
occasionally makes a small spurious spot appear (Swin-T, blur 2, one real face: p rises
from 0.18 to 0.32 with a blob at the top-left corner; noise 10 on a second real face
highlights a cheek), which is the typical false-alarm route under degradation.

## Caveats

- 7x7 maps (stride 32) cannot separate eyes from eyebrows or a blending boundary from
  skin texture; "eye band" means the rows 25-48 percent of the aligned face.
- Per-image min-max scaling makes every map look equally confident. The caption p of a
  tile carries the real confidence.
- Grad-CAM shows where the gradient is large, not necessarily what the model needs.
  The figures support the statement that the decision uses the face interior and does not
  hinge on the image border or background, but they do not prove which artifact is used.

Two sets of figures ship. The files directly in `gradcam/` are from the first fold-0
three-member bundle, and the statistics quoted in the technical report come from them.
`gradcam/fold0_p50/` holds the same figures for the fold-0 trio of the final recipe
(mosaic probability 0.5), with its own `gradcam_stats.json` and `gradcam_cases.csv`.
Neither set covers the six refit members, which were trained on the validation images.
