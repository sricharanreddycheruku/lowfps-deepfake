# Grad-CAM notes

Implementation: `src/gradcam.py` (own code, forward hook plus a tensor hook on the
target layer, no Grad-CAM library). The heat map is the ReLU of the gradient-weighted
sum of the target activations for the single "fake" logit, bilinearly upsampled to
224x224 and min-max scaled per image. These figures are of the shipped bundle
(`submission/model_weights.pt`): the four architectures of the final ensemble.
Target layers: last stage of `features` for ConvNeXt-Base, last stage of the timm
`backbone.stages` for ConvNeXt-Small (ImageNet-22k), the last block of MaxViT-T, and
the output of the final norm of Swin-Small (ImageNet-22k), a channels-last 7x7xC
tensor permuted to channels-first. The maps are therefore 7x7 before upsampling and
coarse by construction.

Images come from the fold-0 validation split (1408 images). The bundle members were refit
on all labelled rows, so they did see these images in training: the maps show where the
deployed model looks, not held-out behaviour. Images are classified by
the fused ensemble at the bundle threshold 0.5: 16 random (seed 42) images each for true
positives, true negatives, false positives and false negatives. Which images were drawn
is listed in `gradcam_cases.csv`. Single forward pass, no flip TTA. The per-architecture
figures show one representative member of that backbone; the ensemble figures average the
normalised maps of all eight bundle members (four architectures, seeds 42 and 7).

## Figures

| file | content |
| --- | --- |
| `gradcam_<arch>_{tp,tn,fp,fn}.png` | 4x4 grids per member, arch in convnext_small_22k, swin_small_22k, maxvit_t, convnext_base |
| `gradcam_<arch>_robust.png` | 6 images (3 fake, 3 real) clean / JPEG 30 / blur 2 / noise 10 |
| `gradcam_ensemble_{tp,tn,fp,fn}.png` | mean of the members' normalised maps, same images |
| `gradcam_ensemble_robust.png` | ensemble-averaged maps for the robustness images |
| `gradcam_stats.json` | share of map mass in four face regions per case and condition |

The region statistics split the 224x224 frame into a border ring (outer 15 percent),
an eye band (rows 25-48 percent), a mouth band (rows 62-85 percent) and the rest of the
centre. The numbers below are enrichments: share of map mass in the region divided by
the region's share of the area (1.0 means evenly spread, above 1 means the model
attends there more than its size suggests).

## Observations

**True positives (fakes found).** All four members light up the inner face, most
strongly the eye and upper-nose region (eye-band enrichment ConvNeXt-Small 2.4,
Swin-Small 1.9, MaxViT-T 1.3, ConvNeXt-Base 3.7, ensemble 2.3) and the adjacent
forehead and cheeks. The border ring is almost ignored (0.14 to 0.72, ensemble 0.42).
The mouth band is used a little by MaxViT-T (1.2) and ConvNeXt-Small (1.1) but barely by
ConvNeXt-Base (0.3). Examples with glasses show the map following the glasses frame and
eyes. Nothing in the maps points to a seam along the face contour; the evidence the
models use sits inside the face, at a texture and detail level that the 7x7 map cannot
resolve any finer.

**True negatives (reals).** The maps are weak and diffuse with no dominant face blob
(ensemble eye band 0.42, border ring 1.24; MaxViT-T leans on the border ring, 1.6): the
gradient of the fake logit has little to point at, so the per-image scaling blows up
small background, neck, hair or chin responses. These grids should be read as "no fake
evidence found", not as a positive explanation of why the image is real.

**False positives (real predicted fake).** They look like the true positives: eye-band
enrichment 1.4 (MaxViT-T) to 2.3 (ConvNeXt-Base), ensemble 1.8, border ring 0.29 to 0.71.
The models fire on the same face region and are simply wrong there. No single image trait
stands out: the grids mix sharp frontal portraits with glasses, pixelated low-resolution
faces and crowded scenes where the map sits on a neighbouring face. Since the training
labels contain conflicting near-duplicates, some of these may be label noise rather than
model error.

**False negatives (fakes missed).** The maps resemble the true-negative ones (ensemble
eye-band enrichment 0.57, border ring 1.08): the face interior is not highlighted; weak
attention lands on the hairline, chin, neck or background. Many of these faces are in
profile, tilted down, motion-blurred or partly covered, so the face region produced no
strong fake evidence.

**Under JPEG 30, blur 2 and noise 10.** The same 6 images keep the same highlighted
region. Correlation with the clean map (mean over the 6 images, JPEG 30 / blur 2 / noise
10): ConvNeXt-Small 0.98 / 0.98 / 0.98, Swin-Small 0.99 / 0.99 / 0.98, MaxViT-T 0.94 /
0.93 / 0.92, ConvNeXt-Base 0.98 / 0.98 / 0.99, ensemble 0.99 / 0.99 / 0.99. The
ensemble's eye-band enrichment stays at 1.47 to 1.52 in all four conditions and the
border ring at 0.81. MaxViT-T's maps move the most under degradation, the ConvNeXt and
Swin maps least, which fits their small spread of AUC across the degradation suite. The
shown fakes keep their high fused probability (for example 0.93 clean to 0.93 under all
three degradations on the first image).

## Caveats

- 7x7 maps (stride 32) cannot separate eyes from eyebrows or a blending boundary from
  skin texture; "eye band" means the rows 25-48 percent of the aligned face.
- Per-image min-max scaling makes every map look equally confident. The caption p of a
  tile carries the real confidence.
- Grad-CAM shows where the gradient is large, not necessarily what the model needs.
  The figures support the statement that the decision uses the face interior and does not
  hinge on the image border or background, but they do not prove which artifact is used.
