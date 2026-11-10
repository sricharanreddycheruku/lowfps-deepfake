"""Backbone construction for the ensemble members.

Backbones come from torchvision (ImageNet-1k weights) or timm
(ImageNet-21k/22k pretraining, fine-tuned on ImageNet-1k); no
deepfake-specific pretrained detectors, per the rules. The classifier
head is replaced in every case with a Dropout + single-logit Linear layer,
so the training script only ever deals with raw logits for
BCEWithLogitsLoss. At inference time pretrained=False, since the bundled
weights get loaded over the architecture instead of downloading anything.

Usage:
    python src/models.py [arch ...]
    (builds each supported arch, or just the ones named, runs a forward
    pass, prints and merges parameter counts into outputs/models_params.json)
"""

import json
import os

import torch
import torch.nn as nn
from torchvision import models

SUPPORTED_ARCHS = [
    "efficientnet_b0",
    "efficientnet_v2_s",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "swin_t",
    "swin_s",
    "resnet50",
    "regnet_y_16gf",
    "maxvit_t",
    "vit_b_16",
    "swin_v2_t",
    "srm_b0",
    "dct_b0",
    "patch_b0",
    "convnext_small_22k",
    "swin_small_22k",
    "swin_base_22k",
    "effv2s_21k",
    "dino_vits16",
]

# timm weight tag and the input normalisation that backbone was pretrained with
TIMM_ARCHS = {
    # Original DINO (ViT-S/16), self-supervised on ImageNet-1k.
    "dino_vits16": ("vit_small_patch16_224.dino",
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "convnext_small_22k": ("convnext_small.fb_in22k_ft_in1k",
                           (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "swin_small_22k": ("swin_small_patch4_window7_224.ms_in22k_ft_in1k",
                       (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "swin_base_22k": ("swin_base_patch4_window7_224.ms_in22k_ft_in1k",
                      (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "effv2s_21k": ("tf_efficientnetv2_s.in21k_ft_in1k",
                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _srm_bank():
    """Fixed high-pass kernels (3x3 KB, 5x5 KV, 3x3 second-order edge), all
    padded to 5x5. Shape (3, 5, 5)."""
    kb = torch.tensor([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=torch.float32) / 4.0
    kv = torch.tensor([[-1, 2, -2, 2, -1],
                       [2, -6, 8, -6, 2],
                       [-2, 8, -12, 8, -2],
                       [2, -6, 8, -6, 2],
                       [-1, 2, -2, 2, -1]], dtype=torch.float32) / 12.0
    edge = torch.tensor([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=torch.float32) / 2.0
    bank = torch.zeros(3, 5, 5)
    bank[0, 1:4, 1:4] = kb
    bank[1] = kv
    bank[2, 1:4, 1:4] = edge
    return bank


class SRMEfficientNet(nn.Module):
    """Noise-residual member. The normalised input is mapped back to the
    0-255 scale, passed through a fixed (non-trainable) bank of SRM
    high-pass filters per RGB channel (9 channels), scaled and clamped to
    +/-3, reduced to 3 channels by a learnable 1x1 conv + BN, and fed to an
    ImageNet EfficientNet-B0. Conv ops only, so ONNX / TF export work."""

    def __init__(self, pretrained=True, dropout=0.3, scale=0.5, clamp=3.0):
        super().__init__()
        self.scale = scale
        self.clamp = clamp
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        # grouped conv: input channel c produces output channels 3c..3c+2
        self.register_buffer("srm_weight", _srm_bank().unsqueeze(1).repeat(3, 1, 1, 1))
        self.reduce = nn.Sequential(nn.Conv2d(9, 3, kernel_size=1, bias=False), nn.BatchNorm2d(3))
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier = _replace_head(self.backbone, in_features, dropout)

    def forward(self, x):
        x = (x * self.std + self.mean) * 255.0
        r = torch.nn.functional.conv2d(x, self.srm_weight, padding=2, groups=3)
        r = torch.clamp(r * self.scale, -self.clamp, self.clamp)
        return self.backbone(self.reduce(r))


class PatchEfficientNet(nn.Module):
    """Patch-level member. ImageNet EfficientNet-B0 features cut after the
    stride-16 stage (small receptive field), a 1x1 conv to one logit per
    14x14 patch, and the image logit is the mean of the patch map."""

    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = models.efficientnet_b0(weights=weights).features[:6]
        self.patch_head = nn.Sequential(nn.Dropout(dropout), nn.Conv2d(112, 1, kernel_size=1))

    def patch_logits(self, x):
        return self.patch_head(self.features(x))

    def forward(self, x):
        return self.patch_logits(x).mean(dim=(2, 3))


def _dct_basis():
    """Orthonormal 8x8 DCT-II basis as 64 conv filters, shape (64, 1, 8, 8)."""
    k = torch.arange(8, dtype=torch.float32)
    c = torch.cos((2 * k[None, :] + 1) * k[:, None] * torch.pi / 16.0)
    c[0] *= 1.0 / torch.sqrt(torch.tensor(2.0))
    c = c * 0.5
    return (c[:, None, :, None] * c[None, :, None, :]).reshape(64, 1, 8, 8)


class DCTNet(nn.Module):
    """JPEG-domain member. Luma of the un-normalised image, a stride-8 conv
    with the fixed 8x8 DCT basis (aligned to the JPEG block grid), log(|c|+1),
    then a small randomly initialised CNN (about 1M parameters)."""

    def __init__(self, dropout=0.3):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.register_buffer("luma", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        self.register_buffer("dct_weight", _dct_basis())

        def block(cin, cout, stride):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                                 nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        self.stem_bn = nn.BatchNorm2d(64)
        self.body = nn.Sequential(block(64, 128, 1), block(128, 256, 2), block(256, 256, 2))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = _replace_head(None, 256, dropout)

    def forward(self, x):
        x = (x * self.std + self.mean) * 255.0
        y = (x * self.luma).sum(dim=1, keepdim=True) - 128.0
        c = torch.nn.functional.conv2d(y, self.dct_weight, stride=8)
        c = torch.log(torch.abs(c) + 1.0)
        f = self.body(self.stem_bn(c))
        return self.head(self.pool(f).flatten(1))


class TimmNet(nn.Module):
    """timm backbone (pooled features) + Dropout/Linear single-logit head.

    Inputs arrive normalised with ImageNet mean/std (the shared eval
    transform); if the backbone was pretrained with other constants the
    input is re-normalised here with fixed values, so preprocessing stays
    identical for every member.
    """

    def __init__(self, arch, pretrained=True, dropout=0.3):
        super().__init__()
        import timm

        name, mean, std = TIMM_ARCHS[arch]
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0)
        self.head = _replace_head(None, self.backbone.num_features, dropout)
        a = torch.tensor(IMAGENET_STD) / torch.tensor(std)
        b = (torch.tensor(IMAGENET_MEAN) - torch.tensor(mean)) / torch.tensor(std)
        self.renorm = not (torch.allclose(a, torch.ones(3)) and torch.allclose(b, torch.zeros(3)))
        self.register_buffer("in_scale", a.view(1, 3, 1, 1), persistent=False)
        self.register_buffer("in_shift", b.view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        if self.renorm:
            x = x * self.in_scale + self.in_shift
        return self.head(self.backbone(x))


def _replace_head(model, in_features, dropout=0.3):
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 1),
    )


def build_model(arch, pretrained=True, dropout=0.3):
    """Returns a model that maps (N, 3, H, W) -> (N, 1) raw logits."""
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier = _replace_head(model, in_features, dropout)

    elif arch == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier = _replace_head(model, in_features, dropout)

    elif arch == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = _replace_head(model, in_features, dropout)

    elif arch == "convnext_small":
        weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = _replace_head(model, in_features, dropout)

    elif arch == "convnext_base":
        weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_base(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = _replace_head(model, in_features, dropout)

    elif arch in ("swin_t", "swin_s"):
        if arch == "swin_t":
            weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.swin_t(weights=weights)
        else:
            weights = models.Swin_S_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.swin_s(weights=weights)
        in_features = model.head.in_features
        model.head = _replace_head(model, in_features, dropout)

    elif arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = _replace_head(model, in_features, dropout)

    elif arch == "swin_v2_t":
        weights = models.Swin_V2_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.swin_v2_t(weights=weights)
        in_features = model.head.in_features
        model.head = _replace_head(model, in_features, dropout)

    elif arch == "regnet_y_16gf":
        # the plain ImageNet-1k weights, not the SWAG pretraining variants
        weights = models.RegNet_Y_16GF_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.regnet_y_16gf(weights=weights)
        in_features = model.fc.in_features
        model.fc = _replace_head(model, in_features, dropout)

    elif arch == "maxvit_t":
        weights = models.MaxVit_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.maxvit_t(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = _replace_head(model, in_features, dropout)

    elif arch == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = _replace_head(model, in_features, dropout)

    elif arch == "srm_b0":
        model = SRMEfficientNet(pretrained=pretrained, dropout=dropout)

    elif arch == "dct_b0":
        model = DCTNet(dropout=dropout)

    elif arch == "patch_b0":
        model = PatchEfficientNet(pretrained=pretrained, dropout=dropout)

    elif arch in TIMM_ARCHS:
        model = TimmNet(arch, pretrained=pretrained, dropout=dropout)

    else:
        raise ValueError(f"unsupported arch: {arch}")

    return model


def get_head_module(model, arch):
    """Returns the replaced classifier head module, so the training script
    can give it a different (higher) learning rate than the backbone."""
    if arch in ("efficientnet_b0", "efficientnet_v2_s"):
        return model.classifier
    if arch in ("convnext_tiny", "convnext_small", "convnext_base"):
        return model.classifier[-1]
    if arch in ("swin_t", "swin_s", "swin_v2_t"):
        return model.head
    if arch in ("resnet50", "regnet_y_16gf"):
        return model.fc
    if arch == "maxvit_t":
        return model.classifier[-1]
    if arch == "vit_b_16":
        return model.heads.head
    if arch == "srm_b0":
        return model.backbone.classifier
    if arch == "dct_b0":
        return model.head
    if arch == "patch_b0":
        return model.patch_head
    if arch in TIMM_ARCHS:
        return model.head
    raise ValueError(f"unsupported arch: {arch}")


def gradcam_target_layer(arch, model):
    """Returns the module whose activations/gradients Grad-CAM hooks onto.

    For the CNN-style backbones this is the last conv block before pooling.
    Swin-T has no spatial conv feature map at the end; its last stage's
    norm output is reshaped to a spatial map in src/gradcam.py instead, so
    here we just point at that norm layer.
    """
    if arch == "efficientnet_b0" or arch == "efficientnet_v2_s":
        return model.features[-1]
    if arch in ("convnext_tiny", "convnext_small", "convnext_base"):
        return model.features[-1]
    if arch == "resnet50":
        return model.layer4[-1]
    if arch == "regnet_y_16gf":
        return model.trunk_output[-1]
    if arch == "maxvit_t":
        return model.blocks[-1]
    if arch == "vit_b_16":
        return model.encoder.layers[-1]
    if arch in ("swin_t", "swin_s", "swin_v2_t"):
        return model.norm
    if arch == "srm_b0":
        return model.backbone.features[-1]
    if arch == "dct_b0":
        return model.body[-1]
    if arch == "patch_b0":
        return model.patch_head[-1]
    if arch == "convnext_small_22k":
        return model.backbone.stages[-1]
    if arch == "swin_small_22k":
        return model.backbone.norm
    if arch == "effv2s_21k":
        return model.backbone.bn2
    raise ValueError(f"unsupported arch: {arch}")


def _self_check(archs=None):
    results = {}
    if os.path.exists("outputs/models_params.json"):
        with open("outputs/models_params.json", encoding="utf-8") as f:
            results = json.load(f)
    for arch in archs or SUPPORTED_ARCHS:
        model = build_model(arch, pretrained=True)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1), f"{arch} produced shape {out.shape}"
        n_params = sum(p.numel() for p in model.parameters())
        results[arch] = n_params
        print(f"{arch}: output shape {tuple(out.shape)}, params {n_params:,}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/models_params.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("wrote outputs/models_params.json")


if __name__ == "__main__":
    import sys

    _self_check(sys.argv[1:] or None)
