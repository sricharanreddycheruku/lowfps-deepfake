"""Backbone construction for the ensemble members.

All backbones come from torchvision with ImageNet weights only (no
deepfake-specific pretrained detectors, per the rules). The classifier
head is replaced in every case with a Dropout + single-logit Linear layer,
so the training script only ever deals with raw logits for
BCEWithLogitsLoss. At inference time pretrained=False, since the bundled
weights get loaded over the architecture instead of downloading anything.

Usage:
    python src/models.py
    (builds each supported arch, runs a forward pass, prints and saves
    parameter counts to outputs/models_params.json)
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
]


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
    raise ValueError(f"unsupported arch: {arch}")


def _self_check():
    results = {}
    for arch in SUPPORTED_ARCHS:
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
    _self_check()
