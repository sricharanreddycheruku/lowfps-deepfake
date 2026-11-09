import argparse
import csv
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TIMM_ARCHS = {
    'convnext_small_22k': ('convnext_small.fb_in22k_ft_in1k', IMAGENET_MEAN, IMAGENET_STD),
    'swin_small_22k': ('swin_small_patch4_window7_224.ms_in22k_ft_in1k', IMAGENET_MEAN, IMAGENET_STD),
    'effv2s_21k': ('tf_efficientnetv2_s.in21k_ft_in1k', (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    'dino_vits16': ('vit_small_patch16_224.dino', IMAGENET_MEAN, IMAGENET_STD),
}

def _srm_bank():
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
    def __init__(self, dropout=0.3, scale=0.5, clamp=3.0):
        super().__init__()
        self.scale = scale
        self.clamp = clamp
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.register_buffer("srm_weight", _srm_bank().unsqueeze(1).repeat(3, 1, 1, 1))
        self.reduce = nn.Sequential(nn.Conv2d(9, 3, kernel_size=1, bias=False), nn.BatchNorm2d(3))
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))

    def forward(self, x):
        x = (x * self.std + self.mean) * 255.0
        r = torch.nn.functional.conv2d(x, self.srm_weight, padding=2, groups=3)
        r = torch.clamp(r * self.scale, -self.clamp, self.clamp)
        return self.backbone(self.reduce(r))

class TimmNet(nn.Module):
    def __init__(self, arch, dropout=0.3):
        super().__init__()
        import timm
        name, mean, std = TIMM_ARCHS[arch]
        self.backbone = timm.create_model(name, pretrained=False, num_classes=0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.num_features, 1))
        a = torch.tensor(IMAGENET_STD) / torch.tensor(std)
        b = (torch.tensor(IMAGENET_MEAN) - torch.tensor(mean)) / torch.tensor(std)
        self.renorm = not (torch.allclose(a, torch.ones(3)) and torch.allclose(b, torch.zeros(3)))
        self.register_buffer('in_scale', a.view(1, 3, 1, 1), persistent=False)
        self.register_buffer('in_shift', b.view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        if self.renorm:
            x = x * self.in_scale + self.in_shift
        return self.head(self.backbone(x))

def build_model(arch, dropout=0.3):
    def head(in_features):
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))

    if arch in ('efficientnet_b0', 'efficientnet_v2_s'):
        m = getattr(models, arch)(weights=None)
        m.classifier = head(m.classifier[-1].in_features)
    elif arch in ('convnext_tiny', 'convnext_small'):
        m = getattr(models, arch)(weights=None)
        m.classifier[-1] = head(m.classifier[-1].in_features)
    elif arch == 'swin_t':
        m = models.swin_t(weights=None)
        m.head = head(m.head.in_features)
    elif arch == 'resnet50':
        m = models.resnet50(weights=None)
        m.fc = head(m.fc.in_features)
    elif arch == 'maxvit_t':
        m = models.maxvit_t(weights=None)
        m.classifier[-1] = head(m.classifier[-1].in_features)
    elif arch == 'srm_b0':
        m = SRMEfficientNet(dropout=dropout)
    elif arch in TIMM_ARCHS:
        m = TimmNet(arch, dropout=dropout)
    else:
        raise ValueError(f'unsupported arch: {arch}')
    return m

class TestDataset(Dataset):
    def __init__(self, paths, sizes, mean, std):
        self.paths = paths
        self.transforms = {
            size: transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]) for size in sorted(set(sizes))
        }

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return {size: t(img) for size, t in self.transforms.items()}

def list_images(test_dir):
    names = [n for n in os.listdir(test_dir) if n.lower().endswith(IMAGE_EXTENSIONS)]
    return [os.path.join(test_dir, n) for n in sorted(names)]

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def fuse(logits, rule, weights):
    if rule == 'mean_logit':
        return sigmoid(logits.mean(axis=0))
    if rule == 'mean_prob':
        return sigmoid(logits).mean(axis=0)
    if rule == 'weighted_logit':
        return sigmoid((np.asarray(weights)[:, None] * logits).sum(axis=0))
    raise ValueError(f'unknown fusion rule: {rule}')

def load_bundle(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        return torch.load(path, map_location='cpu', weights_only=False)

def load_members(bundle, device):
    members = []
    for m in bundle['members']:
        name = m['name']
        arch = m['arch']
        img_size = int(m['img_size'])
        model = build_model(arch)
        model.load_state_dict(m['state_dict'], strict=True)
        model = model.to(device, memory_format=torch.channels_last).eval()
        members.append((model, img_size))
        print(f"[INFO] loaded {name} ({arch}, {img_size}px)")
    return members

@torch.inference_mode()
def predict(bundle, paths, device, batch_size, num_workers):
    members = load_members(bundle, device)
    sizes = sorted({size for _, size in members})
    dataset = TestDataset(paths, sizes, bundle['mean'], bundle['std'])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == 'cuda')

    chunks = []
    tta = bundle['tta']
    fusion = bundle['fusion']
    print(f"[INFO] {len(paths)} images, batch {batch_size}, tta {tta}, fusion {fusion}, device {device}")
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
        for batch in loader:
            views_by_size = {}
            for size in sizes:
                imgs = batch[size].to(device, memory_format=torch.channels_last, non_blocking=True)
                views_by_size[size] = (imgs, imgs.flip((3,))) if tta else (imgs,)

            batch_logits = torch.stack([
                torch.stack([model(view).squeeze(1).float() for view in views_by_size[size]])
                for model, size in members
            ])
            chunks.append(batch_logits.cpu().numpy().mean(axis=1))

    logits = np.concatenate(chunks, axis=1)
    weights = [m['weight'] for m in bundle['members']]
    return fuse(logits, bundle['fusion'], weights)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', default='predictions.csv')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--num_workers', type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()

    start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    paths = list_images(args.test_dir)
    ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]

    bundle = load_bundle(args.weights)
    threshold = bundle['threshold'] if args.threshold is None else args.threshold
    probs = predict(bundle, paths, device, args.batch_size, args.num_workers)
    probs = np.round(np.clip(probs, 0.0, 1.0), 6)
    labels = (probs >= threshold).astype(int)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'label', 'confidence'])
        for img_id, lbl, p in zip(ids, labels, probs):
            writer.writerow([img_id, int(lbl), f'{p:.6f}'])

    elapsed = time.time() - start
    print(f"[DONE] wrote {args.output}")
    print(f"  images: {len(ids)}, predicted fake: {int(labels.sum())} ({labels.mean()*100:.1f}%), mean confidence {probs.mean():.4f}")
    print(f"  elapsed: {elapsed:.1f} s")
    if device.type == 'cuda':
        print(f"  peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")

if __name__ == '__main__':
    main()
