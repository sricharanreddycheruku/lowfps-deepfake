"""Inference throughput and peak GPU memory per backbone.

Forward-only, fp16 autocast, channels_last, batch 128 by default, random
input (decode and resize are not measured, they are the same for every
backbone). Weights are not loaded since speed does not depend on them.

Usage:
    python src/bench.py --out outputs/ablations/bench.json
    python src/bench.py --archs efficientnet_b0 swin_t --batch 128
"""

import argparse
import json
import os
import time

import torch

from models import SUPPORTED_ARCHS, build_model


@torch.inference_mode()
def bench_arch(arch, batch, iters, warmup, device, size=224):
    model = build_model(arch, pretrained=False).to(device, memory_format=torch.channels_last)
    model.eval()
    x = torch.randn(batch, 3, size, size, device=device).contiguous(memory_format=torch.channels_last)

    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_cuda):
        for _ in range(warmup):
            model(x)
        if use_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            model(x)
        if use_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if use_cuda else float("nan")
    n_params = sum(p.numel() for p in model.parameters())
    del model, x
    return {
        "arch": arch,
        "size": size,
        "batch": batch,
        "img_per_s": round(batch * iters / elapsed, 1),
        "ms_per_batch": round(1000 * elapsed / iters, 2),
        "peak_mb": round(peak_mb, 1),
        "params_m": round(n_params / 1e6, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archs", nargs="+", default=SUPPORTED_ARCHS)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", default="outputs/ablations/bench.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"

    results = []
    for arch in args.archs:
        r = bench_arch(arch, args.batch, args.iters, args.warmup, device, args.size)
        print(r)
        results.append(r)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"device": gpu, "precision": "fp16" if device.type == "cuda" else "fp32",
                   "results": results}, f, indent=2)
    print(f"wrote {args.out} ({gpu})")


if __name__ == "__main__":
    main()
