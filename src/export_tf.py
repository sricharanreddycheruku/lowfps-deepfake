"""Exports the bundled ensemble members to TensorFlow SavedModels.

Each member of submission/model_weights.pt is traced to ONNX (opset 17, dynamic
batch axis) and converted with onnx2tf. The result is a directory

    <out>/meta.json              fusion rule, TTA flag, threshold, mean/std and
                                 the members that converted (name, arch, size,
                                 weight, subdirectory, input/output names)
    <out>/<arch>/                one SavedModel per converted member

onnx2tf turns the NCHW input into NHWC, so the SavedModels take (N, H, W, C)
float32 images that are already normalised. A member that fails to export or
convert is skipped, recorded under "failed" in meta.json, and left out of the
fusion (the remaining weights are renormalised), so a partial export still
yields a working TF ensemble.

The ONNX graph of every member is checked against the PyTorch model with
onnxruntime before conversion. The TF-vs-PyTorch parity on validation images
is a separate step (--parity), which runs the TF pipeline and the PyTorch
pipeline in fp32 on the same preprocessed batches and writes a json report.

Needs the packages in submission/requirements_tf.txt plus torch/torchvision
(only for the export and parity steps, not for inference_script_tf.py).

Usage:
    python src/export_tf.py --weights submission/model_weights.pt --out tf_model \
        --work work_tf
    python src/export_tf.py --parity --weights submission/model_weights.pt \
        --out tf_model --split_csv data/splits/split_v1.csv --n 500 \
        --parity_json outputs/eval/tf_parity.json
"""

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

from models import build_model

OPSET = 17


def load_bundle(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


class ExactGelu(torch.nn.Module):
    """GELU written as x * 0.5 * (1 + erf(x * 2^-0.5)). Same function as
    nn.GELU, but the constants are one-element tensors instead of Python
    scalars, which is the form onnx2tf converts without trouble."""

    def __init__(self):
        super().__init__()
        self.register_buffer("c_half", torch.tensor([0.5]))
        self.register_buffer("c_one", torch.tensor([1.0]))
        self.register_buffer("inv_sqrt2", torch.tensor([0.7071067811865476]))

    def forward(self, x):
        return x * self.c_half * (self.c_one + torch.erf(x * self.inv_sqrt2))


def rewrite_gelu(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.GELU) and child.approximate == "none":
            setattr(module, name, ExactGelu())
        else:
            rewrite_gelu(child)


def export_onnx(model, size, path, static_batch=None):
    """Dynamic batch axis by default; a fixed batch when static_batch is given."""
    dummy = torch.randn(static_batch or 2, 3, size, size)
    axes = None if static_batch else {"images": {0: "batch"}, "logits": {0: "batch"}}
    torch.onnx.export(
        model, dummy, path, opset_version=OPSET, dynamo=False,
        input_names=["images"], output_names=["logits"], dynamic_axes=axes,
    )


def check_onnx(model, size, path):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    # a batch size different from the trace batch checks the dynamic axis
    x = torch.randn(3, 3, size, size)
    with torch.no_grad():
        ref = model(x).numpy()
    got = sess.run(None, {"images": x.numpy()})[0]
    return float(np.abs(ref - got).max())


def convert_to_tf(onnx_path, out_dir, work, prf=None):
    """Runs onnx2tf. Returns None on success or the tail of its log on failure.

    -nuo skips the onnxsim pass (on ConvNeXt it leaves a Python constant in the
    graph that Keras rejects, on Swin it runs for many minutes). onnx2tf looks
    for its validation sample in the working directory and otherwise downloads
    it, so it runs from `work`, where a stand-in array is placed (it is only used
    for onnx2tf's own shape checks, never for our parity numbers).
    """
    out_dir = os.path.abspath(out_dir)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    sample = os.path.join(work, "calibration_image_sample_data_20x128x128x3_float32.npy")
    if not os.path.exists(sample):
        np.save(sample, np.random.RandomState(0).rand(20, 128, 128, 3).astype(np.float32))
    cmd = [sys.executable, "-m", "onnx2tf", "-i", os.path.abspath(onnx_path), "-o", out_dir,
           "-osd", "-nuo"]
    if prf:
        cmd += ["-prf", os.path.abspath(prf)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=work)
    if result.returncode == 0 and os.path.exists(os.path.join(out_dir, "saved_model.pb")):
        # keep only the SavedModel, drop the tflite files onnx2tf also writes
        for path in glob.glob(os.path.join(out_dir, "*.tflite")):
            os.remove(path)
        return None
    # onnx2tf proposes a parameter replacement file for the op it failed on; keep it
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    auto = os.path.join(out_dir, f"{stem}_auto.json")
    if os.path.exists(auto):
        shutil.copy(auto, os.path.join(work, f"{stem}_auto.json"))
    return (result.stdout + result.stderr)[-1500:]


def signature_names(saved_model_dir):
    import tensorflow as tf
    loaded = tf.saved_model.load(saved_model_dir)
    sig = loaded.signatures["serving_default"]
    inputs = list(sig.structured_input_signature[1].keys())
    outputs = list(sig.structured_outputs.keys())
    return inputs[0], outputs[0]


def run_export(args):
    bundle = load_bundle(args.weights)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.work, exist_ok=True)
    meta = {
        "fusion": bundle["fusion"], "tta": bool(bundle["tta"]),
        "threshold": float(bundle["threshold"]),
        "mean": list(bundle["mean"]), "std": list(bundle["std"]),
        "layout": "NHWC", "members": [], "failed": [],
    }
    meta_path = os.path.join(args.out, "meta.json")
    if args.only and os.path.exists(meta_path):
        # re-export a subset and keep the entries of the others
        with open(meta_path, encoding="utf-8") as f:
            old = json.load(f)
        meta["members"] = [x for x in old["members"] if x["arch"] not in args.only]
        meta["failed"] = [x for x in old["failed"] if x["arch"] not in args.only]
    for m in bundle["members"]:
        arch, size = m["arch"], int(m["img_size"])
        if args.only and arch not in args.only:
            continue
        print(f"[{arch}] exporting to ONNX", flush=True)
        try:
            model = build_model(arch, pretrained=False)
            model.load_state_dict(m["state_dict"])
            model.eval()
            rewrite_gelu(model)
            tf_dir = os.path.join(args.out, arch)
            static_batch = None
            onnx_path = os.path.join(args.work, f"{arch}.onnx")
            export_onnx(model, size, onnx_path)
            onnx_diff = check_onnx(model, size, onnx_path)
            print(f"[{arch}] onnx vs torch max abs logit diff {onnx_diff:.2e}", flush=True)
            error = convert_to_tf(onnx_path, tf_dir, args.work)
            if error:
                # second try: fixed batch, which onnx2tf handles for attention blocks
                print(f"[{arch}] dynamic batch failed, retrying with batch {args.static_batch}",
                      flush=True)
                static_batch = args.static_batch
                onnx_path = os.path.join(args.work, f"{arch}_b{static_batch}.onnx")
                export_onnx(model, size, onnx_path, static_batch)
                error = convert_to_tf(onnx_path, tf_dir, args.work)
                auto = os.path.join(args.work, f"{arch}_b{static_batch}_auto.json")
                if error and os.path.exists(auto):
                    print(f"[{arch}] retrying with the parameter replacement file", flush=True)
                    error = convert_to_tf(onnx_path, tf_dir, args.work, prf=auto)
            if error:
                raise RuntimeError("onnx2tf failed:\n" + error)
            in_name, out_name = signature_names(tf_dir)
            meta["members"].append({
                "name": m["name"], "arch": arch, "img_size": size, "weight": float(m["weight"]),
                "dir": arch, "batch": static_batch, "input": in_name, "output": out_name,
                "onnx_max_abs_logit_diff": onnx_diff,
            })
            print(f"[{arch}] converted ({in_name} -> {out_name})", flush=True)
        except Exception as exc:  # keep going: a partial export is still usable
            reason = str(exc).strip().splitlines()
            meta["failed"].append({"name": m["name"], "arch": arch,
                                   "reason": " | ".join(reason[-3:])[:600]})
            print(f"[{arch}] FAILED: {reason[-1] if reason else exc}", flush=True)
    # fusion weights of the converted members, renormalised to sum to one
    raw = {m["name"]: float(m["weight"]) for m in bundle["members"]}
    total = sum(raw[x["name"]] for x in meta["members"])
    for x in meta["members"]:
        x["weight"] = raw[x["name"]] / total if total else 0.0
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {args.out}/meta.json: {len(meta['members'])} converted, "
          f"{len(meta['failed'])} failed")


def fuse(logits, rule, weights):
    if rule == "mean_logit":
        z = logits.mean(axis=0)
    elif rule == "weighted_logit":
        z = (np.asarray(weights)[:, None] * logits).sum(axis=0)
    elif rule == "mean_prob":
        return (1 / (1 + np.exp(-logits))).mean(axis=0)
    else:
        raise ValueError(rule)
    return 1 / (1 + np.exp(-z))


def run_tf(infer, member, nhwc):
    """One SavedModel call; members exported with a fixed batch get padded chunks."""
    import tensorflow as tf
    fixed = member.get("batch")
    if not fixed:
        return infer(**{member["input"]: tf.constant(nhwc)})[member["output"]].numpy().reshape(-1)
    out = []
    for start in range(0, len(nhwc), fixed):
        chunk = nhwc[start:start + fixed]
        pad = fixed - len(chunk)
        if pad:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], pad, axis=0)])
        res = infer(**{member["input"]: tf.constant(chunk)})[member["output"]].numpy().reshape(-1)
        out.append(res[:fixed - pad])
    return np.concatenate(out)


def run_parity(args):
    import tensorflow as tf
    from PIL import Image
    from sklearn.metrics import roc_auc_score

    from augment import build_eval_transform

    with open(os.path.join(args.out, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    bundle = load_bundle(args.weights)
    by_name = {m["name"]: m for m in bundle["members"]}

    with open(args.split_csv, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "val"]
    # fixed-seed shuffle so the subset holds both classes
    np.random.RandomState(0).shuffle(rows)
    rows = rows[: args.n]
    labels = np.array([int(r["label"]) for r in rows])

    tf_logits, pt_logits, used = [], [], []
    for m in meta["members"]:
        size = m["img_size"]
        tf_model = tf.saved_model.load(os.path.join(args.out, m["dir"]))
        infer = tf_model.signatures["serving_default"]
        pt_model = build_model(m["arch"], pretrained=False)
        pt_model.load_state_dict(by_name[m["name"]]["state_dict"])
        pt_model.eval()
        transform = build_eval_transform(size)
        tf_views, pt_views = [], []
        flips = (False, True) if meta["tta"] else (False,)
        for flip in flips:
            tf_out, pt_out = [], []
            for start in range(0, len(rows), args.batch):
                chunk = rows[start:start + args.batch]
                x = torch.stack([transform(Image.open(r["path"]).convert("RGB")) for r in chunk])
                if flip:
                    x = torch.flip(x, dims=[3])
                with torch.no_grad():
                    pt_out.append(pt_model(x).squeeze(1).numpy())
                nhwc = x.permute(0, 2, 3, 1).contiguous().numpy()
                tf_out.append(run_tf(infer, m, nhwc))
            tf_views.append(np.concatenate(tf_out))
            pt_views.append(np.concatenate(pt_out))
        tf_logits.append(np.mean(tf_views, axis=0))
        pt_logits.append(np.mean(pt_views, axis=0))
        used.append(m["name"])
        diff = np.abs(tf_logits[-1] - pt_logits[-1]).max()
        print(f"[{m['arch']}] max abs logit diff {diff:.2e}", flush=True)

    tf_logits, pt_logits = np.stack(tf_logits), np.stack(pt_logits)
    weights = [m["weight"] for m in meta["members"]]
    p_tf = fuse(tf_logits, meta["fusion"], weights)
    p_pt = fuse(pt_logits, meta["fusion"], weights)
    report = {
        "n_images": len(rows), "members": used, "failed_members": meta["failed"],
        "fusion": meta["fusion"], "tta": meta["tta"],
        "reference": "PyTorch fp32 on CPU, same preprocessing and fusion",
        "max_abs_diff_confidence": float(np.abs(p_tf - p_pt).max()),
        "mean_abs_diff_confidence": float(np.abs(p_tf - p_pt).mean()),
        "max_abs_diff_logit_per_member": {
            n: float(np.abs(a - b).max()) for n, a, b in zip(used, tf_logits, pt_logits)},
        "auc_tf": round(float(roc_auc_score(labels, p_tf)), 6),
        "auc_pytorch": round(float(roc_auc_score(labels, p_pt)), 6),
        "label_flips_at_threshold": int(((p_tf >= meta["threshold"]) != (p_pt >= meta["threshold"])).sum()),
    }
    report["auc_equal_to_3_decimals"] = round(report["auc_tf"], 3) == round(report["auc_pytorch"], 3)
    report["pass"] = bool(report["max_abs_diff_confidence"] < 1e-3 and report["auc_equal_to_3_decimals"])

    if args.shipped_preds and os.path.exists(args.shipped_preds):
        # the fp16 GPU predictions produced for the submission, same first n val rows
        with open(args.shipped_preds, newline="", encoding="utf-8") as f:
            shipped = {r["image_id"]: float(r["confidence"]) for r in csv.DictReader(f)}
        if all(r["image_id"] in shipped for r in rows) and len(used) == len(bundle["members"]):
            ref = np.array([shipped[r["image_id"]] for r in rows])
            report["max_abs_diff_vs_shipped_gpu_fp16"] = float(np.abs(p_tf - ref).max())
            report["auc_shipped_gpu_fp16"] = round(float(roc_auc_score(labels, ref)), 6)

    os.makedirs(os.path.dirname(os.path.abspath(args.parity_json)), exist_ok=True)
    with open(args.parity_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Export bundle members to TF SavedModels")
    parser.add_argument("--weights", default="submission/model_weights.pt")
    parser.add_argument("--out", default="tf_model", help="output directory for the SavedModels")
    parser.add_argument("--work", default="work_tf", help="scratch directory for ONNX files")
    parser.add_argument("--static_batch", type=int, default=16,
                        help="fixed batch used when a member does not convert with a dynamic one")
    parser.add_argument("--only", nargs="*", default=None, help="export only these archs")
    parser.add_argument("--parity", action="store_true", help="compare TF and PyTorch instead of exporting")
    parser.add_argument("--split_csv", default="data/splits/split_v1.csv")
    parser.add_argument("--n", type=int, default=500, help="validation images for the parity check")
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--parity_json", default="outputs/eval/tf_parity.json")
    parser.add_argument("--shipped_preds", default="outputs/ablations/ensemble_val_preds.csv")
    args = parser.parse_args()
    if args.parity:
        run_parity(args)
    else:
        run_export(args)


if __name__ == "__main__":
    main()
