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


class StaticWindowAttention(torch.nn.Module):
    """Swin window attention for one fixed input size and batch.

    The relative position bias and the shifted-window mask depend only on the
    resolution, so they are computed once and stored as constants. The forward
    pass is plain reshape / transpose / matmul / softmax (the roll is a slice and
    a concat), which onnx2tf converts reliably. Same function as the torchvision
    module in eval mode (checked by check_swin_rewrite).
    """

    def __init__(self, attn, batch, height, width):
        super().__init__()
        self.batch, self.height, self.width = batch, height, width
        self.heads = attn.num_heads
        self.win = list(attn.window_size)
        shift = list(attn.shift_size)
        # torchvision drops the shift when the window covers the whole map
        self.shift = [0 if self.win[0] >= height else shift[0],
                      0 if self.win[1] >= width else shift[1]]
        self.qkv, self.proj = attn.qkv, attn.proj
        n = self.win[0] * self.win[1]
        self.n_tokens = n
        self.n_win = (height // self.win[0]) * (width // self.win[1])
        with torch.no_grad():
            bias = attn.get_relative_position_bias().detach().reshape(1, self.heads, n, n)
        self.register_buffer("bias", bias)
        if sum(self.shift) > 0:
            self.register_buffer("mask", self.make_mask())
        else:
            self.mask = None

    def make_mask(self):
        h, w = self.height, self.width
        wh, ww = self.win
        sh, sw = self.shift
        m = torch.zeros(h, w)
        count = 0
        for hs in ((0, -wh), (-wh, -sh), (-sh, None)):
            for ws in ((0, -ww), (-ww, -sw), (-sw, None)):
                m[hs[0]:hs[1], ws[0]:ws[1]] = count
                count += 1
        m = m.view(h // wh, wh, w // ww, ww).permute(0, 2, 1, 3).reshape(self.n_win, wh * ww)
        diff = m.unsqueeze(1) - m.unsqueeze(2)
        diff = diff.masked_fill(diff != 0, -100.0).masked_fill(diff == 0, 0.0)
        return diff.reshape(self.n_win, 1, wh * ww, wh * ww)

    @staticmethod
    def roll(x, shift, dim):
        """torch.roll along one dim as slice + concat (the shift may be negative)."""
        k = shift % x.shape[dim]
        if k == 0:
            return x
        head = x.narrow(dim, x.shape[dim] - k, k)
        tail = x.narrow(dim, 0, x.shape[dim] - k)
        return torch.cat([head, tail], dim=dim)

    def forward(self, x):
        b, h, w, c = self.batch, self.height, self.width, x.shape[-1]
        wh, ww = self.win
        if self.shift[0] or self.shift[1]:
            x = self.roll(self.roll(x, -self.shift[0], 1), -self.shift[1], 2)
        # windows are ordered window-major (window, batch) so that the mask below
        # is a 4-D broadcast, which onnx2tf handles; a 5-D one it does not
        x = x.reshape(b, h // wh, wh, w // ww, ww, c).permute(1, 3, 0, 2, 4, 5)
        x = x.reshape(self.n_win * b, self.n_tokens, c)
        qkv = self.qkv(x).reshape(self.n_win * b, self.n_tokens, 3, self.heads, c // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * (c // self.heads) ** -0.5, qkv[1], qkv[2]
        attn = q.matmul(k.transpose(-2, -1)) + self.bias
        if self.mask is not None:
            attn = attn.reshape(self.n_win, b * self.heads, self.n_tokens, self.n_tokens) + self.mask
            attn = attn.reshape(self.n_win * b, self.heads, self.n_tokens, self.n_tokens)
        attn = torch.softmax(attn, dim=-1)
        x = attn.matmul(v).transpose(1, 2).reshape(self.n_win * b, self.n_tokens, c)
        x = self.proj(x)
        x = x.reshape(h // wh, w // ww, b, wh, ww, c).permute(2, 0, 3, 1, 4, 5).reshape(b, h, w, c)
        if self.shift[0] or self.shift[1]:
            x = self.roll(self.roll(x, self.shift[0], 1), self.shift[1], 2)
        return x


def rewrite_swin(model, size, batch):
    """Replaces every window attention of a torchvision Swin by StaticWindowAttention."""
    from torchvision.models.swin_transformer import ShiftedWindowAttention
    attns = [m for m in model.modules() if isinstance(m, ShiftedWindowAttention)]
    shapes = {}
    hooks = [m.register_forward_pre_hook(lambda mod, inp: shapes.__setitem__(mod, inp[0].shape[1:3]))
             for m in attns]
    with torch.no_grad():
        model(torch.zeros(1, 3, size, size))
    for hook in hooks:
        hook.remove()
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if child in shapes:
                height, width = shapes[child]
                setattr(parent, name, StaticWindowAttention(child, batch, int(height), int(width)))


def check_swin_rewrite(model, size, batch):
    """Returns (max abs logit diff to the torchvision model, the rewritten copy)."""
    import copy
    static = copy.deepcopy(model)
    rewrite_swin(static, size, batch)
    x = torch.randn(batch, 3, size, size)
    with torch.no_grad():
        return float((model(x) - static(x)).abs().max()), static


def build_swin_savedmodel(model, size, batch, out_dir):
    """Writes a TensorFlow SavedModel of a (rewritten) torchvision Swin.

    onnx2tf cannot convert the window attention of Swin (its layout heuristics
    fail on the 4-D attention tensors), so the network is rebuilt directly in
    TensorFlow ops from the PyTorch weights and the constants of
    StaticWindowAttention. Input: (batch, size, size, 3) float32, already
    normalised; output: (batch, 1) logits.
    """
    import tensorflow as tf
    from torchvision.models.swin_transformer import PatchMerging, SwinTransformerBlock

    def const(t):
        return tf.constant(t.detach().cpu().numpy())

    def layer_norm(x, ln):
        g, b = const(ln.weight), const(ln.bias)
        mu = tf.reduce_mean(x, -1, keepdims=True)
        var = tf.reduce_mean(tf.square(x - mu), -1, keepdims=True)
        return (x - mu) * tf.math.rsqrt(var + ln.eps) * g + b

    def linear(x, lin):
        y = tf.matmul(x, const(lin.weight.t()))
        return y + const(lin.bias) if lin.bias is not None else y

    def attention(x, a):
        h, w, c, n, nw, heads = a.height, a.width, x.shape[-1], a.n_tokens, a.n_win, a.heads
        wh, ww = a.win
        if a.shift[0] or a.shift[1]:
            x = tf.roll(x, shift=[-a.shift[0], -a.shift[1]], axis=[1, 2])
        x = tf.reshape(x, [batch, h // wh, wh, w // ww, ww, c])
        x = tf.reshape(tf.transpose(x, [1, 3, 0, 2, 4, 5]), [nw * batch, n, c])
        qkv = tf.transpose(tf.reshape(linear(x, a.qkv), [nw * batch, n, 3, heads, c // heads]),
                           [2, 0, 3, 1, 4])
        q, k, v = qkv[0] * (c // heads) ** -0.5, qkv[1], qkv[2]
        att = tf.matmul(q, k, transpose_b=True) + const(a.bias)
        if a.mask is not None:
            att = tf.reshape(att, [nw, batch * heads, n, n]) + const(a.mask)
            att = tf.reshape(att, [nw * batch, heads, n, n])
        x = tf.matmul(tf.nn.softmax(att, axis=-1), v)
        x = linear(tf.reshape(tf.transpose(x, [0, 2, 1, 3]), [nw * batch, n, c]), a.proj)
        x = tf.reshape(x, [h // wh, w // ww, batch, wh, ww, c])
        x = tf.reshape(tf.transpose(x, [2, 0, 3, 1, 4, 5]), [batch, h, w, c])
        if a.shift[0] or a.shift[1]:
            x = tf.roll(x, shift=[a.shift[0], a.shift[1]], axis=[1, 2])
        return x

    def forward(images):
        stem = model.features[0]
        x = tf.nn.conv2d(images, const(stem[0].weight.permute(2, 3, 1, 0)), strides=4, padding="VALID")
        x = layer_norm(x + const(stem[0].bias), stem[2])
        for stage in list(model.features)[1:]:
            if isinstance(stage, PatchMerging):
                x = tf.concat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
                x = linear(layer_norm(x, stage.norm), stage.reduction)
                continue
            for block in stage:
                assert isinstance(block, SwinTransformerBlock)
                x = x + attention(layer_norm(x, block.norm1), block.attn)
                y = linear(layer_norm(x, block.norm2), block.mlp[0])
                x = x + linear(tf.nn.gelu(y, approximate=False), block.mlp[3])
        x = tf.reduce_mean(layer_norm(x, model.norm), axis=[1, 2])
        return linear(x, model.head[-1])

    module = tf.Module()
    module.serve = tf.function(
        lambda images: {"logits": forward(images)},
        input_signature=[tf.TensorSpec([batch, size, size, 3], tf.float32, name="images")])
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    tf.saved_model.save(module, out_dir, signatures={"serving_default": module.serve})


def export_onnx(model, size, path, static_batch=None):
    """Dynamic batch axis by default; a fixed batch when static_batch is given."""
    dummy = torch.randn(static_batch or 2, 3, size, size)
    axes = None if static_batch else {"images": {0: "batch"}, "logits": {0: "batch"}}
    torch.onnx.export(
        model, dummy, path, opset_version=OPSET, dynamo=False,
        input_names=["images"], output_names=["logits"], dynamic_axes=axes,
    )


def check_onnx(model, size, path, batch=3):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    # for a dynamic export a batch different from the trace batch checks the axis
    x = torch.randn(batch, 3, size, size)
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
    archs = [m["arch"] for m in bundle["members"]]
    for m in bundle["members"]:
        arch, size = m["arch"], int(m["img_size"])
        if args.only and arch not in args.only:
            continue
        # seeds of one arch need their own directory
        tag = m["name"] if archs.count(arch) > 1 else arch
        print(f"[{arch}] exporting to ONNX", flush=True)
        try:
            model = build_model(arch, pretrained=False)
            model.load_state_dict(m["state_dict"])
            model.eval()
            rewrite_gelu(model)
            tf_dir = os.path.join(args.out, tag)
            static_batch = None
            onnx_path = os.path.join(args.work, f"{tag}.onnx")
            if arch.startswith("swin"):
                # fixed batch, rebuilt in TensorFlow ops (see build_swin_savedmodel)
                static_batch = args.static_batch
                rewrite_diff, model = check_swin_rewrite(model, size, static_batch)
                print(f"[{arch}] static rewrite vs torchvision max abs logit diff {rewrite_diff:.2e}",
                      flush=True)
                build_swin_savedmodel(model, size, static_batch, tf_dir)
                onnx_diff, error = None, None
            else:
                export_onnx(model, size, onnx_path)
                onnx_diff = check_onnx(model, size, onnx_path)
                print(f"[{arch}] onnx vs torch max abs logit diff {onnx_diff:.2e}", flush=True)
                error = convert_to_tf(onnx_path, tf_dir, args.work)
            if error:
                raise RuntimeError("onnx2tf failed:\n" + error)
            in_name, out_name = signature_names(tf_dir)
            meta["members"].append({
                "name": m["name"], "arch": arch, "img_size": size, "weight": float(m["weight"]),
                "dir": tag, "batch": static_batch, "input": in_name, "output": out_name,
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
                        help="fixed batch of the Swin SavedModel (the TF script pads the last batch)")
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
