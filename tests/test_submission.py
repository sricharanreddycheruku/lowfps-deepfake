"""Checks for the split, the model definitions, the inference script and the
shipped predictions. Run from the project root:

    python -m unittest discover -s tests -v

Tests that need data which is not present (for example the dataset in a
source-only copy) are skipped, not failed. Nothing is downloaded.
"""

import csv
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
TEST_DIR = ROOT / "data" / "raw" / "hackathon_dataset" / "test" / "images"
SCRIPT = ROOT / "submission" / "inference_script.py"
PREDICTIONS = ROOT / "submission" / "predictions.csv"
FINAL_ARCHS = ["convnext_small_22k", "swin_small_22k", "maxvit_t", "convnext_base"]

sys.path.insert(0, str(ROOT / "src"))


def load_inference_module():
    spec = importlib.util.spec_from_file_location("inference_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def has_timm():
    return importlib.util.find_spec("timm") is not None


@unittest.skipUnless((SPLITS / "split_v1.csv").exists(), "split files not present")
class SplitIntegrity(unittest.TestCase):
    def fold_files(self):
        return [SPLITS / "split_v1.csv"] + [SPLITS / f"split_v1_fold{k}.csv" for k in range(1, 5)]

    def test_split_v1_rows_and_labels(self):
        rows = read_csv(SPLITS / "split_v1.csv")
        self.assertEqual(len(rows), 7000)
        self.assertEqual(len({r["image_id"] for r in rows}), 7000)
        for r in rows:
            folder = Path(r["path"]).parent.name
            self.assertEqual(int(r["label"]), {"real": 0, "fake": 1}[folder], r["image_id"])

    def test_no_group_straddles_train_and_val(self):
        for path in self.fold_files():
            side = {}
            for r in read_csv(path):
                side.setdefault(r["group_id"], set()).add(r["split"])
            mixed = [g for g, s in side.items() if len(s) > 1]
            self.assertEqual(mixed, [], f"{path.name}: groups on both sides")

    def test_fold_val_sets_partition_the_data(self):
        seen = []
        for path in self.fold_files():
            seen += [r["image_id"] for r in read_csv(path) if r["split"] == "val"]
        self.assertEqual(len(seen), 7000)
        self.assertEqual(len(set(seen)), 7000)


@unittest.skipUnless(has_timm(), "timm not installed")
class ModelParity(unittest.TestCase):
    """The inference script rebuilds the models without importing src/; both
    definitions must give the same parameters, or the bundle will not load."""

    def test_state_dicts_match(self):
        from models import build_model

        inference = load_inference_module()
        torch.manual_seed(0)
        for arch in FINAL_ARCHS:
            with self.subTest(arch=arch):
                train_model = build_model(arch, pretrained=False).eval()
                infer_model = inference.build_model(arch).eval()
                a, b = train_model.state_dict(), infer_model.state_dict()
                self.assertEqual(list(a), list(b))
                for k in a:
                    self.assertEqual(a[k].shape, b[k].shape, k)
                infer_model.load_state_dict(a, strict=True)
                x = torch.randn(2, 3, 224, 224)
                with torch.no_grad():
                    out_a, out_b = train_model(x), infer_model(x)
                self.assertEqual(tuple(out_b.shape), (2, 1))
                self.assertTrue(torch.allclose(out_a, out_b, atol=1e-5))


class InferenceEndToEnd(unittest.TestCase):
    """Runs the real script on a tiny random two-member bundle."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        cls.images = tmp / "images"
        cls.images.mkdir()
        rng = np.random.default_rng(0)
        names = [f"img_{i:02d}.jpg" for i in range(6)] + ["img_png.png"]
        for name in names:
            arr = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls.images / name)
        (cls.images / "notes.txt").write_text("not an image")
        cls.ids = sorted(Path(n).stem for n in names)

        inference = load_inference_module()
        torch.manual_seed(0)
        members = []
        for i, size in enumerate((64, 96)):
            model = inference.build_model("efficientnet_b0").eval()
            members.append({"name": f"m{i}", "arch": "efficientnet_b0", "img_size": size,
                            "weight": 0.5, "state_dict": model.state_dict()})
        cls.bundle = tmp / "bundle.pt"
        torch.save({"members": members, "fusion": "mean_logit", "tta": True, "threshold": 0.5,
                    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                    "image_size": 64}, cls.bundle)
        cls.outputs = []
        for run in range(2):
            out = tmp / f"pred_{run}.csv"
            subprocess.run([sys.executable, str(SCRIPT), "--test_dir", str(cls.images),
                            "--weights", str(cls.bundle), "--output", str(out),
                            "--batch_size", "3", "--num_workers", "0"],
                           check=True, capture_output=True, cwd=tmp)
            cls.outputs.append(out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_format(self):
        text = self.outputs[0].read_text(encoding="utf-8")
        self.assertTrue(text.startswith("image_id,label,confidence\n"))
        rows = read_csv(self.outputs[0])
        self.assertEqual([r["image_id"] for r in rows], self.ids)
        for r in rows:
            p = float(r["confidence"])
            self.assertTrue(0.0 <= p <= 1.0)
            self.assertEqual(len(r["confidence"].split(".")[1]), 6)
            self.assertEqual(int(r["label"]), int(p >= 0.5))

    def test_rerun_is_identical(self):
        self.assertEqual(self.outputs[0].read_bytes(), self.outputs[1].read_bytes())

    def test_threshold_flag(self):
        out = Path(self.tmp.name) / "pred_thr.csv"
        subprocess.run([sys.executable, str(SCRIPT), "--test_dir", str(self.images),
                        "--weights", str(self.bundle), "--output", str(out),
                        "--threshold", "1.0", "--num_workers", "0"],
                       check=True, capture_output=True)
        self.assertTrue(all(r["label"] == "0" for r in read_csv(out)))

    def test_checker_accepts_output(self):
        result = subprocess.run([sys.executable, str(ROOT / "src" / "check_predictions.py"),
                                 "--pred", str(self.outputs[0]), "--test_dir", str(self.images)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(PREDICTIONS.exists(), "submission/predictions.csv not present")
class ShippedPredictions(unittest.TestCase):
    def test_columns_and_values(self):
        rows = read_csv(PREDICTIONS)
        self.assertEqual(list(rows[0]), ["image_id", "label", "confidence"])
        ids = [r["image_id"] for r in rows]
        self.assertEqual(len(ids), len(set(ids)))
        conf = np.array([float(r["confidence"]) for r in rows])
        labels = np.array([int(r["label"]) for r in rows])
        self.assertTrue(((conf >= 0) & (conf <= 1)).all())
        self.assertTrue(set(labels) <= {0, 1})
        if labels.any() and not labels.all():
            # one threshold separates the labels
            self.assertGreaterEqual(conf[labels == 1].min(), conf[labels == 0].max())

    @unittest.skipUnless(TEST_DIR.exists(), "test images not present")
    def test_one_row_per_test_image(self):
        stems = {p.stem for p in TEST_DIR.iterdir()
                 if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
        self.assertEqual({r["image_id"] for r in read_csv(PREDICTIONS)}, stems)


if __name__ == "__main__":
    unittest.main()
