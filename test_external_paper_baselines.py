"""Offline smoke tests for the paper-baseline patch (no GPU or network)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from external_paper_baselines import (
    build_upstream_command,
    compare_run_receipts,
    convert_obb_line_to_hbb,
    get_method,
    load_registry,
    prepare_hbb,
)


class PaperBaselineTests(unittest.TestCase):
    def test_registry_is_only_named_papers_with_official_sources(self):
        for name, row in load_registry().items():
            self.assertTrue(row["paper_url"].startswith("https://"), name)
            self.assertEqual(row["source_kind"], "authors_official")
            self.assertTrue(row["source"].startswith("https://github.com/"))
            self.assertIn(row["prediction_geometry"], ("obb", "hbb"))
            self.assertNotEqual(name, "sjpa_tr_proposal")

    def test_obb_to_hbb(self):
        self.assertEqual(
            convert_obb_line_to_hbb("1 0.2 0.3 0.6 0.3 0.6 0.7 0.2 0.7", 2),
            "1 0.40000000 0.50000000 0.40000000 0.40000000",
        )
        for malformed in ("0 0.1 0.2", "2 0.2 0.3 0.6 0.3 0.6 0.7 0.2 0.7", "0 1.2 0.3 0.6 0.3 0.6 0.7 0.2 0.7"):
            with self.assertRaises(ValueError):
                convert_obb_line_to_hbb(malformed, 2)

    def make_fixture(self, root, same_stem=False):
        import yaml
        root = Path(root) / "source"
        for split in ("train", "val"):
            stem = "same" if same_stem else split + "1"
            for kind in ("rgb", "ir"):
                file = root / kind / "images" / split / f"{stem}.png"
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b"fixture-image-bytes")
            file = root / "labels" / split / f"{stem}.txt"
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("0 0.2 0.3 0.6 0.3 0.6 0.7 0.2 0.7\n")
        conf = root / "data.yaml"
        conf.write_text(yaml.safe_dump({"path": str(root), "names": {0: "car"},
                                        "source": {"task": "obb"},
                                        "train": "rgb/images/train", "val": "rgb/images/val",
                                        "train_ir": "ir/images/train", "val_ir": "ir/images/val",
                                        "train_labels": "labels/train", "val_labels": "labels/val"}))
        return conf

    def test_hbb_convert_two_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = self.make_fixture(temp)
            out = Path(temp) / "hbb"
            report = prepare_hbb(cfg, out)
            self.assertEqual(report["num_pairs"], 2)
            self.assertEqual(report["split_counts"], {"train": 1, "val": 1})
            self.assertTrue((out / "ir/images/train/train1.png").is_file())
            self.assertIn("0.40000000", (out / "labels/train/train1.txt").read_text())
            self.assertTrue((out / "conversion_manifest.json").is_file())

    def test_leakage_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = self.make_fixture(temp, same_stem=True)
            with self.assertRaisesRegex(ValueError, "leakage"):
                prepare_hbb(cfg, Path(temp) / "output")

    def test_official_training_templates(self):
        cmd = build_upstream_command("c2former_obb", "python3", "a.py", "results")
        self.assertEqual(cmd[:3], ["python3", "tools/train.py", "a.py"])
        cmd = build_upstream_command("superyolo_hbb", "python3", "SRvedai.yaml", "results")
        self.assertIn("--super", cmd)
        with self.assertRaisesRegex(ValueError, "No independently confirmed"):
            build_upstream_command("icafusion_hbb", "python3", "foo", "results")

    def test_receipts_reject_mixed_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "weights.pt").write_bytes(b"w")
            (root / "predictions.json").write_text("{}")
            shared = {"status": "completed", "source_kind": "measured",
                      "dataset_sha256": "data", "split_sha256": "split", "evaluator": "common",
                      "seed": 0, "mAP50": 0.7, "mAP50_95": 0.4,
                      "prediction_log": str(root / "predictions.json"),
                      "checkpoint": str(root / "weights.pt")}
            a = root / "a.json"
            b = root / "b.json"
            a.write_text(json.dumps({**shared, "method": "sjpa_tr_proposal", "geometry": "obb"}))
            b.write_text(json.dumps({**shared, "method": "superyolo_hbb", "geometry": "hbb"}))
            with self.assertRaisesRegex(ValueError, "Cannot combine"):
                compare_run_receipts([a, b], root / "compare.csv")
            b.write_text(json.dumps({**shared, "method": "c2former_obb", "geometry": "obb"}))
            report = compare_run_receipts([a, b], root / "compare.csv")
            self.assertEqual(report["rows"], 2)
            self.assertTrue((root / "compare.csv").is_file())


if __name__ == "__main__":
    unittest.main()
