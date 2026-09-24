import copy
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from training.common import taxonomy, digest, write_json
from training.dataset import audit_manifest, assign_splits, crop_box, prepare_images
from training.bundle import install_bundle


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rimpang-test-")
        self.root = Path(self.temporary.name)
        self.rows = []
        for split_index, split in enumerate(("train", "valid", "test")):
            for class_id in range(10):
                name = f"{split}-{class_id}.png"
                Image.new("RGB", (64, 48), (class_id * 20, split_index * 70, 50)).save(self.root / name)
                self.rows.append({"path": name, "sourceId": "synthetic-unit-test", "groupId": f"{split}-{class_id}",
                                  "split": split, "objects": [{"classId": class_id, "bbox": [0.1, 0.1, 0.5, 0.6]}]})
        self.manifest = {"schemaVersion": 1, "labelsVersion": 1, "images": self.rows}

    def tearDown(self):
        self.temporary.cleanup()

    def test_valid_data_roundtrip_and_crop(self):
        rows, report = audit_manifest(self.manifest, self.root)
        self.assertEqual(report["images"], 30)
        audited, _ = audit_manifest({**self.manifest, "images": rows}, self.root)
        result = assign_splits(audited)
        destination = self.root / "prepared"
        prepare_images(result, self.root, destination, 0.1)
        self.assertEqual(len(list(destination.rglob("*.jpg"))), 60)
        self.assertEqual((destination / "detector" / "labels" / "train").is_dir(), True)
        self.assertEqual(crop_box([0, 0, 1, 1], (100, 80), 0.2), (0, 0, 100, 80))

    def test_empty_missing_and_traversal(self):
        for path in ("empty.jpg", "missing.jpg", "../escape.jpg", "C:\\secret.jpg"):
            if path == "empty.jpg":
                (self.root / path).touch()
            candidate = copy.deepcopy(self.manifest)
            candidate["images"][0]["path"] = path
            with self.assertRaises(ValueError):
                audit_manifest(candidate, self.root)

    def test_bad_bbox_class_and_group_leak(self):
        for modify in (
            lambda row: row["objects"][0].update(bbox=[0.9, 0, 0.5, 0.5]),
            lambda row: row["objects"][0].update(classId=10),
            lambda row: row.update(groupId="valid-0"),
        ):
            candidate = copy.deepcopy(self.manifest)
            modify(candidate["images"][0])
            with self.assertRaises(ValueError):
                audit_manifest(candidate, self.root)

    def test_partial_split_and_missing_coverage(self):
        candidate = copy.deepcopy(self.manifest)
        del candidate["images"][0]["split"]
        rows, _ = audit_manifest(candidate, self.root)
        with self.assertRaises(ValueError):
            assign_splits(rows)
        rows, _ = audit_manifest(self.manifest, self.root)
        with self.assertRaises(ValueError):
            assign_splits(rows[:-1])

    def test_automatic_split_is_reproducible_and_grouped(self):
        candidate = copy.deepcopy(self.manifest)
        for row in candidate["images"]:
            row["groupId"] = f" {row.pop('split')} "
        rows, _ = audit_manifest(candidate, self.root)
        result = assign_splits(rows, 7)
        self.assertEqual(result, assign_splits(rows, 7))
        self.assertEqual({row["split"] for row in result}, {"train", "valid", "test"})
        for group in {row["groupId"] for row in result}:
            self.assertEqual(group, group.strip())
            self.assertEqual(len({row["split"] for row in result if row["groupId"] == group}), 1)

    def test_exact_duplicates_rejected(self):
        candidate = copy.deepcopy(self.manifest)
        (self.root / "duplicate.png").write_bytes((self.root / self.rows[0]["path"]).read_bytes())
        candidate["images"][1]["path"] = "duplicate.png"
        with self.assertRaises(ValueError):
            audit_manifest(candidate, self.root)

    def test_install_validates_both_files_before_publishing(self):
        bundle, destination = self.root / "bundle", self.root / "web-models"
        bundle.mkdir()
        for key in ("detector", "classifier"):
            (bundle / f"{key}.onnx").write_bytes(f"not-an-onnx-unit-test-{key}".encode())
        artifact = lambda key: {"file": f"{key}.onnx", "sha256": digest(bundle / f"{key}.onnx"),
                                "inputName": "images", "outputName": "output0", "inputSize": 224, "dtype": "float32"}
        manifest = {
            "schemaVersion": 1, "labelsVersion": 1, "status": "ready", "version": "unit-test-only",
            "classSlugs": [row["slug"] for row in taxonomy()["labels"]], "maxObjects": 5,
            "detector": {**artifact("detector"), "outputLayout": "1x5xN", "padValue": 114, "scoreThreshold": 0.5, "iouThreshold": 0.5},
            "classifier": {**artifact("classifier"), "mean": [0.5]*3, "std": [0.5]*3, "cropMargin": 0.1, "scoreThreshold": 0.8, "minMargin": 0.1},
        }
        write_json(bundle / "manifest.json", manifest)
        installed = install_bundle(bundle, destination)
        previous = (destination / "manifest.json").read_bytes()
        self.assertTrue((destination / installed["classifier"]["file"]).is_file())
        (bundle / "classifier.onnx").write_bytes(b"corrupted")
        with self.assertRaises(ValueError):
            install_bundle(bundle, destination)
        self.assertEqual((destination / "manifest.json").read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
