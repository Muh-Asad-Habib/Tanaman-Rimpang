import copy
import hashlib
import io
import math
import shutil
import subprocess
import sys
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from training.common import ROOT, digest, read_json, taxonomy, write_json
from training.evaluation import (
    DEFAULT_GRID, PROTOCOL_VERSION, Detection, FramePrediction, GroundTruth,
    atomic_json, atomic_write, calibrate_validation, checked_thresholds, classify,
    checkpoint_eligibility, decode_detector, detector_checkpoint_binding,
    evaluate_frames, image_tensor, inspect_manifest, json_fingerprint,
    letterbox_geometry, load_predictions, match_detections, needed_anchors,
    non_max_suppression, padded_crop, predictions_document, prepared_dataset_binding,
    protocol, verify_calibration_evidence,
)
from training.scripts import evaluate_models
from training.scripts.train_classifier import validate_resume
from training.scripts.train_detector import checkpoint_receipt


SLUGS = [label["slug"] for label in taxonomy()["labels"]]
BOX = (0.1, 0.1, 0.25, 0.25)
THRESHOLDS = {
    "detectorScoreThreshold": 0.25, "iouThreshold": 0.45,
    "classifierScoreThreshold": 0.65, "minMargin": 0.1,
}


def logits(target, confidence=None):
    if confidence is None:
        return tuple(12.0 if index == target else 0.0 for index in range(10))
    return tuple(math.log(confidence if index == target else (1 - confidence) / 9) for index in range(10))


def frame(target=0, *, split="valid", detected=True, predicted=None, name=None):
    name = name or f"{split}-{target}.png"
    truth = (GroundTruth(BOX, target, logits(target)),)
    detections = (Detection(BOX, 0.9, 0, logits(target if predicted is None else predicted)),) if detected else ()
    return FramePrediction(name, name, split, truth, detections, len(detections))


def master_manifest(complete=True):
    rows = []
    for split in ("train", "valid", "test"):
        counts = [1] * 10 + ([0, 2, 3, 4, 5] if complete and split != "train" else [])
        for index, count in enumerate(counts):
            name = f"{split}-{index}.png"
            objects = [
                {"classId": index if index < 10 else item % 10,
                 "bbox": [0.02 + item * 0.18, 0.1, 0.15, 0.25]}
                for item in range(count)
            ]
            rows.append({
                "path": name, "groupId": name, "sourceId": "synthetic-unit-test-only", "split": split,
                "sha256": hashlib.sha256(name.encode()).hexdigest(), "objects": objects,
            })
    return {"schemaVersion": 1, "labelsVersion": 1, "images": rows}


def fixture_predictions(manifest, split):
    frames = []
    for row in manifest["images"]:
        if row["split"] != split:
            continue
        truth = tuple(GroundTruth(tuple(obj["bbox"]), obj["classId"], logits(obj["classId"])) for obj in row["objects"])
        detections = tuple(Detection(obj.box, 0.9, index, obj.logits) for index, obj in enumerate(truth))
        frames.append(FramePrediction(row["path"], row["groupId"], split, truth, detections, len(detections)))
    return tuple(frames)


class ProjectFilesTest(unittest.TestCase):
    def setUp(self):
        self.folder = Path(f".evaluation-unit-{uuid.uuid4().hex}")
        self.folder.mkdir()
        self.addCleanup(shutil.rmtree, self.folder)


class GeometryAndMetricsTests(unittest.TestCase):
    def test_positive_math_round_not_bankers_round(self):
        self.assertEqual(letterbox_geometry(8, 5, 4),
                         {"resizedWidth": 4, "resizedHeight": 3, "left": 0, "top": 0})
        self.assertEqual(letterbox_geometry(800, 400)["top"], 104)
        for values in ((0, 2, 416), (2, -1, 416), (2, 2, True)):
            with self.assertRaises(ValueError):
                letterbox_geometry(*values)

    def test_crop_uses_normalized_floor_ceil_and_clamps(self):
        self.assertEqual(padded_crop((0, 0, 1, 1), 80, 60), (0, 0, 80, 60))
        self.assertEqual(padded_crop((0.123, 0.25, 0.31, 0.4), 100, 80), (9, 16, 47, 56))
        with self.assertRaises(ValueError):
            padded_crop((0.9, 0.1, 0.3, 0.4), 100, 80)

    def test_decode_channel_layout_letterbox_and_type(self):
        values = np.array([[[208], [208], [208], [104], [0.9]]], dtype=np.float32)
        detections, count = decode_detector(values, letterbox_geometry(800, 400), 0.25)
        self.assertEqual(count, 1)
        self.assertEqual(detections[0].box, (0.25, 0.25, 0.5, 0.5))
        for broken in (values.transpose(0, 2, 1), values.astype(np.float64),
                       np.zeros((1, 5, 0), np.float32), values * np.nan):
            with self.assertRaises(ValueError):
                decode_detector(broken, letterbox_geometry(800, 400), 0.25)
        values[0, 4, 0] = 1.1
        with self.assertRaises(ValueError):
            decode_detector(values, letterbox_geometry(800, 400), 0.25)

    def test_image_tensor_rgb_normalization_and_padding(self):
        from PIL import Image
        image = Image.new("RGB", (8, 4), (255, 128, 0))
        tensor, geometry = image_tensor(image, 8, detector=True)
        self.assertEqual(tensor.shape, (1, 3, 8, 8))
        self.assertEqual(tensor.dtype, np.float32)
        np.testing.assert_allclose(tensor[0, :, 0, 0], [114 / 255] * 3)
        np.testing.assert_allclose(tensor[0, :, geometry["top"], 0], [1, 128 / 255, 0])
        tensor, _ = image_tensor(image, 4, box=(0, 0, 1, 1))
        np.testing.assert_allclose(tensor[0, :, 0, 0], [1, (128 / 255 - 0.5) / 0.5, -1])

    def test_nms_strict_iou_stable_ties_and_candidate_limit(self):
        same = tuple(Detection(BOX, 0.9, index) for index in range(300))
        other = Detection((0.6, 0.6, 0.1, 0.1), 0.8, 300)
        self.assertEqual([item.anchor for item in non_max_suppression(same + (other,), 0.25, 0.45)], [0])
        self.assertEqual(len(non_max_suppression(same, 0.25, 1.0)), 30)
        self.assertEqual(non_max_suppression(same, 0.95, 0.45), ())

    def test_perfect_10_class_scores(self):
        result = evaluate_frames([frame(index) for index in range(10)], THRESHOLDS)
        self.assertEqual(result["endToEnd"]["macroF1"], 1)
        self.assertEqual(result["classifierGroundTruthCrops"]["unconditional"]["macroF1"], 1)
        self.assertEqual(result["detectorAfterObjectCap"]["recall"], 1)
        self.assertEqual(result["endToEnd"]["acceptedGtObjectCoverage"], 1)
        self.assertIsNone(result["negativeOrUnknown"]["frameFalseAcceptRate"])
        self.assertFalse(result["negativeOrUnknown"]["measured"])
        self.assertIsNone(result["multiobject"]["2"]["acceptedCorrectCoverage"])

    def test_negative_false_accept_is_not_a_new_classifier_class(self):
        negative = FramePrediction("negative.png", "negative", "valid", (),
                                   (Detection(BOX, 0.9, 0, logits(0)),), 1)
        result = evaluate_frames([*[frame(index) for index in range(10)], negative], THRESHOLDS)
        self.assertAlmostEqual(result["endToEnd"]["macroF1"], (9 + 2 / 3) / 10)
        self.assertEqual(result["negativeOrUnknown"]["frameFalseAcceptRate"], 1)
        self.assertEqual(result["negativeOrUnknown"]["acceptedBoxes"], 1)
        self.assertEqual(len(result["endToEnd"]["perClass"]), 10)
        self.assertEqual(result["endToEnd"]["confusion"]["counts"][10][0], 1)
        self.assertEqual(result["classifierMatchedPredictedCrops"]["unconditional"]["macroF1"], 1)

    def test_wrong_predicted_crop_missing_gt_and_rejection_are_separate(self):
        uncertain = FramePrediction("uncertain.png", "uncertain", "valid",
                                    (GroundTruth(BOX, 2, logits(2)),),
                                    (Detection(BOX, 0.9, 0, logits(2, 0.6)),), 1)
        result = evaluate_frames([frame(0, predicted=1), frame(1, detected=False), uncertain], THRESHOLDS)
        matrix = result["endToEnd"]["confusion"]["counts"]
        self.assertEqual(matrix[0][1], 1)
        self.assertEqual(matrix[1][11], 1)
        self.assertEqual(matrix[2][10], 1)
        self.assertEqual(result["endToEnd"]["macroF1"], 0)
        self.assertEqual(result["classifierGroundTruthCrops"]["unconditional"]["macroF1"], 0.3)
        self.assertEqual(result["classifierMatchedPredictedCrops"]["crops"], 2)
        self.assertEqual(result["endToEnd"]["acceptedGtObjectCoverage"], 1 / 3)

    def test_duplicate_boxes_cannot_match_the_same_instance(self):
        truth = (GroundTruth(BOX, 0, logits(0)),)
        detections = (Detection(BOX, 0.9, 0, logits(0)), Detection(BOX, 0.8, 1, logits(0)))
        self.assertEqual(match_detections(detections, truth), {0: 0})
        item = FramePrediction("duplicate.png", "duplicate", "valid", truth, detections, 2)
        result = evaluate_frames([item], {**THRESHOLDS, "iouThreshold": 1.0})
        self.assertAlmostEqual(result["endToEnd"]["perClass"][SLUGS[0]]["f1"], 2 / 3)
        self.assertEqual(result["detectorAfterObjectCap"]["falsePositive"], 1)

    def test_five_object_cap_is_applied_before_classification(self):
        truth = tuple(GroundTruth((index * 0.15, 0.1, 0.1, 0.1), index, logits(index)) for index in range(6))
        detections = tuple(Detection(obj.box, 0.9 - index * 0.01, index,
                                    obj.logits if index < 5 else None) for index, obj in enumerate(truth))
        item = FramePrediction("six.png", "six", "valid", truth, detections, 6)
        result = evaluate_frames([item], THRESHOLDS)
        self.assertEqual(result["detectorAfterNms"]["recall"], 1)
        self.assertEqual(result["detectorAfterObjectCap"]["recall"], 5 / 6)
        self.assertEqual(result["endToEnd"]["acceptedCorrectObjectCoverage"], 5 / 6)
        self.assertEqual(result["overflowFrames"], 1)
        self.assertEqual(result["framesAboveObjectCap"], 1)
        self.assertEqual(needed_anchors(detections), set(range(5)))

    def test_multiobject_coverage_reports_real_counts_and_groups(self):
        result = evaluate_frames(fixture_predictions(master_manifest(), "valid"), THRESHOLDS)
        for count in range(2, 6):
            metrics = result["multiobject"][str(count)]
            self.assertEqual(metrics["frames"], 1)
            self.assertEqual(metrics["groups"], 1)
            self.assertEqual(metrics["groundTruthObjects"], count)
            self.assertEqual(metrics["fullyRecognizedFrameRate"], 1)

    def test_missing_logits_invalid_scores_and_bad_class_ids_fail(self):
        item = frame()
        missing = FramePrediction(item.path, item.group_id, item.split, item.truth,
                                  (Detection(BOX, 0.9, 0),), 1)
        with self.assertRaises(ValueError):
            evaluate_frames([missing], THRESHOLDS)
        for target in (True, 10, -1, 1.5):
            with self.assertRaises(ValueError):
                GroundTruth(BOX, target, logits(0))
        for score in (float("nan"), -1, 2, True):
            with self.assertRaises(ValueError):
                Detection(BOX, score, 0, logits(0))
        with self.assertRaises(ValueError):
            Detection(BOX, 0.9, 0, (0,) * 11)
        with self.assertRaises(ValueError):
            checked_thresholds({**THRESHOLDS, "minMargin": float("inf")})

    def test_top_two_margin_is_required_and_class_ties_follow_shared_order(self):
        close = (20.0, 19.95) + (0.0,) * 8
        self.assertEqual(classify(close, 0.5, 0.0), (0, True))
        self.assertEqual(classify(close, 0.5, 0.1), (0, False))
        self.assertEqual(classify((0.0,) * 10, 0.1, 0.0), (0, True))
        self.assertEqual(classify((0.0,) * 10, 0.1, 0.01), (0, False))

    def test_selection_is_validation_only_finite_and_deterministic(self):
        negatives = FramePrediction("neg.png", "neg", "valid", (), (Detection(BOX, 0.2, 0, logits(0)),), 1)
        frames = tuple(frame(index) for index in range(10)) + (negatives,)
        selection = calibrate_validation(frames)
        self.assertEqual(selection, calibrate_validation(frames))
        self.assertEqual(selection["thresholds"], {
            "detectorScoreThreshold": 0.25, "iouThreshold": 0.3,
            "classifierScoreThreshold": 0.5, "minMargin": 0.0,
        })
        self.assertEqual(len(selection["candidates"]), math.prod(len(values) for values in DEFAULT_GRID.values()))
        with self.assertRaises(ValueError):
            calibrate_validation([frame(split="test")])
        with self.assertRaises(ValueError):
            calibrate_validation([frame(), frame(split="test")])

    def test_predictions_roundtrip_and_group_label_checksum_rejection(self):
        manifest = master_manifest()
        predictions = fixture_predictions(manifest, "valid")
        document = predictions_document(predictions, "test-input-fingerprint")
        self.assertEqual(load_predictions(document, manifest["images"], "valid", "test-input-fingerprint"), predictions)
        for change in (
            lambda value: value.update(inputFingerprint="different"),
            lambda value: value["frames"][0].update(groupId="wrong"),
            lambda value: value["frames"][0]["truth"][0].update(classId=9),
            lambda value: value["frames"].pop(),
        ):
            broken = copy.deepcopy(document)
            change(broken)
            with self.assertRaises(ValueError):
                load_predictions(broken, manifest["images"], "valid", "test-input-fingerprint")


class DatasetAndPersistenceTests(ProjectFilesTest):
    def make_prepared(self, complete=True):
        prepared = self.folder / "prepared"
        manifest = master_manifest(complete)
        prepared.mkdir()
        write_json(prepared / "manifest.json", manifest)
        write_json(prepared / "report.json", {"cropMargin": 0.1, "seed": 42})
        for index, row in enumerate(manifest["images"]):
            stem = f"{index:06d}_{row['sha256'][:12]}"
            image = prepared / "detector" / "images" / row["split"] / f"{stem}.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"unit-test-inventory-only-not-an-image")
            labels = prepared / "detector" / "labels" / row["split"] / f"{stem}.txt"
            labels.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for instance, obj in enumerate(row["objects"]):
                x, y, w, h = obj["bbox"]
                lines.append(f"0 {x + w / 2:.9f} {y + h / 2:.9f} {w:.9f} {h:.9f}")
                crop = prepared / "classifier" / row["split"] / SLUGS[obj["classId"]] / f"{stem}_{instance}.jpg"
                crop.parent.mkdir(parents=True, exist_ok=True)
                crop.write_bytes(b"unit-test-inventory-only-not-a-crop")
            labels.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="ascii")
        write_json(prepared / "detector" / "data.yaml", {
            "path": str((prepared / "detector").resolve()), "names": ["rimpang"],
            "train": "images/train", "val": "images/valid", "test": "images/test",
        })
        return prepared, manifest

    def test_group_audit_ignores_claimed_counts_and_requires_scenarios(self):
        complete = inspect_manifest(master_manifest())
        self.assertTrue(complete["eligible"])
        self.assertEqual(complete["splits"]["valid"]["groups"], 15)
        incomplete = inspect_manifest(master_manifest(False))
        self.assertFalse(incomplete["eligible"])
        self.assertTrue(any("negative/unknown" in message for message in incomplete["unmetGates"]))
        self.assertTrue(any("2, 3, 4, 5" in message for message in incomplete["unmetGates"]))

    def test_cross_split_groups_duplicate_images_missing_split_rejected(self):
        for change in (
            lambda rows: rows[10].update(groupId=rows[0]["groupId"]),
            lambda rows: rows[1].update(sha256=rows[0]["sha256"]),
            lambda rows: rows[0].pop("split"),
            lambda rows: rows[0].pop("sha256"),
        ):
            manifest = master_manifest()
            change(manifest["images"])
            with self.assertRaises(ValueError):
                inspect_manifest(manifest)

    def test_prepared_fingerprint_binds_actual_files_but_not_label_caches(self):
        prepared, _ = self.make_prepared()
        binding, _, _ = prepared_dataset_binding(prepared)
        cache = prepared / "detector" / "labels" / "valid.cache"
        cache.write_bytes(b"unit-test-cache")
        self.assertEqual(binding, prepared_dataset_binding(prepared)[0])
        crop = next((prepared / "classifier").rglob("*.jpg"))
        crop.write_bytes(b"changed-crop")
        self.assertNotEqual(binding["datasetFingerprint"], prepared_dataset_binding(prepared)[0]["datasetFingerprint"])

    def test_prepared_inventory_and_yolo_labels_cannot_diverge(self):
        prepared, _ = self.make_prepared()
        label = next((prepared / "detector" / "labels" / "train").glob("*.txt"))
        original = label.read_text()
        label.write_text("0 0.5 0.5 1 1\n")
        with self.assertRaises(ValueError):
            prepared_dataset_binding(prepared)
        label.write_text(original)
        extra = prepared / "classifier" / "train" / SLUGS[0] / "unlisted.jpg"
        extra.write_bytes(b"unlisted-unit-test-crop")
        with self.assertRaises(ValueError):
            prepared_dataset_binding(prepared)

    def test_atomic_failure_keeps_previous_checkpoint(self):
        destination = self.folder / "checkpoint.bin"
        destination.write_bytes(b"complete")

        def incomplete(stream):
            stream.write(b"partial")
            raise OSError("simulated interrupted serialization")

        with self.assertRaises(OSError):
            atomic_write(destination, incomplete)
        self.assertEqual(destination.read_bytes(), b"complete")
        self.assertEqual(list(self.folder.iterdir()), [destination])
        atomic_json(self.folder / "history.json", {"epoch": 1})
        with self.assertRaises(ValueError):
            atomic_json(self.folder / "history.json", {"epoch": float("nan")})
        self.assertEqual(read_json(self.folder / "history.json"), {"epoch": 1})

    def test_classifier_resume_binds_data_config_and_best_history(self):
        checkpoint = {
            "formatVersion": 1, "config": {"epochs": 50}, "classSlugs": SLUGS,
            "datasetBinding": {"datasetFingerprint": "one"}, "configFileSha256": "config",
            "epoch": 2, "best": 0.7, "stale": 1,
            "history": [{"epoch": 1, "validMacroF1": 0.5}, {"epoch": 2, "validMacroF1": 0.7},
                        {"epoch": 3, "validMacroF1": 0.6}],
        }
        validate_resume(checkpoint, {"epochs": 50}, SLUGS, {"datasetFingerprint": "one"}, "config")
        for change in (
            lambda value: value.update(datasetBinding={"datasetFingerprint": "two"}),
            lambda value: value.update(configFileSha256="changed"),
            lambda value: value.update(stale=0),
            lambda value: value.update(best=0.8),
            lambda value: value["history"].pop(),
        ):
            broken = copy.deepcopy(checkpoint)
            change(broken)
            with self.assertRaises(ValueError):
                validate_resume(broken, {"epochs": 50}, SLUGS, {"datasetFingerprint": "one"}, "config")

    def test_detector_atomic_recovery_receipt_survives_failed_commit(self):
        output = self.folder / "detector"
        weights = output / "weights"
        weights.mkdir(parents=True)
        binding = {"schemaVersion": 1, "datasetBinding": {"datasetFingerprint": "unit-test"}}
        write_json(output / "training-binding.json", binding)
        (weights / "last.pt").write_bytes(b"epoch-one-unit-test-not-a-model")
        (weights / "best.pt").write_bytes(b"epoch-one-unit-test-not-a-model")
        checkpoint_receipt(output, snapshot=True)
        prior = read_json(output / "checkpoint-inventory.json")
        previous_recovery = output / prior["resumeLast"]
        self.assertEqual(detector_checkpoint_binding(previous_recovery), binding)
        (weights / "last.pt").write_bytes(b"epoch-two-unit-test-not-a-model")
        with patch("training.scripts.train_detector.atomic_json", side_effect=OSError("simulated interrupted receipt")):
            with self.assertRaises(OSError):
                checkpoint_receipt(output, snapshot=True)
        self.assertEqual(detector_checkpoint_binding(previous_recovery), binding)
        checkpoint_receipt(output, snapshot=True)
        self.assertFalse(previous_recovery.exists())
        current = read_json(output / "checkpoint-inventory.json")
        self.assertEqual(detector_checkpoint_binding(output / current["resumeLast"]), binding)
        (weights / "last.pt").write_bytes(b"optimizer-stripped-final-unit-test")
        checkpoint_receipt(output, snapshot=False)
        final = read_json(output / "checkpoint-inventory.json")
        self.assertEqual(final["resumeLast"], current["resumeLast"])
        self.assertEqual(detector_checkpoint_binding(weights / "last.pt"), binding)
        (weights / "last.pt").write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            detector_checkpoint_binding(weights / "last.pt")

    def test_provenance_missing_is_a_blocking_gate_not_fabricated(self):
        dataset = {"cropMargin": 0.1}
        checkpoint = {"config": read_json(ROOT / "training" / "configs" / "classifier.json")}
        config = read_json(ROOT / "training" / "configs" / "export.json")
        self.assertEqual(len(checkpoint_eligibility(checkpoint, None, dataset, config)), 2)
        checkpoint["datasetBinding"] = dataset
        detector = {"datasetBinding": dataset, "config": {"model": "yolo11n.pt"}}
        self.assertEqual(checkpoint_eligibility(checkpoint, detector, dataset, config), [])


class EvaluationOrchestrationTests(ProjectFilesTest):
    def evaluate_fixture(self, *, complete=True, output_name="evaluation", resume=False, test_predictions=None):
        manifest = master_manifest(complete)
        binding = {
            "preparedManifestSha256": json_fingerprint(manifest),
            "preparedReportSha256": "b" * 64, "preparedFilesSha256": "c" * 64,
            "preparedFileCount": 1, "cropMargin": 0.1, "datasetFingerprint": "d" * 64,
        }
        config = read_json(ROOT / "training" / "configs" / "classifier.json")
        detector_config = read_json(ROOT / "training" / "configs" / "detector.json")
        args = SimpleNamespace(
            prepared=self.folder / "prepared", data_root=self.folder / "images",
            detector=self.folder / "detector.pt", classifier=self.folder / "classifier.pt",
            output=self.folder / output_name, config=ROOT / "training" / "configs" / "export.json",
            device="cpu", workers=4, resume=resume,
        )
        for file in (args.detector, args.classifier):
            if not file.exists():
                file.write_bytes(b"orchestration-unit-test-only-never-loaded-as-checkpoint")
        backend = SimpleNamespace(
            device="cpu", classifier_config=config,
            checkpoint={"config": config, "datasetBinding": binding},
            detector_metrics=lambda *unused: {
                "status": "measured", "precision": 1.0, "recall": 1.0, "mAP50": 1.0, "mAP50_95": 1.0,
            },
        )
        calls = []

        def predict(records, split, data_root, selected_backend, thresholds=None):
            calls.append(split)
            self.assertIs(selected_backend, backend)
            if split == "test":
                self.assertTrue((args.output / "frozen-thresholds.json").is_file())
                frozen = read_json(args.output / "frozen-thresholds.json")
                self.assertEqual(thresholds, frozen["thresholds"])
                if test_predictions is not None:
                    return test_predictions
            else:
                self.assertIsNone(thresholds)
            return fixture_predictions(manifest, split)

        with (
            patch.object(evaluate_models, "prepared_dataset_binding", return_value=(binding, manifest, inspect_manifest(manifest))),
            patch("training.dataset.audit_manifest", return_value=(manifest["images"], {})),
            patch.object(evaluate_models, "TorchBackend", return_value=backend),
            patch.object(evaluate_models, "detector_checkpoint_binding", return_value={"datasetBinding": binding, "config": detector_config}),
            patch.object(evaluate_models, "predict_split", side_effect=predict),
            patch.object(evaluate_models.importlib.metadata, "version", return_value="unit-test"),
            redirect_stdout(io.StringIO()),
        ):
            code = evaluate_models.evaluate(args)
        return args, code, calls, manifest

    def test_freeze_precedes_test_and_eligible_evidence_passes_export_gate(self):
        args, code, calls, _ = self.evaluate_fixture()
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["valid", "test"])
        calibration, report, frozen = verify_calibration_evidence(
            args.output / "calibration.json", digest(args.detector), digest(args.classifier),
            read_json(args.config), SLUGS,
        )
        self.assertEqual(calibration["status"], "validated")
        self.assertEqual(report["deployment"]["status"], "blocked-pending-parity-and-review")
        self.assertEqual(frozen["selectionSplit"], "valid")
        self.assertEqual(frozen["protocol"], protocol())
        self.assertEqual(report["metrics"]["test"]["endToEnd"]["macroF1"], 1)
        self.assertFalse((args.output / "calibration-diagnostic.json").exists())

    def test_missing_negative_and_multiobject_never_emits_validated_calibration(self):
        args, code, _, _ = self.evaluate_fixture(complete=False)
        self.assertEqual(code, 2)
        self.assertFalse((args.output / "calibration.json").exists())
        diagnostic = read_json(args.output / "calibration-diagnostic.json")
        self.assertEqual(diagnostic["status"], "blocked")
        self.assertFalse(diagnostic["eligibility"]["eligible"])
        report = read_json(args.output / "evaluation-report.json")
        self.assertIsNone(report["metrics"]["test"]["negativeOrUnknown"]["frameFalseAcceptRate"])
        with self.assertRaises(ValueError):
            verify_calibration_evidence(args.output / "calibration-diagnostic.json",
                                        digest(args.detector), digest(args.classifier), read_json(args.config), SLUGS)

    def test_resume_reuses_frozen_thresholds_and_predictions(self):
        args, _, _, _ = self.evaluate_fixture()
        frozen_sha = digest(args.output / "frozen-thresholds.json")
        with patch.object(evaluate_models, "calibrate_validation", side_effect=AssertionError("must not retune")):
            resumed, code, calls, _ = self.evaluate_fixture(resume=True)
        self.assertEqual(code, 0)
        self.assertEqual(calls, [])
        self.assertEqual(digest(resumed.output / "frozen-thresholds.json"), frozen_sha)

    def test_test_predictions_cannot_change_selected_thresholds(self):
        manifest = master_manifest()
        difficult = tuple(
            FramePrediction(item.path, item.group_id, item.split, item.truth, (), 0)
            for item in fixture_predictions(manifest, "test")
        )
        args, _, _, _ = self.evaluate_fixture(output_name="good-test")
        other, _, _, _ = self.evaluate_fixture(output_name="bad-test", test_predictions=difficult)
        self.assertEqual(read_json(args.output / "frozen-thresholds.json")["thresholds"],
                         read_json(other.output / "frozen-thresholds.json")["thresholds"])
        report = read_json(other.output / "evaluation-report.json")
        self.assertEqual(report["metrics"]["test"]["endToEnd"]["macroF1"], 0)
        self.assertEqual(report["metrics"]["valid"]["endToEnd"]["macroF1"], 1)

    def test_tampered_frozen_predictions_checkpoint_or_report_is_rejected(self):
        args, _, _, _ = self.evaluate_fixture()
        config = read_json(args.config)
        with self.assertRaises(ValueError):
            verify_calibration_evidence(args.output / "calibration.json", "0" * 64, digest(args.classifier), config, SLUGS)
        search = args.output / "validation-search.json"
        search.write_text("{}")
        with self.assertRaises(ValueError):
            self.evaluate_fixture(resume=True)
        report = args.output / "evaluation-report.json"
        report.write_text("{}")
        with self.assertRaises(ValueError):
            verify_calibration_evidence(args.output / "calibration.json", digest(args.detector), digest(args.classifier), config, SLUGS)

    def test_existing_output_guard_runs_before_dependency_loading(self):
        args, _, _, _ = self.evaluate_fixture()
        with patch.object(evaluate_models, "TorchBackend", side_effect=AssertionError("must not load models")):
            with self.assertRaises(ValueError):
                evaluate_models.evaluate(args)

    def test_help_requires_only_standard_library(self):
        for module in ("evaluate_models", "train_classifier", "train_detector", "export_models"):
            result = subprocess.run(
                [sys.executable, "-S", "-m", f"training.scripts.{module}", "--help"],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
