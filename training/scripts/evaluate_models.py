import argparse
import datetime
import importlib.metadata
import os
import platform
from pathlib import Path

from training.common import ROOT, digest, read_json, taxonomy
from training.evaluation import (
    LIMITATIONS, PROTOCOL_VERSION,
    atomic_json, calibrate_validation, checked_thresholds, checkpoint_eligibility,
    detector_checkpoint_binding, evaluate_frames, finite_number, image_tensor,
    json_fingerprint, load_predictions, predictions_document, prepared_dataset_binding,
    predict_split, protocol,
)


class TorchBackend:
    def __init__(self, detector, classifier, config, device):
        import numpy as np
        import torch
        os.environ["YOLO_AUTOINSTALL"] = "False"
        from ultralytics import YOLO
        from training.models.efficientnetv2_cbam import RimpangClassifier

        self.torch, self.numpy = torch, np
        self.device = torch.device("cuda:0" if device in ("0", "cuda", "cuda:0") else "cpu")
        self.ultralytics_device = "0" if self.device.type == "cuda" else "cpu"
        self.config = config
        self.detector_path = detector
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.checkpoint = torch.load(classifier, map_location="cpu", weights_only=False)
        slugs = [label["slug"] for label in taxonomy()["labels"]]
        if (not isinstance(self.checkpoint, dict) or self.checkpoint.get("formatVersion") != 1
                or self.checkpoint.get("classSlugs") != slugs
                or type(self.checkpoint.get("epoch")) is not int or self.checkpoint["epoch"] < 0):
            raise ValueError("Classifier must be a trusted trained project checkpoint with exact shared class order.")
        self.classifier_config = self.checkpoint["config"]
        if (self.classifier_config.get("inputSize") != 224
                or self.classifier_config.get("mean") != [0.5] * 3
                or self.classifier_config.get("std") != [0.5] * 3):
            raise ValueError("Classifier preprocessing differs from the approved 224/RGB mean/std 0.5 contract.")
        self.classifier = RimpangClassifier(self.classifier_config, pretrained=False).to(self.device).float().eval()
        self.classifier.load_state_dict(self.checkpoint["state_dict"], strict=True)
        yolo = YOLO(str(detector))
        if yolo.task != "detect" or yolo.names != {0: "rimpang"}:
            raise ValueError("Detector must be the trained one-class rimpang detector, not generic COCO weights.")
        self.detector = yolo.model.to(self.device).float().eval()

    def detect(self, image):
        tensor, geometry = image_tensor(image, self.config["detectorInputSize"], detector=True)
        with self.torch.inference_mode():
            output = self.detector(self.torch.from_numpy(tensor).to(self.device))
        # Native Ultralytics Detect returns (decoded predictions, feature maps).
        if isinstance(output, tuple) and len(output) == 2:
            output = output[0]
        if not isinstance(output, self.torch.Tensor) or output.dtype != self.torch.float32:
            raise ValueError("Native detector did not return float32 decoded predictions.")
        return output.detach().cpu().numpy(), geometry

    def classify(self, image, box):
        config = self.classifier_config
        tensor, _ = image_tensor(image, config["inputSize"], box=box, margin=self.config["cropMargin"],
                                 mean=config["mean"], std=config["std"])
        with self.torch.inference_mode():
            output = self.classifier(self.torch.from_numpy(tensor).to(self.device))
        if (not isinstance(output, self.torch.Tensor) or output.dtype != self.torch.float32
                or tuple(output.shape) != (1, 10) or not self.torch.isfinite(output).all().item()):
            raise ValueError("Classifier output must be finite float32 logits 1x10.")
        return output.detach().cpu().tolist()[0]

    def detector_metrics(self, prepared, split, output, thresholds, workers):
        from ultralytics import YOLO
        # Keep Ultralytics fusion/validator state separate from the browser-style native predictions.
        metrics = YOLO(str(self.detector_path)).val(
            data=str((prepared / "detector" / "data.yaml").resolve()),
            split="val" if split == "valid" else "test",
            imgsz=self.config["detectorInputSize"], batch=16, device=self.ultralytics_device,
            workers=workers, single_cls=True, half=False, plots=False, save_json=False,
            verbose=False, conf=0.001, iou=thresholds["iouThreshold"], max_det=300,
            project=str((output / "ultralytics").resolve()), name=split, exist_ok=True,
            seed=42, deterministic=True,
        )
        if metrics is None or not hasattr(metrics, "box"):
            raise ValueError(f"Ultralytics returned no detector metrics for {split}.")
        result = {
            "status": "measured",
            "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
            "settings": protocol()["ultralytics"],
        }
        for key in ("precision", "recall", "mAP50", "mAP50_95"):
            finite_number(result[key], f"Ultralytics {key}", 0, 1)
        return result


def validate_frozen(output, inputs, input_fingerprint, records):
    frozen = read_json(output / "frozen-thresholds.json")
    if (frozen.get("schemaVersion") != 1 or frozen.get("inputs") != inputs
            or frozen.get("inputFingerprint") != input_fingerprint
            or frozen.get("protocol") != protocol() or frozen.get("selectionSplit") != "valid"):
        raise ValueError("Frozen calibration is for a different run, protocol, configuration or dataset; never retune test.")
    checked_thresholds(frozen["thresholds"])
    for filename, key in (
        ("valid-predictions.json", "validPredictionsSha256"),
        ("valid-metrics.json", "validMetricsSha256"),
        ("validation-search.json", "validationSearchSha256"),
    ):
        if digest(output / filename) != frozen.get(key):
            raise ValueError(f"Frozen validation evidence changed or is incomplete: {filename}")
    frames = load_predictions(read_json(output / "valid-predictions.json"), records, "valid", input_fingerprint)
    metrics = evaluate_frames(frames, frozen["thresholds"])
    if metrics != read_json(output / "valid-metrics.json"):
        raise ValueError("Frozen validation metrics do not reproduce from the saved predictions.")
    return frozen, metrics


def evaluate(args):
    if args.output.exists() and not args.resume:
        raise ValueError("Output exists; use a new evaluation directory or explicit --resume. Never retune against test.")
    if args.resume and not args.output.is_dir():
        raise ValueError("--resume requires an existing evaluation directory.")
    if args.device not in ("cpu", "0", "cuda", "cuda:0"):
        raise ValueError("Use cpu or logical CUDA device 0; server resource isolation is configured by the runner.")
    if not 0 <= args.workers <= 4:
        raise ValueError("Use at most four evaluation data-loader workers.")
    for checkpoint in (args.detector, args.classifier):
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise ValueError(f"Missing trusted trained checkpoint: {checkpoint}")
    config = read_json(args.config)
    if (config.get("precision") != "float32" or config.get("detectorInputSize") != 416
            or config.get("cropMargin") != 0.1 or config.get("opset") != 17):
        raise ValueError("Use the approved FP32/opset17/detector416/cropMargin0.1 export configuration.")

    dataset_binding, manifest, inspection = prepared_dataset_binding(args.prepared)
    from training.dataset import audit_manifest
    records, _ = audit_manifest(manifest, args.data_root)
    if any(not inspection["splits"][split]["images"] for split in ("valid", "test")):
        raise ValueError("Both explicit validation and test holdouts are required; no synthetic holdout fallback exists.")
    detector_sha, classifier_sha = digest(args.detector), digest(args.classifier)
    detector_binding = detector_checkpoint_binding(args.detector)
    backend = TorchBackend(args.detector, args.classifier, config, args.device)
    if digest(args.detector) != detector_sha or digest(args.classifier) != classifier_sha:
        raise ValueError("A checkpoint changed while models were loading.")
    provenance_gates = checkpoint_eligibility(backend.checkpoint, detector_binding, dataset_binding, config)
    inputs = {
        "datasetBinding": dataset_binding,
        "detectorCheckpointSha256": detector_sha, "classifierCheckpointSha256": classifier_sha,
        "classSlugs": [label["slug"] for label in taxonomy()["labels"]],
        "classifierConfig": backend.classifier_config,
        "classifierConfigSha256": json_fingerprint(backend.classifier_config),
        "detectorTrainingConfig": detector_binding["config"] if detector_binding else None,
        "exportConfig": config, "exportConfigSha256": json_fingerprint(config),
        "evaluatorSourceSha256": digest(Path(__file__)),
        "evaluationHelpersSha256": digest(ROOT / "training" / "evaluation.py"),
        "classifierDefinitionSha256": digest(ROOT / "training" / "models" / "efficientnetv2_cbam.py"),
        "environment": {
            "python": platform.python_version(), "device": str(backend.device), "workers": args.workers,
            **{name: importlib.metadata.version(name) for name in ("numpy", "torch", "timm", "ultralytics", "Pillow")},
        },
    }
    plan = {"schemaVersion": 1, "inputs": inputs, "protocol": protocol()}
    input_fingerprint = json_fingerprint(plan)
    if args.resume:
        if read_json(args.output / "evaluation-inputs.json") != plan:
            raise ValueError("Resume requires identical model/data/source/config/environment/protocol bindings.")
    else:
        args.output.mkdir(parents=True)
        atomic_json(args.output / "evaluation-inputs.json", plan)

    frozen_path = args.output / "frozen-thresholds.json"
    if frozen_path.exists():
        frozen, valid_metrics = validate_frozen(args.output, inputs, input_fingerprint, records)
    else:
        if (args.output / "test-predictions.json").exists() or (args.output / "evaluation-report.json").exists():
            raise ValueError("Test evidence without frozen validation thresholds is invalid; never recalibrate it.")
        valid_path = args.output / "valid-predictions.json"
        if valid_path.exists():
            frames = load_predictions(read_json(valid_path), records, "valid", input_fingerprint)
        else:
            frames = predict_split(records, "valid", args.data_root, backend)
            atomic_json(valid_path, predictions_document(frames, input_fingerprint))
        selection = calibrate_validation(frames)
        valid_metrics = selection["metrics"]
        atomic_json(args.output / "valid-metrics.json", valid_metrics)
        atomic_json(args.output / "validation-search.json", {
            "inputFingerprint": input_fingerprint, "objective": protocol()["objective"],
            "selectedThresholds": selection["thresholds"], "candidates": selection["candidates"],
        })
        frozen = {
            "schemaVersion": 1, "inputs": inputs, "inputFingerprint": input_fingerprint,
            "protocol": protocol(), "selectionSplit": "valid", "thresholds": selection["thresholds"],
            "validPredictionsSha256": digest(valid_path),
            "validMetricsSha256": digest(args.output / "valid-metrics.json"),
            "validationSearchSha256": digest(args.output / "validation-search.json"),
        }
        # This durable commit must precede every test model invocation.
        atomic_json(frozen_path, frozen)
    frozen_sha = digest(frozen_path)
    thresholds = frozen["thresholds"]

    test_path = args.output / "test-predictions.json"
    if test_path.exists():
        test_document = read_json(test_path)
        if test_document.get("frozenThresholdsSha256") != frozen_sha:
            raise ValueError("Cached test predictions are not bound to the frozen validation thresholds.")
        test_frames = load_predictions(test_document, records, "test", input_fingerprint)
    else:
        test_frames = predict_split(records, "test", args.data_root, backend, thresholds)
        atomic_json(test_path, {
            **predictions_document(test_frames, input_fingerprint), "frozenThresholdsSha256": frozen_sha,
        })
    test_metrics = evaluate_frames(test_frames, thresholds)
    metrics_path = args.output / "ultralytics-metrics.json"
    if metrics_path.exists():
        detector_metrics = read_json(metrics_path)
        if (detector_metrics.get("inputFingerprint") != input_fingerprint
                or detector_metrics.get("frozenThresholdsSha256") != frozen_sha):
            raise ValueError("Cached Ultralytics metrics are not bound to this frozen evaluation.")
    else:
        detector_metrics = {"inputFingerprint": input_fingerprint, "frozenThresholdsSha256": frozen_sha, "splits": {}}
        for split in ("valid", "test"):
            if inspection["splits"][split]["objects"]:
                detector_metrics["splits"][split] = backend.detector_metrics(
                    args.prepared, split, args.output, thresholds, args.workers)
            else:
                detector_metrics["splits"][split] = {"status": "unmeasured", "reason": "No GT objects in this split."}
        atomic_json(metrics_path, detector_metrics)
    for split in ("valid", "test"):
        current = detector_metrics["splits"][split]
        if current["status"] == "measured":
            for key in ("precision", "recall", "mAP50", "mAP50_95"):
                finite_number(current[key], f"{split} Ultralytics {key}", 0, 1)
        elif inspection["splits"][split]["objects"]:
            raise ValueError(f"Detector metrics unexpectedly unmeasured on {split}.")

    current_binding, _, _ = prepared_dataset_binding(args.prepared)
    if (current_binding != dataset_binding or digest(args.detector) != detector_sha
            or digest(args.classifier) != classifier_sha or digest(frozen_path) != frozen_sha):
        raise ValueError("Prepared data, checkpoints or frozen thresholds changed during evaluation.")
    unmet = inspection["unmetGates"] + provenance_gates
    eligibility = {"eligible": not unmet, "unmetGates": unmet}
    report = {
        "schemaVersion": 1, "status": "evaluated", "protocolVersion": PROTOCOL_VERSION,
        "evaluatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "inputs": inputs, "protocol": protocol(), "eligibility": eligibility,
        "datasetInspection": inspection, "frozenThresholdsSha256": frozen_sha,
        "predictionsSha256": {"valid": digest(args.output / "valid-predictions.json"), "test": digest(test_path)},
        "metrics": {
            "valid": {**valid_metrics, "ultralyticsDetector": detector_metrics["splits"]["valid"]},
            "test": {**test_metrics, "ultralyticsDetector": detector_metrics["splits"]["test"]},
        },
        "limitations": LIMITATIONS,
        "deployment": {
            "status": "blocked-pending-parity-and-review",
            "requiredChecks": [
                "Review held-out errors and representativeness; calibration eligibility is not quality approval.",
                "Compare real-image tensors, logits and boxes between PyTorch and exported ONNX.",
                "Run browser golden-image checks covering aspect ratios, border crops, padding, class order, "
                "threshold/margin boundaries and 1..5 objects; resolve interpolation differences.",
                "Validate temporal tracking/confirmation and WASM/WebGPU on actual target phones before activation.",
            ],
        },
    }
    report_path = args.output / "evaluation-report.json"
    atomic_json(report_path, report)
    calibration = {
        "schemaVersion": 1, "status": "validated" if eligibility["eligible"] else "blocked",
        "split": "valid", "protocol": "group-disjoint", "protocolVersion": PROTOCOL_VERSION,
        **{key: inputs[key] for key in (
            "datasetBinding", "detectorCheckpointSha256", "classifierCheckpointSha256",
            "classSlugs", "classifierConfigSha256", "exportConfigSha256",
        )},
        "detectorInputSize": config["detectorInputSize"], "classifierInputSize": backend.classifier_config["inputSize"],
        "classifierMean": backend.classifier_config["mean"], "classifierStd": backend.classifier_config["std"],
        "cropMargin": config["cropMargin"], **thresholds,
        "frozenThresholdsSha256": frozen_sha, "evaluationReportSha256": digest(report_path),
        "eligibility": eligibility, "deploymentStatus": report["deployment"]["status"],
        "note": "Validated means measured threshold selection under this protocol, not production accuracy, "
                "browser parity, human approval, or permission to activate the web bundle.",
    }
    filename = "calibration.json" if eligibility["eligible"] else "calibration-diagnostic.json"
    atomic_json(args.output / filename, calibration)
    print(f"Evaluation saved: {report_path}", flush=True)
    print(f"valid end-to-end macro-F1={valid_metrics['endToEnd']['macroF1']:.6f}; "
          f"frozen test macro-F1={test_metrics['endToEnd']['macroF1']:.6f}", flush=True)
    if unmet:
        print("Export blocked:\n- " + "\n- ".join(unmet), flush=True)
        return 2
    print("Calibration is eligible for export. Real-image/browser parity and human quality review still block activation.", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trusted trained checkpoints, select thresholds only on valid, then evaluate frozen test. "
                    "No training or pretrained downloads. Exit 2 means research metrics saved but export gates unmet.")
    parser.add_argument("--prepared", required=True, type=Path, help="Frozen prepared root containing manifest/report/detector/classifier.")
    parser.add_argument("--data-root", required=True, type=Path, help="Root of original EXIF-normalized manifest images, not classifier crops.")
    parser.add_argument("--detector", required=True, type=Path, help="Trusted project-trained YOLO11n checkpoint; pickle files are executable.")
    parser.add_argument("--classifier", required=True, type=Path, help="Trusted project-trained EfficientNetV2-B0 + CBAM checkpoint.")
    parser.add_argument("--output", required=True, type=Path, help="New evaluation directory, or the same directory with --resume.")
    parser.add_argument("--config", type=Path, default=ROOT / "training" / "configs" / "export.json")
    parser.add_argument("--device", default="cpu", help="cpu or logical CUDA device 0; no automatic GPU fallback.")
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--resume", action="store_true", help="Reuse exact bound predictions/frozen thresholds; never retune after test.")
    args = parser.parse_args()
    try:
        return evaluate(args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
