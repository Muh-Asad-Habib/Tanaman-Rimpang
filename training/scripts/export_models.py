import argparse
import os
import shutil
import tempfile
from pathlib import Path

from training.common import ROOT, digest, read_json, taxonomy, write_json, validate_schema


def session_io(session, input_size):
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Each model must have exactly one input and one output.")
    if inputs[0].type != "tensor(float)" or outputs[0].type != "tensor(float)" or inputs[0].shape != [1, 3, input_size, input_size]:
        raise ValueError("Model must use fixed batch1 RGB NCHW float32 input and float32 output.")
    return inputs[0].name, outputs[0].name


def main():
    parser = argparse.ArgumentParser(description="Export trained models only; a checksum-bound validation calibration report is mandatory.")
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "training" / "configs" / "export.json")
    args = parser.parse_args()
    config, calibration = read_json(args.config), read_json(args.calibration)
    if args.output.exists():
        parser.error("Use a new output directory. Existing bundles are never overwritten.")
    if config["precision"] != "float32":
        parser.error("Browser contract v1 supports float32 only.")
    if calibration.get("status") != "validated" or calibration.get("split") != "valid" or calibration.get("protocol") != "group-disjoint":
        parser.error("Require actual group-disjoint validation calibration; test-set tuning is not accepted.")
    if calibration.get("detectorCheckpointSha256") != digest(args.detector) or calibration.get("classifierCheckpointSha256") != digest(args.classifier):
        parser.error("Calibration was not performed for these exact checkpoints.")
    if calibration.get("cropMargin") != config["cropMargin"]:
        parser.error("Crop margin differs from calibration.")
    slugs = [label["slug"] for label in taxonomy()["labels"]]
    if calibration.get("classSlugs") != slugs:
        parser.error("Calibration class order must match shared labels.")
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    os.environ["YOLO_AUTOINSTALL"] = "False"
    from ultralytics import YOLO
    from training.models.efficientnetv2_cbam import RimpangClassifier
    checkpoint = torch.load(args.classifier, map_location="cpu", weights_only=False)
    if checkpoint.get("classSlugs") != slugs or checkpoint.get("formatVersion") != 1 or checkpoint.get("epoch", -1) < 0:
        parser.error("Classifier checkpoint is incompatible or not trained.")
    model_config = checkpoint["config"]
    if not model_config["cbam"]:
        parser.error("The production hybrid bundle requires CBAM; keep ablation results separate.")
    model = RimpangClassifier(model_config).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    detector_model = YOLO(str(args.detector))
    if detector_model.task != "detect" or len(detector_model.names) != 1 or next(iter(detector_model.names.values())) != "rimpang":
        parser.error("Detector must be a trained one-class 'rimpang' detection model, not a generic COCO checkpoint.")
    if calibration.get("detectorInputSize") != config["detectorInputSize"] or calibration.get("classifierInputSize") != model_config["inputSize"]:
        parser.error("Export resolutions differ from calibration.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rimpang-export-", dir=args.output.parent) as temporary:
        stage = Path(temporary)
        copied_detector = stage / "detector.pt"
        shutil.copyfile(args.detector, copied_detector)
        detector_path = Path(YOLO(str(copied_detector)).export(format="onnx", imgsz=config["detectorInputSize"],
                            batch=1, dynamic=False, nms=False, half=False, simplify=False, opset=config["opset"], device="cpu"))
        classifier_path = stage / "classifier.onnx"
        example = torch.zeros((1, 3, model_config["inputSize"], model_config["inputSize"]))
        torch.onnx.export(model, (example,), str(classifier_path), opset_version=config["opset"],
                          input_names=["images"], output_names=["logits"], dynamic_axes=None, dynamo=False, external_data=False)
        for file in (detector_path, classifier_path):
            graph = onnx.load(str(file))
            onnx.checker.check_model(graph)
            if any(item.data_location == onnx.TensorProto.EXTERNAL for item in graph.graph.initializer):
                raise ValueError("Contract v1 requires self-contained ONNX files.")
        detector_session = ort.InferenceSession(str(detector_path), providers=["CPUExecutionProvider"])
        classifier_session = ort.InferenceSession(str(classifier_path), providers=["CPUExecutionProvider"])
        detector_input, detector_output = session_io(detector_session, config["detectorInputSize"])
        classifier_input, classifier_output = session_io(classifier_session, model_config["inputSize"])
        rng = np.random.default_rng(42)
        # Numeric parity is only a smoke test, not an accuracy or browser benchmark.
        detector_reference = detector_model.model.cpu().float().eval()
        detector_shape = (1, 3, config["detectorInputSize"], config["detectorInputSize"])
        for tensor in (np.zeros(detector_shape, dtype=np.float32), rng.random(detector_shape, dtype=np.float32)):
            actual = detector_session.run(None, {detector_input: tensor})[0]
            if actual.ndim != 3 or actual.shape[:2] != (1, 5) or actual.shape[2] == 0 or not np.isfinite(actual).all():
                raise ValueError("Exported detector output must be finite 1x5xN.")
            with torch.inference_mode():
                expected = detector_reference(torch.from_numpy(tensor))
                if isinstance(expected, (tuple, list)):
                    expected = expected[0]
            np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-3, atol=1e-3)
        for tensor in (example.numpy(), rng.normal(0, 0.5, example.shape).astype(np.float32)):
            with torch.inference_mode():
                expected = model(torch.from_numpy(tensor)).numpy()
            actual = classifier_session.run(None, {classifier_input: tensor})[0]
            if actual.shape != (1, 10) or not np.isfinite(actual).all():
                raise ValueError("Exported classifier output must be finite logits1x10.")
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
        manifest = {
            "schemaVersion": 1, "labelsVersion": 1, "status": "ready", "version": args.version,
            "maxObjects": 5, "classSlugs": slugs,
            "detector": {
                "file": "detector.onnx", "sha256": digest(detector_path), "inputName": detector_input,
                "outputName": detector_output, "inputSize": config["detectorInputSize"], "dtype": "float32",
                "outputLayout": "1x5xN", "padValue": 114,
                "scoreThreshold": calibration["detectorScoreThreshold"], "iouThreshold": calibration["iouThreshold"],
            },
            "classifier": {
                "file": "classifier.onnx", "sha256": digest(classifier_path), "inputName": classifier_input, "outputName": classifier_output,
                "inputSize": model_config["inputSize"], "dtype": "float32",
                "mean": model_config["mean"], "std": model_config["std"], "cropMargin": config["cropMargin"],
                "scoreThreshold": calibration["classifierScoreThreshold"], "minMargin": calibration["minMargin"],
            },
        }
        validate_schema(manifest, "model-manifest.schema.json")
        output = stage / "bundle"
        output.mkdir()
        shutil.copyfile(detector_path, output / "detector.onnx")
        shutil.copyfile(classifier_path, output / "classifier.onnx")
        write_json(output / "manifest.json", manifest)
        write_json(output / "calibration.json", calibration)
        output.rename(args.output)
    print(f"Bundle exported: {args.output}. Real-image/browser parity and held-out evaluation remain required.")


if __name__ == "__main__":
    main()
