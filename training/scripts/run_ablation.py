import argparse
import hashlib
import io
from pathlib import Path

from training.ablation import (
    WHOLE_IMAGE_BOX, build_whole_image_folder, image_level_metrics, pipeline_image_prediction, single_species,
    summarize, summary_markdown, whole_image_prediction,
)
from training.common import ROOT, digest, read_json, source_path
from training.evaluation import atomic_json, atomic_write, checked_thresholds, predict_split


def build(args):
    manifest = read_json(args.prepared / "manifest.json")
    counts = build_whole_image_folder(manifest["images"], args.prepared)
    for split, values in counts.items():
        print(f"{split}: {sum(values.values())} citra satu-spesies")


def whole_image_samples(records, data_root, backend, thresholds):
    from PIL import Image
    samples = []
    for row in records:
        target = single_species(row)
        if row["split"] != "test" or target is None:
            continue
        contents = source_path(data_root, row["path"]).read_bytes()
        if hashlib.sha256(contents).hexdigest() != row["sha256"]:
            raise ValueError(f"Source image changed since preparation: {row['path']}")
        with Image.open(io.BytesIO(contents)) as original:
            image = original.convert("RGB")
        samples.append((target, whole_image_prediction(backend.classify(image, WHOLE_IMAGE_BOX), thresholds)))
    return samples


def evaluate(args):
    from training.scripts.evaluate_models import TorchBackend
    if args.output.exists():
        raise SystemExit("Output exists; use a new ablation directory.")
    records = read_json(args.prepared / "manifest.json")["images"]
    config = read_json(args.config)
    # Thresholds come from the full model's validation calibration; test is never used for tuning.
    thresholds = checked_thresholds(read_json(args.frozen)["thresholds"])
    results, checkpoints = {}, {}
    for variant, checkpoint in (("full", args.full), ("tanpa-cbam", args.no_cbam)):
        if checkpoint is None:
            continue
        backend = TorchBackend(args.detector, checkpoint, config, args.device)
        frames = {frame.path: frame for frame in predict_split(records, "test", args.data_root, backend, thresholds)}
        samples = [(single_species(row), pipeline_image_prediction(frames[row["path"]], thresholds))
                   for row in records if row["split"] == "test" and single_species(row) is not None]
        results[variant], checkpoints[variant] = image_level_metrics(samples), digest(checkpoint)
    if args.whole_image is not None:
        backend = TorchBackend(args.detector, args.whole_image, config, args.device)
        if backend.checkpoint.get("dataVariant") != "ablation-whole-image":
            raise ValueError("--whole-image must be trained on prepared/ablation-whole-image.")
        results["tanpa-yolo"] = image_level_metrics(whole_image_samples(records, args.data_root, backend, thresholds))
        checkpoints["tanpa-yolo"] = digest(args.whole_image)
    rows = summarize(results)
    args.output.mkdir(parents=True)
    atomic_json(args.output / "ablation.json", {
        "schemaVersion": 1, "split": "test", "unit": "single-species image",
        "thresholds": thresholds, "detectorSha256": digest(args.detector),
        "checkpointSha256": checkpoints, "summary": rows, "variants": results,
    })
    table = summary_markdown(rows)
    atomic_write(args.output / "ablation.md", lambda stream: stream.write(table.encode("utf-8")))
    print(table)


def main():
    parser = argparse.ArgumentParser(description="Ablation study: contribution of YOLOv11 localization and CBAM attention.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("build", help="Create prepared/ablation-whole-image for the no-YOLO classifier.")
    prepare.add_argument("--prepared", required=True, type=Path)
    run = commands.add_parser("evaluate", help="Score all variants on the same single-species test images.")
    run.add_argument("--prepared", required=True, type=Path)
    run.add_argument("--data-root", required=True, type=Path)
    run.add_argument("--detector", required=True, type=Path)
    run.add_argument("--full", required=True, type=Path, help="EfficientNetV2-B0 + CBAM checkpoint (crops).")
    run.add_argument("--no-cbam", type=Path, help="Checkpoint trained with configs/classifier.no-cbam.json.")
    run.add_argument("--whole-image", type=Path, help="CBAM checkpoint trained on prepared/ablation-whole-image.")
    run.add_argument("--frozen", required=True, type=Path, help="frozen-thresholds.json from evaluate_models of the full model.")
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--config", type=Path, default=ROOT / "training" / "configs" / "export.json")
    run.add_argument("--device", default="cpu")
    args = parser.parse_args()
    build(args) if args.command == "build" else evaluate(args)


if __name__ == "__main__":
    main()
