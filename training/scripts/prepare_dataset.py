import argparse
import tempfile
from pathlib import Path

from training.common import read_json, write_json
from training.dataset import audit_manifest, assign_splits, prepare_images


def main():
    parser = argparse.ArgumentParser(description="Build reviewed group-disjoint YOLO + ROI datasets without changing original files.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-margin", type=float, default=0.1)
    args = parser.parse_args()
    if not 0 <= args.crop_margin <= 0.5:
        parser.error("--crop-margin must be between 0 and 0.5")
    destination = args.output.resolve()
    if destination.exists():
        parser.error("Output exists. Use a new versioned directory; no overwrite is performed.")
    records, report = audit_manifest(read_json(args.manifest), args.data_root)
    records = assign_splits(records, args.seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rimpang-prepare-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        prepare_images(records, args.data_root, stage, args.crop_margin)
        write_json(stage / "manifest.json", {"schemaVersion": 1, "labelsVersion": 1, "images": records})
        write_json(stage / "report.json", {**report, "seed": args.seed, "cropMargin": args.crop_margin})
        # JSON is valid YAML; Ultralytics resolves the absolute dataset root on this server.
        write_json(stage / "detector" / "data.yaml", {"path": str(destination / "detector"),
                   "train": "images/train", "val": "images/valid", "test": "images/test", "names": ["rimpang"]})
        stage.rename(destination)
    print(f"Prepared {len(records)} images: {destination}")


if __name__ == "__main__":
    main()
