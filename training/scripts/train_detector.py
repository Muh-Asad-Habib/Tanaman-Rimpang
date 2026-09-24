import argparse
import os
from pathlib import Path

from training.common import ROOT, read_json


def main():
    parser = argparse.ArgumentParser(description="Train generic YOLO11n on the server; never run as part of the web startup.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "training" / "configs" / "detector.json")
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error("Prepared data.yaml is missing.")
    if args.output.exists() and not args.resume:
        parser.error("Output exists; use another experiment directory or an explicit --resume checkpoint.")
    os.environ["YOLO_AUTOINSTALL"] = "False"
    from ultralytics import YOLO
    config = read_json(args.config)
    weights = config.pop("model")
    if args.resume:
        if not args.resume.is_file():
            parser.error("Resume checkpoint does not exist.")
        YOLO(str(args.resume)).train(resume=True, device=args.device)
    else:
        YOLO(weights).train(data=str(args.data.resolve()), project=str(args.output.parent.resolve()),
                            name=args.output.name, device=args.device, exist_ok=False, **config)


if __name__ == "__main__":
    main()
