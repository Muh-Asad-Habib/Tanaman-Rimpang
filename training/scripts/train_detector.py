import argparse
import os
from pathlib import Path

from training.common import ROOT, digest, read_json
from training.evaluation import atomic_copy, atomic_json, detector_checkpoint_binding, prepared_dataset_binding


def checkpoint_receipt(output, *, snapshot):
    output = Path(output)
    inventory_path = output / "checkpoint-inventory.json"
    prior = read_json(inventory_path) if inventory_path.exists() else {"files": {}}
    recoveries = {name: prior.get(name) for name in ("resumeLast", "resumeBest")}
    for name in ("last", "best"):
        source = output / "weights" / f"{name}.pt"
        if snapshot and source.is_file():
            sha = digest(source)
            recovery = output / "weights" / f"resume-{name}-{sha[:16]}.pt"
            if not recovery.is_file():
                atomic_copy(source, recovery)
            if digest(recovery) != sha or digest(source) != sha:
                raise ValueError("Detector checkpoint changed while recording its recovery snapshot.")
            recoveries["resumeLast" if name == "last" else "resumeBest"] = recovery.relative_to(output).as_posix()
    names = ["weights/last.pt", "weights/best.pt"] + [name for name in recoveries.values() if name]
    files = {file.relative_to(output).as_posix(): digest(file)
             for name in names if (file := output / name).is_file()}
    if "weights/last.pt" not in files:
        raise ValueError("Ultralytics did not write a last checkpoint.")
    atomic_json(inventory_path, {
        "schemaVersion": 1, "bindingSha256": digest(output / "training-binding.json"),
        "files": files, **recoveries,
    })
    # Retire only snapshots referenced by the previous committed receipt, after the new commit.
    for name in (prior.get("resumeLast"), prior.get("resumeBest")):
        if name and name not in files:
            file = output / name
            if file.parent == output / "weights" and file.name.startswith(("resume-last-", "resume-best-")):
                file.unlink(missing_ok=True)


def restore_class_names(output):
    # Ultralytics single_cls=True renames the only class to "item"; the shared contract requires "rimpang".
    import torch
    for name in ("last", "best"):
        path = Path(output) / "weights" / f"{name}.pt"
        if not path.is_file():
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("model", "ema"):
            if checkpoint.get(key) is not None:
                checkpoint[key].names = {0: "rimpang"}
        temporary = path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)


def finish_training(output):
    restore_class_names(output)
    checkpoint_receipt(output, snapshot=False)


def main():
    parser = argparse.ArgumentParser(description="Train generic YOLO11n on the server; never run as part of the web startup.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "training" / "configs" / "detector.json")
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", type=Path, help="Trusted last.pt or the atomic resumeLast path in checkpoint-inventory.json.")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error("Prepared data.yaml is missing.")
    if args.output.exists() and not args.resume:
        parser.error("Output exists; use another experiment directory or an explicit --resume checkpoint.")
    if args.data.name != "data.yaml" or args.data.parent.name != "detector":
        parser.error("--data must be detector/data.yaml in a frozen prepared dataset.")
    config = read_json(args.config)
    dataset_binding, _, inspection = prepared_dataset_binding(args.data.parent.parent)
    if any(not count for split in inspection["splits"].values() for count in split["objectsPerClass"].values()):
        parser.error("Prepared train/valid/test must cover all existing ten classes.")
    binding = {
        "schemaVersion": 1, "config": config, "configFileSha256": digest(args.config),
        "datasetBinding": dataset_binding, "dataPath": str(args.data.resolve()),
        "outputPath": str(args.output.resolve()),
    }
    if args.resume:
        if (not args.resume.is_file() or args.resume.resolve().parent != args.output.resolve() / "weights"
                or not (args.resume.name == "last.pt" or
                        args.resume.name.startswith("resume-last-") and args.resume.suffix == ".pt")):
            parser.error("Resume must use last.pt or the receipt's resumeLast checkpoint in the original weights directory.")
        if detector_checkpoint_binding(args.resume) != binding:
            parser.error("Resume source/config/prepared dataset/run differs, or the checkpoint receipt is missing.")
    os.environ["YOLO_AUTOINSTALL"] = "False"
    from ultralytics import YOLO
    weights = config["model"]
    model = YOLO(str(args.resume) if args.resume else weights)

    def initialize_run(trainer):
        if Path(trainer.save_dir).resolve() != args.output.resolve():
            raise ValueError("Ultralytics changed the requested run directory; refusing an untracked experiment.")
        atomic_json(args.output / "training-binding.json", binding)

    model.add_callback("on_pretrain_routine_start", initialize_run)
    model.add_callback("on_model_save", lambda trainer: checkpoint_receipt(args.output, snapshot=True))
    # Ultralytics strips optimizer state in final_eval; keep the earlier atomic resume snapshots.
    model.add_callback("on_train_end", lambda trainer: finish_training(args.output))
    if args.resume:
        if model.ckpt.get("optimizer") is None or model.ckpt.get("epoch", -1) < 0:
            parser.error("Checkpoint was finalized without optimizer state; use the receipt's atomic resumeLast snapshot if needed.")
        model.train(resume=True, device=args.device)
    else:
        model.train(data=str(args.data.resolve()), project=str(args.output.parent.resolve()),
                    name=args.output.name, device=args.device, exist_ok=False,
                    **{key: value for key, value in config.items() if key != "model"})


if __name__ == "__main__":
    main()
