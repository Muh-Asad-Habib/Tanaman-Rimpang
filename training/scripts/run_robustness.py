import argparse
from pathlib import Path

from training.common import ROOT, digest, read_json
from training.evaluation import atomic_json, checked_thresholds


def main():
    parser = argparse.ArgumentParser(
        description="Robustness test: re-run the frozen pipeline on degraded test photos and report per-capture strata.")
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--detector", required=True, type=Path)
    parser.add_argument("--classifier", required=True, type=Path)
    parser.add_argument("--frozen", required=True, type=Path, help="frozen-thresholds.json from evaluate_models.")
    parser.add_argument("--output", required=True, type=Path, help="New JSON report path.")
    parser.add_argument("--config", type=Path, default=ROOT / "training" / "configs" / "export.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--perturbations", nargs="*", help="Subset of degradations; default all.")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; robustness reports are write-once.")

    from training.evaluation import evaluate_frames, predict_split
    from training.robustness import PERTURBATIONS, capture_strata, degradation_table, perturb
    from training.scripts.evaluate_models import TorchBackend
    names = args.perturbations or list(PERTURBATIONS)
    unknown = set(names) - set(PERTURBATIONS)
    if unknown:
        parser.error(f"Unknown perturbations: {sorted(unknown)}; choose from {list(PERTURBATIONS)}")
    records = read_json(args.prepared / "manifest.json")["images"]
    thresholds = checked_thresholds(read_json(args.frozen)["thresholds"])
    backend = TorchBackend(args.detector, args.classifier, read_json(args.config), args.device)
    clean_frames = predict_split(records, "test", args.data_root, backend, thresholds)
    baseline = evaluate_frames(clean_frames, thresholds, include_cases=False)
    degraded = {}
    for name in names:
        frames = predict_split(records, "test", args.data_root, backend, thresholds,
                               transform=lambda image, key, name=name: perturb(image, name, key))
        degraded[name] = evaluate_frames(frames, thresholds, include_cases=False)
        print(f"{name}: macro-F1={degraded[name]['endToEnd']['macroF1']:.4f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, {
        "schemaVersion": 1, "split": "test", "thresholds": thresholds,
        "frozenThresholdsSha256": digest(args.frozen),
        "detectorSha256": digest(args.detector), "classifierSha256": digest(args.classifier),
        "degradations": degradation_table(baseline, degraded),
        "captureStrata": capture_strata(clean_frames, records, thresholds),
        "note": "Synthetic degradations approximate field conditions; they do not replace photos "
                "collected under real lighting/background/location strata.",
    })
    print(f"Robustness report saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
