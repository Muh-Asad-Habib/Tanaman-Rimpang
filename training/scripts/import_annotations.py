import argparse
import json
from pathlib import Path

from training.annotations import (
    import_annotation_tasks,
    invalid_review_report,
    publish_annotation_files,
    read_annotation_json,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate human-reviewed Label Studio JSON against the authoritative image sidecar.",
        epilog=(
            "Only full JSON exports are supported. A required reviewed checkbox, an include/negative/exclude "
            "decision, and verified groups gate acceptance; no enterprise review feature is needed. "
            "Conflicting completed annotations must be resolved in Label Studio. Every expected task must "
            "be accepted or explicitly excluded. Incomplete/invalid imports write review-report.json only "
            "and exit 2; no partial manifest is published. Ready imports additionally write manifest.json. "
            "Then run audit_dataset/prepare_dataset for group-disjoint class coverage; ready here is not permission to train."
        ),
    )
    parser.add_argument("--export", required=True, type=Path, dest="export_path", help="Full Label Studio JSON export, not JSON_MIN.")
    parser.add_argument("--sidecar", required=True, type=Path, help="Original sidecar.json produced by prepare_annotations.")
    parser.add_argument("--image-root", required=True, type=Path, help="The unchanged normalized images, relative root for the output manifest.")
    parser.add_argument("--output", required=True, type=Path, help="New versioned result directory outside image-root; never overwritten.")
    args = parser.parse_args(argv)
    try:
        manifest, report = import_annotation_tasks(read_annotation_json(args.export_path),
                                                    read_annotation_json(args.sidecar), args.image_root)
    except (ValueError, OSError) as exc:
        manifest, report = None, invalid_review_report(exc)
    files = {"review-report.json": report}
    if manifest is not None:
        files["manifest.json"] = manifest
    try:
        destination = publish_annotation_files(args.output, args.image_root, files)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": report["status"], "counts": report["counts"], "output": str(destination)}))
    return 0 if manifest is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
