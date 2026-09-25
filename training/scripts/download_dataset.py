import argparse
import json
from pathlib import Path

from training.common import read_json


def main():
    parser = argparse.ArgumentParser(
        description="Recover the approved public Drive dataset into incoming, never directly into raw.",
        epilog=(
            "Default: strict 10-class/5507-image reconciliation. --inventory-only downloads only "
            "ATRIBUSI.txt and rename_mapping.txt. Progress/checksums: DATA/manifests/. "
            "Rerun the same command after interruption; failed or changed files are quarantined, not deleted. "
            "Next run audit_raw_dataset; only its complete normalized-inventory.json is annotation-ready. "
            "Neither command approves sources, licenses, groups or labels, or starts training."
        ),
    )
    parser.add_argument("--dataset-root", required=True, type=Path, help="Dedicated DATA directory on the server.")
    parser.add_argument("--inventory-only", action="store_true", help="Full discovery/reconciliation; no image transfers.")
    parser.add_argument(
        "--expectations", type=Path,
        help='Explicit reviewed deviation JSON: {"imageCounts": {all ten slugs: counts}, "provenanceCounts": {source: counts}}.',
    )
    parser.add_argument("--timeout", type=float, default=60, help="Read timeout in seconds, 0 < timeout <= 600 (default 60).")
    parser.add_argument("--retries", type=int, default=3, help="Transient transfer retries per file, 0..10 (default 3). Discovery errors are not hidden.")
    args = parser.parse_args()
    from training.ingestion import download_dataset
    try:
        report = download_dataset(
            args.dataset_root, inventory_only=args.inventory_only,
            expectations=read_json(args.expectations) if args.expectations else None,
            timeout=args.timeout, retries=args.retries,
        )
    except KeyboardInterrupt:
        print(json.dumps({"status": "interrupted", "progress": str(args.dataset_root / "manifests" / "download-progress.json")}))
        return 130
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error),
                          "progress": str(args.dataset_root / "manifests" / "download-progress.json")}))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
