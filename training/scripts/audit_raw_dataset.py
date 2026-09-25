import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Audit recovered images, normalize EXIF losslessly, and promote complete incoming to immutable raw.",
        epilog=(
            "Outputs: DATA/manifests/raw-audit.json (including failures and duplicate proposals), "
            "normalized-inventory.partial.json (NOT annotation-ready), and only after success "
            "normalized-inventory.json with status=complete. Its images rows expose relative PNG path, "
            "sha256, width/height, driveId, originalPath/originalSha256, classProposal, pending source provenance, "
            "and a pending groupSuggestion or null. No bbox, reviewed group or master manifest is invented. "
            "Raw files remain unchanged; incomplete audits keep incoming/images.incoming for resume. "
            "Exit: 0 complete, 1 incomplete/failed, 130 interrupted."
        ),
    )
    parser.add_argument("--dataset-root", required=True, type=Path, help="Same DATA root used by download_dataset.")
    parser.add_argument("--near-duplicates", action="store_true", help="Report dHash similarity candidates; never infer independent groups.")
    parser.add_argument("--max-distance", type=int, default=4, help="dHash Hamming distance, 0..16 (default 4).")
    parser.add_argument("--max-pairs", type=int, default=10000, help="Maximum reported near-duplicate pairs; truncation is explicit.")
    args = parser.parse_args()
    from training.ingestion import audit_raw_dataset
    try:
        report = audit_raw_dataset(
            args.dataset_root, near_duplicates=args.near_duplicates,
            max_distance=args.max_distance, max_pairs=args.max_pairs,
        )
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        return 1
    print(json.dumps({
        key: report[key] for key in (
            "status", "expectedImages", "recoveredImages", "validImages", "excludedImages",
            "readyForAnnotation", "trainingReady", "errors",
        )
    }, indent=2))
    print(f"Audit report: {args.dataset_root / 'manifests' / 'raw-audit.json'}")
    return 0 if report["status"] == "complete" else 130 if report["status"] == "interrupted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
