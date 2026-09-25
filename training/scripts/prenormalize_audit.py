"""Normalize images missing from an interrupted audit cache in parallel, then resume the audit.

Uses the exact ingestion._normalize function; the resumed audit_raw_dataset re-verifies every
cached derivative (hash, size, orientation) before accepting it.
"""

import argparse
from multiprocessing import get_context
from pathlib import Path

from training.common import digest
from training.ingestion import _atomic_json, _load_document, _normalize, _source, _target


def _work(job):
    row, receipt, source, destination, relative = job
    try:
        info = _normalize(Path(source), Path(destination))
    except Exception as error:  # the resumed audit records the real exclusion
        return None, f"{row['path']}: {error}"
    return {
        "path": relative, "sha256": digest(Path(destination)), **info, "driveId": row["driveId"],
        "originalPath": row["path"], "originalSha256": receipt["sha256"],
        "classProposal": row["classProposal"], "source": row["source"],
        "groupSuggestion": None, "reviewStatus": "pending",
    }, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    drive = _load_document(root, "manifests/drive-inventory.json", "drive-image-inventory")
    state = _load_document(root, "manifests/download-progress.json", "drive-download-progress")
    partial = _load_document(root, "manifests/normalized-inventory.partial.json", "normalized-image-inventory")
    incoming, staging = _target(root, "incoming"), _target(root, "images.incoming")
    done = {row["originalPath"] for row in partial["images"]}
    jobs = []
    for row in drive["files"]:
        if row["kind"] != "image" or row["path"] in done:
            continue
        receipt = state["files"].get(row["path"], {})
        if receipt.get("status") != "complete" or receipt.get("driveId") != row["driveId"]:
            continue
        relative = f"{row['classProposal']['slug']}/{Path(row['path']).name}.png"
        jobs.append((row, receipt, str(_source(incoming, row["path"])), str(_target(staging, relative)), relative))
    print(f"cached={len(done)} pending={len(jobs)}", flush=True)
    failures = []
    with get_context("spawn").Pool(args.workers) as pool:
        for count, (entry, error) in enumerate(pool.imap_unordered(_work, jobs, chunksize=4), 1):
            if entry:
                partial["images"].append(entry)
            else:
                failures.append(error)
            if count % 50 == 0:
                print(f"normalized {count}/{len(jobs)}", flush=True)
    _atomic_json(_target(root, "manifests/normalized-inventory.partial.json"), partial)
    print(f"added={len(jobs) - len(failures)} failed={len(failures)}", flush=True)
    for failure in failures:
        print("FAILED", failure, flush=True)


if __name__ == "__main__":
    main()
