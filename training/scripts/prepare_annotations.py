import argparse
import json
from pathlib import Path

from training.annotations import (
    PRIVATE_SERVICE_HELP,
    adapt_normalized_inventory,
    prepare_annotation_tasks,
    publish_annotation_files,
    read_annotation_json,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Create unannotated Label Studio CE tasks from a strict normalized-image inventory.",
        epilog=PRIVATE_SERVICE_HELP, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--inventory", required=True, type=Path, help=(
        'JSON {schemaVersion:1, labelsVersion:1, images:[{path,sourceId,proposedClassId,sha256,width,height}]}; '
        "optional row license; or DATA/manifests/normalized-inventory.json from audit_raw_dataset. "
        "Paths are relative to --image-root."
    ))
    parser.add_argument("--image-root", required=True, type=Path, help="Normalized DATA/images; also the Label Studio local-file document root.")
    parser.add_argument("--output", required=True, type=Path, help="New directory outside image-root for tasks.json, label-config.xml and sidecar.json.")
    args = parser.parse_args(argv)
    try:
        inventory, duplicates = adapt_normalized_inventory(read_annotation_json(args.inventory))
        tasks, config, sidecar = prepare_annotation_tasks(inventory, args.image_root)
        files = {"tasks.json": tasks, "label-config.xml": config, "sidecar.json": sidecar}
        if duplicates is not None:
            files["duplicates.json"] = duplicates
        destination = publish_annotation_files(args.output, args.image_root, files)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": "awaiting-human-review", "tasks": len(tasks),
                      "excludedDuplicateAliases": duplicates["excludedAliases"] if duplicates else 0,
                      "bundleId": sidecar["bundleId"], "output": str(destination)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
