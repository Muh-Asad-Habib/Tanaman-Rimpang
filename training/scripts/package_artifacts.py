import argparse
import json
from pathlib import Path

from training.operations import extract_artifacts, package_artifacts


def main():
    parser = argparse.ArgumentParser(description="Package or safely receive checksum-verified run artifacts.")
    commands = parser.add_subparsers(dest="action", required=True)
    package = commands.add_parser("pack")
    package.add_argument("--run", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)
    package.add_argument("--allow-incomplete", action="store_true")
    receive = commands.add_parser("receive")
    receive.add_argument("--archive", type=Path, required=True)
    receive.add_argument("--output", type=Path, required=True)
    receive.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if args.action == "pack":
        result = package_artifacts(args.run, args.output, args.allow_incomplete)
    else:
        manifest = extract_artifacts(args.archive, args.output, args.sha256)
        result = {"runId": manifest["runId"], "status": manifest["status"],
                  "destination": str(args.output.resolve()), "files": len(manifest["files"])}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
