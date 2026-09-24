import argparse
import json
from pathlib import Path

from training.common import read_json
from training.dataset import audit_manifest


def main():
    parser = argparse.ArgumentParser(description="Read-only audit of master annotations and actual image contents.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()
    _, report = audit_manifest(read_json(args.manifest), args.data_root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
