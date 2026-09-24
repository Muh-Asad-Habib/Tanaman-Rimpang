import argparse
from pathlib import Path

from training.bundle import install_bundle
from training.common import ROOT


def main():
    parser = argparse.ArgumentParser(description="Install both validated model artifacts before atomically publishing their manifest.")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--destination", type=Path, default=ROOT / "web" / "public" / "models")
    args = parser.parse_args()
    manifest = install_bundle(args.bundle, args.destination)
    print(f"Installed bundle {manifest['version']} at {args.destination.resolve()}")
    print("Rebuild/redeploy Next.js if the hosting platform serves an immutable artifact.")


if __name__ == "__main__":
    main()
