import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")


def reject_constant(value):
    raise ValueError(f"Non-finite JSON number is not allowed: {value}")


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject_constant)


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def taxonomy():
    return read_json(ROOT / "shared" / "labels.json")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate_schema(value, name):
    from jsonschema import Draft202012Validator
    errors = sorted(
        Draft202012Validator(read_json(ROOT / "shared" / name)).iter_errors(value),
        key=lambda error: str(error.path),
    )
    if errors:
        raise ValueError("; ".join(f"{list(error.path)}: {error.message}" for error in errors[:10]))


def source_path(root, relative):
    root = Path(root).resolve()
    # Accept portable manifest separators, but never Windows absolute paths on Linux.
    if ":" in relative or relative.startswith(("/", "\\")):
        raise ValueError(f"Manifest path must be relative: {relative}")
    candidate = (root / relative.replace("\\", "/")).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError(f"Missing or out-of-root image: {relative}")
    return candidate
