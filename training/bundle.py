import re
import shutil
import uuid
from pathlib import Path

from training.common import digest, read_json, taxonomy, validate_schema, write_json


def validate_bundle(folder):
    folder = Path(folder).resolve()
    manifest = read_json(folder / "manifest.json")
    validate_schema(manifest, "model-manifest.schema.json")
    if manifest["status"] != "ready":
        raise ValueError("Only a complete, calibrated ready bundle can be installed.")
    if manifest["classSlugs"] != [label["slug"] for label in taxonomy()["labels"]]:
        raise ValueError("Bundle class order differs from shared labels.")
    if manifest["detector"]["file"] == manifest["classifier"]["file"]:
        raise ValueError("Detector and classifier must be distinct artifacts.")
    for key in ("detector", "classifier"):
        artifact = manifest[key]
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+\.onnx", artifact["file"]):
            raise ValueError("Unsafe artifact file name.")
        file = folder / artifact["file"]
        if file.is_symlink() or not file.is_file() or file.stat().st_size == 0 or digest(file) != artifact["sha256"]:
            raise ValueError(f"Missing/empty/mismatched artifact: {file.name}")
    return manifest


def install_bundle(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    manifest = validate_bundle(source)
    destination.mkdir(parents=True, exist_ok=True)
    for key in ("detector", "classifier"):
        artifact = manifest[key]
        name = f"{key}-{artifact['sha256']}.onnx"
        output = destination / name
        if output.exists():
            if output.is_symlink() or digest(output) != artifact["sha256"]:
                raise ValueError(f"Existing content-addressed artifact is corrupt: {name}")
        else:
            temporary = destination / f".{name}.{uuid.uuid4().hex}.tmp"
            try:
                shutil.copyfile(source / artifact["file"], temporary)
                if digest(temporary) != artifact["sha256"]:
                    raise ValueError("Copied artifact checksum mismatch.")
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
        artifact["file"] = name
    temporary_manifest = destination / f".manifest.{uuid.uuid4().hex}.tmp"
    try:
        write_json(temporary_manifest, manifest)
        temporary_manifest.replace(destination / "manifest.json")
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return manifest
