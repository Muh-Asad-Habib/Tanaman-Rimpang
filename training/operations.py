import hashlib
import json
import os
import re
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from training.common import digest, read_json, write_json


def identifier(value):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}", value) or value in (".", ".."):
        raise ValueError(f"Invalid workspace identifier: {value!r}")
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe archive path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError(f"Unsafe archive path: {value!r}")
    return path


def verify_artifacts(folder):
    folder = Path(folder).resolve()
    manifest = read_json(folder / "artifact-manifest.json")
    if manifest.get("schemaVersion") != 1 or not isinstance(manifest.get("files"), list):
        raise ValueError("Invalid artifact manifest.")
    expected = {"artifact-manifest.json"}
    for item in manifest["files"]:
        relative = relative_path(item["path"])
        if str(relative) in expected:
            raise ValueError(f"Duplicate artifact path: {relative}")
        expected.add(str(relative))
        file = folder.joinpath(*relative.parts)
        if file.is_symlink() or not file.resolve().is_relative_to(folder) or not file.is_file():
            raise ValueError(f"Missing or unsafe artifact: {relative}")
        if file.stat().st_size != item["bytes"] or digest(file) != item["sha256"]:
            raise ValueError(f"Artifact checksum/size mismatch: {relative}")
    actual = {file.relative_to(folder).as_posix() for file in folder.rglob("*") if file.is_file()}
    if actual != expected:
        raise ValueError(f"Unexpected artifact inventory: {sorted(actual ^ expected)}")
    return manifest


def extract_artifacts(archive, destination, expected_sha256):
    archive, destination = Path(archive).resolve(), Path(destination).resolve()
    if not re.fullmatch(r"[a-f0-9]{64}", expected_sha256) or digest(archive) != expected_sha256:
        raise ValueError("Downloaded archive checksum mismatch.")
    if destination.exists():
        raise ValueError("Artifact destination exists; choose a new directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rimpang-results-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "results"
        stage.mkdir()
        with tarfile.open(archive, "r:*") as bundle:
            members, names = bundle.getmembers(), set()
            for member in members:
                relative = relative_path(member.name.rstrip("/") if member.isdir() else member.name)
                if str(relative) in names or not (member.isfile() or member.isdir()):
                    raise ValueError(f"Unsafe or duplicate archive member: {member.name}")
                names.add(str(relative))
            bundle.extractall(stage, members=members, filter="data")
        manifest = verify_artifacts(stage)
        stage.rename(destination)
    return manifest


def package_artifacts(run, output, allow_incomplete=False):
    run, output = Path(run).resolve(), Path(output).resolve()
    if not run.is_dir():
        raise ValueError(f"Run directory does not exist: {run}")
    if output.is_relative_to(run):
        raise ValueError("The transfer directory must be outside the run directory.")
    required = (
        "detector/weights/best.pt", "detector/weights/last.pt",
        "classifier/best.pt", "classifier/last.pt", "web/manifest.json",
        "web/detector.onnx", "web/classifier.onnx", "web/calibration.json",
    )
    missing = [name for name in required if not (run / name).is_file()]
    if missing and not allow_incomplete:
        raise ValueError(f"Run is incomplete; diagnostics require --allow-incomplete: {missing}")
    for status_file in (run / "jobs").glob("*/status.json"):
        if read_json(status_file).get("status") == "running":
            raise ValueError(f"Cannot package a run while a job is running: {status_file.parent.name}")
    files = []
    for file in sorted(run.rglob("*")):
        if file.is_symlink():
            raise ValueError(f"Symlinks cannot be packaged: {file}")
        if not file.is_file():
            continue
        relative = file.relative_to(run).as_posix()
        if file.name == ".lock":
            continue
        if relative == "artifact-manifest.json":
            raise ValueError("An already collected artifact folder cannot be packaged as a new run.")
        if file.name.startswith(".") or file.suffix in (".tmp", ".part"):
            raise ValueError(f"Remove or resolve temporary/private run file before packaging: {relative}")
        files.append({"path": relative, "bytes": file.stat().st_size, "sha256": digest(file)})
    if not files:
        raise ValueError("There are no run artifacts to collect.")
    manifest = {
        "schemaVersion": 1, "runId": identifier(run.name),
        "status": "incomplete" if missing else "complete", "missing": missing, "files": files,
    }
    output.mkdir(parents=True, exist_ok=True)
    content_id = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    archive = output / f"{run.name}-{content_id}.tar.gz"
    if archive.exists():
        metadata = archive.with_suffix(".json")
        if not metadata.is_file():
            raise ValueError(f"Existing result archive has no transfer receipt: {archive}")
        result = read_json(metadata)
        if result.get("sha256") != digest(archive) or result.get("bytes") != archive.stat().st_size:
            raise ValueError(f"Existing result archive is corrupt: {archive}")
        return {**result, "archive": str(archive)}
    descriptor, temporary_name = tempfile.mkstemp(prefix=".results-", suffix=".tar.gz", dir=output)
    os.close(descriptor)
    try:
        with tempfile.TemporaryDirectory(prefix=".manifest-", dir=output) as temporary:
            manifest_file = Path(temporary) / "artifact-manifest.json"
            write_json(manifest_file, manifest)
            with tarfile.open(temporary_name, "w:gz", compresslevel=1) as bundle:
                bundle.add(manifest_file, arcname="artifact-manifest.json", recursive=False)
                for item in files:
                    file = run / item["path"]
                    if file.stat().st_size != item["bytes"] or digest(file) != item["sha256"]:
                        raise ValueError(f"Artifact changed during collection: {item['path']}")
                    bundle.add(file, arcname=item["path"], recursive=False)
        os.replace(temporary_name, archive)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    result = {"archive": str(archive), "sha256": digest(archive), "bytes": archive.stat().st_size,
              "runId": run.name, "status": manifest["status"]}
    atomic_json(archive.with_suffix(".json"), result)
    return result
