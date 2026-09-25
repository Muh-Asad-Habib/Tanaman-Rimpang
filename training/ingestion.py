"""Recover the approved public Drive snapshot; never manufacture reviewed labels.

Download writes only ``incoming``. Audit writes lossless, EXIF-normalized PNGs
to ``images.incoming`` and publishes ``raw``/``images`` only after every expected
file passes. The final normalized inventory is the publication marker; partial
inventories and reports are deliberately not annotation or training manifests.
"""

import contextlib
import hashlib
import importlib.metadata
import math
import os
import re
import time
import uuid
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageFile, ImageOps, __version__ as PILLOW_VERSION

from training.common import digest, read_json, source_path, taxonomy, write_json

FOLDER_ID = "1VEkTUavfY_q47g3jy7hSXwbmTG69_TeX"
METADATA_IDS = {
    "ATRIBUSI.txt": "1KkrM_6ANqZjw_gHMm8hmKS30muKWBVOl",
    "rename_mapping.txt": "1LPQdYOEwqSuGnpvDHbQe23XeFZ7Qg9X7",
}
EXPECTED_COUNTS = {
    "jahe": 741, "jahe-merah": 230, "kencur": 741, "kunyit": 741,
    "kunyit-putih": 347, "lempuyang": 541, "lengkuas": 741,
    "temu-hitam": 237, "temu-kunci": 541, "temulawak": 647,
}
EXPECTED_PROVENANCE = {
    "original": 1200, "dimas_v4": 2940, "empon": 967, "spices": 205,
    "rfempon": 107, "yusuf": 78, "uploads": 10,
}
FOLDER_SLUGS = {
    "jahe": "jahe", "jahe merah": "jahe-merah", "jahe-merah": "jahe-merah",
    "jahe_merah": "jahe-merah", "kencur": "kencur", "kunyit": "kunyit",
    "kunyit putih": "kunyit-putih", "kunyit-putih": "kunyit-putih",
    "kunyit_putih": "kunyit-putih", "lempuyang": "lempuyang", "lengkuas": "lengkuas",
    "temu hitam": "temu-hitam", "temu-hitam": "temu-hitam", "temu_hitam": "temu-hitam",
    "temu kunci": "temu-kunci", "temu-kunci": "temu-kunci", "temu_kunci": "temu-kunci",
    "temulawak": "temulawak",
}
IMAGE_FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}
NORMALIZATION = {"recipe": "exif-transpose-rgb-png-v1", "pillowVersion": PILLOW_VERSION}
PENDING_REVIEWS = [
    "class identity", "source and license", "specimen/session/parent groups",
    "instance annotations", "negative/unknown and real multi-object coverage",
]


class IngestionError(ValueError):
    pass


class ContentError(IngestionError):
    def __init__(self, reason, message):
        self.reason = reason
        super().__init__(message)


def _now():
    return datetime.now(timezone.utc).isoformat()


def safe_relative(value):
    if not isinstance(value, str) or not value or value.startswith(("/", "\\")):
        raise IngestionError(f"Expected a nonempty relative path: {value!r}")
    portable = value.replace("\\", "/")
    parts = portable.split("/")
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
    }
    if any(
        part in ("", ".", "..") or part.endswith((" ", "."))
        or any(ord(char) < 32 or char in ':*?"<>|' for char in part)
        or part.split(".")[0].upper() in reserved
        for part in parts
    ):
        raise IngestionError(f"Unsafe or nonportable path: {value!r}")
    return portable


def _target(root, relative):
    relative = safe_relative(relative)
    root = Path(root)
    if root.is_symlink():
        raise IngestionError(f"Symlink root is not allowed: {root}")
    candidate = root
    for part in relative.split("/"):
        candidate = candidate / part
        if candidate.is_symlink():
            raise IngestionError(f"Symlink is not allowed: {candidate}")
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise IngestionError(f"Path escapes its root: {relative}")
    return candidate


def _source(root, relative):
    _target(root, relative)
    return source_path(root, safe_relative(relative))


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = _target(path.parent, path.name + ".writing")
    try:
        write_json(staging, value)
        with staging.open("r+b") as stream:
            os.fsync(stream.fileno())
        staging.replace(path)
    finally:
        if staging.exists():
            staging.unlink()


@contextlib.contextmanager
def _workspace(dataset_root):
    root = Path(dataset_root).absolute()
    if root.is_symlink():
        raise IngestionError("Dataset root must not be a symlink.")
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    lock_path = _target(root, ".ingestion.lock")
    with lock_path.open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"\0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise IngestionError("Another ingestion/audit process owns this dataset.") from error
        try:
            _target(root, "manifests").mkdir(exist_ok=True)
            yield root
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _expectations(value=None):
    value = value if value is not None else {
        "imageCounts": EXPECTED_COUNTS, "provenanceCounts": EXPECTED_PROVENANCE,
    }
    labels = {item["slug"]: item for item in taxonomy()["labels"]}
    if not isinstance(value, dict) or set(value) != {"imageCounts", "provenanceCounts"}:
        raise IngestionError("Expectations require exactly imageCounts and provenanceCounts.")
    counts, provenance = value["imageCounts"], value["provenanceCounts"]
    if (
        not isinstance(counts, dict) or set(counts) != set(labels)
        or len(labels) != 10 or set(labels) != set(EXPECTED_COUNTS)
        or any(type(number) is not int or number <= 0 for number in counts.values())
        or not isinstance(provenance, dict) or not provenance
        or any(not isinstance(key, str) or not key for key in provenance)
        or any(type(number) is not int or number < 0 for number in provenance.values())
        or sum(counts.values()) != sum(provenance.values())
    ):
        raise IngestionError("Expectations must cover all ten taxonomy slugs and matching provenance totals.")
    return {"imageCounts": dict(counts), "provenanceCounts": dict(provenance)}


def _class_proposal(relative, labels):
    parts = safe_relative(relative).split("/")
    if len(parts) != 2 or parts[0] not in FOLDER_SLUGS:
        raise IngestionError(f"Expected a known class folder and one filename: {relative}")
    if Path(parts[1]).suffix.lower() not in IMAGE_FORMATS:
        raise IngestionError(f"Unsupported image extension: {relative}")
    slug = FOLDER_SLUGS[parts[0]]
    return {"classId": labels[slug]["id"], "slug": slug, "status": "pending"}


def infer_provenance(original_path):
    """Classify name evidence, not licensing, botanical identity or independence."""
    original_path = safe_relative(original_path)
    parts = original_path.split("/")
    filename = parts[-1]
    source_id = "unknown"
    for prefix in ("dimas_v4", "upn_v2", "taufiq_v8", "empon", "rfempon", "yusuf", "spices"):
        if filename.startswith(prefix + "_"):
            source_id = prefix
            break
    if parts[0] == "uploads":
        source_id = "uploads"
    elif (
        len(parts) == 3 and parts[0] in FOLDER_SLUGS
        and parts[1] == parts[0] and re.fullmatch(r"IMG_\d+\.[A-Za-z]+", filename)
    ):
        source_id = "original"
    parent = None
    if source_id in {"dimas_v4", "upn_v2", "taufiq_v8"} and ".rf." in filename:
        parent = {
            "key": filename.split(".rf.", 1)[0],
            "basis": "filename-before-roboflow-suffix",
            "status": "pending",
        }
    return {
        "id": source_id, "originalPath": original_path, "status": "pending",
        "licenseStatus": "pending", "parentSuggestion": parent,
    }


def parse_mapping(text):
    mapping, folded = {}, set()
    for number, line in enumerate(text.lstrip("\ufeff").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("->") + line.count("<-") != 1:
            raise IngestionError(f"Malformed rename mapping at line {number}.")
        if "->" in line:
            old, new = line.split("->")
        else:
            new, old = line.split("<-")
        old, new = safe_relative(old.strip()), safe_relative(new.strip())
        if new.casefold() in folded:
            raise IngestionError(f"Duplicate mapping destination at line {number}: {new}")
        folded.add(new.casefold())
        mapping[new] = infer_provenance(old)
    if not mapping:
        raise IngestionError("Rename mapping contains no destinations.")
    return mapping


def _listing(items):
    if items is None:
        raise IngestionError("gdown returned no inventory; folder discovery did not succeed.")
    rows, seen_paths, seen_ids = [], set(), set()
    labels = {row["slug"]: row for row in taxonomy()["labels"]}
    for item in items:
        path, drive_id = safe_relative(item.path), item.id
        if not isinstance(drive_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{25,200}", drive_id):
            raise IngestionError(f"Invalid Drive file ID for {path}")
        if path.casefold() in seen_paths or drive_id in seen_ids:
            raise IngestionError(f"Duplicate Drive path or file ID: {path}")
        seen_paths.add(path.casefold())
        seen_ids.add(drive_id)
        row = {"path": path, "driveId": drive_id}
        if path in METADATA_IDS:
            if drive_id != METADATA_IDS[path]:
                raise IngestionError(f"Source metadata ID changed: {path}")
            row["kind"] = "metadata"
        else:
            row.update(kind="image", classProposal=_class_proposal(path, labels))
        rows.append(row)
    return sorted(rows, key=lambda row: row["path"])


def reconcile_inventory(rows, mapping, expectations):
    expected = _expectations(expectations)
    images = [row for row in rows if row["kind"] == "image"]
    paths = {row["path"] for row in images}
    actual_counts = Counter(row["classProposal"]["slug"] for row in images)
    provenance = Counter(row["id"] for row in mapping.values()) if mapping is not None else None
    folders = {row["path"].split("/")[0] for row in images}
    report = {
        "schemaVersion": 1,
        "expectedImages": sum(expected["imageCounts"].values()), "listedImages": len(images),
        "expectedImageCounts": expected["imageCounts"], "actualImageCounts": dict(actual_counts),
        "expectedProvenanceCounts": expected["provenanceCounts"],
        "actualProvenanceCounts": dict(provenance) if provenance is not None else None,
        "missingMetadata": sorted(set(METADATA_IDS) - {row["path"] for row in rows}),
        "missingPaths": sorted(set(mapping) - paths) if mapping is not None else [],
        "unexpectedPaths": sorted(paths - set(mapping)) if mapping is not None else [],
        "mappingChecked": mapping is not None,
        "physicalClassFolders": sorted(folders),
    }
    matched = (
        dict(actual_counts) == expected["imageCounts"] and len(folders) == 10
        and not report["missingMetadata"] and not report["missingPaths"] and not report["unexpectedPaths"]
        and (provenance is None or provenance == Counter(expected["provenanceCounts"]))
    )
    report["status"] = "matched" if matched else "mismatch"
    return report


def _read_metadata(path):
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ContentError("metadata_size", f"Source metadata is unexpectedly large: {path.name}")
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip() or "\0" in text or text.lstrip().lower().startswith(("<!doctype html", "<html")):
        raise ContentError("invalid_metadata", f"Empty, binary or HTML source metadata: {path.name}")
    return text


def _quick_check(path, kind):
    size = path.stat().st_size
    if not size:
        raise ContentError("empty_file", f"Empty file: {path.name}")
    if kind == "metadata":
        _read_metadata(path)
        return
    with path.open("rb") as stream:
        prefix = stream.read(64)
    expected = IMAGE_FORMATS.get(path.suffix.lower())
    matches = {
        "JPEG": prefix.startswith(b"\xff\xd8\xff"),
        "PNG": prefix.startswith(b"\x89PNG\r\n\x1a\n"),
        "WEBP": prefix.startswith(b"RIFF") and prefix[8:12] == b"WEBP",
    }
    if not matches.get(expected, False):
        raise ContentError("extension_content_mismatch", f"Image signature does not match extension: {path.name}")


def _tree_issues(root, expected_paths):
    present, issues = set(), []
    if not root.exists():
        return [f"Missing directory: {root.name}"], present
    for directory, subdirs, files in os.walk(
        root, followlinks=False, onerror=lambda error: issues.append(f"Cannot inspect directory: {error}"),
    ):
        for name in subdirs + files:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            try:
                _target(root, relative)
            except IngestionError as error:
                issues.append(str(error))
                if name in subdirs:
                    subdirs.remove(name)
                continue
            if name in files:
                if not path.is_file():
                    issues.append(f"Not a regular file: {relative}")
                else:
                    present.add(relative)
    for relative in sorted(present - set(expected_paths)):
        issues.append(f"Unexpected/partial file: {relative}")
    return issues, present


def _save_progress(root, state):
    state["updatedAt"] = _now()
    state["completedImages"] = sum(
        row.get("status") == "complete" and path not in METADATA_IDS
        for path, row in state["files"].items()
    )
    _atomic_json(_target(root, "manifests/download-progress.json"), state)


def _transfer(root, row, state, downloader, timeout, retries):
    destination = _target(_target(root, "incoming"), row["path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = state["files"].get(row["path"], {})
    if (
        destination.is_file() and receipt.get("status") == "complete"
        and receipt.get("driveId") == row["driveId"]
        and receipt.get("bytes") == destination.stat().st_size
        and receipt.get("sha256") == digest(destination)
    ):
        _quick_check(destination, row["kind"])
        return
    if destination.exists():
        quarantine = _target(root, f"quarantine/{uuid.uuid4().hex}/{row['path']}")
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        destination.rename(quarantine)
        state.setdefault("quarantined", []).append({
            "path": row["path"], "quarantinePath": quarantine.relative_to(root).as_posix(),
            "reason": "unreceipted, changed, or previously invalid download",
        })
    for partial in destination.parent.iterdir():
        if partial.name.startswith(destination.name) and partial.name.endswith(".part"):
            _target(destination.parent, partial.name)
    try:
        downloaded = downloader.download(
            id=row["driveId"], output=str(destination), quiet=True, use_cookies=False,
            resume=True, timeout=(min(30, timeout), timeout), retries=retries,
        )
        if not isinstance(downloaded, str) or Path(downloaded).resolve() != destination.resolve():
            raise IngestionError(f"Downloader did not complete the expected destination: {row['path']}")
        _source(_target(root, "incoming"), row["path"])
        _quick_check(destination, row["kind"])
        state["files"][row["path"]] = {
            "driveId": row["driveId"], "status": "complete",
            "bytes": destination.stat().st_size, "sha256": digest(destination),
        }
    except Exception as error:
        state["files"][row["path"]] = {
            "driveId": row["driveId"], "status": "failed", "error": str(error),
        }
        raise
    finally:
        _save_progress(root, state)


def _get_downloader():
    try:
        version = importlib.metadata.version("gdown")
    except importlib.metadata.PackageNotFoundError as error:
        raise IngestionError("Install training/requirements-data.txt before downloading.") from error
    if version != "6.4.0":
        raise IngestionError(f"Expected gdown==6.4.0, found {version}; install requirements-data.txt.")
    import gdown
    return DirectDriveDownloader(gdown)


class DirectDriveDownloader:
    """gdown folder discovery plus throttled direct file transfer.

    gdown's per-file ``uc?id=`` flow is rejected with "many accesses" after a
    few dozen anonymous requests; the public usercontent endpoint is not.
    """

    URL = "https://drive.usercontent.google.com/download"
    DESCRIPTION = "gdown==6.4.0 folder listing; throttled drive.usercontent file transfer"

    def __init__(self, lister, session=None, interval=0.5, backoff=10.0, sleep=time.sleep, clock=time.monotonic):
        self.lister, self.interval, self.backoff = lister, interval, backoff
        self.sleep, self.clock, self.last = sleep, clock, None
        if session is None:
            import requests
            session = requests.Session()
        self.session = session

    def download_folder(self, **kwargs):
        return self.lister.download_folder(**kwargs)

    def _wait(self):
        if self.last is not None:
            remaining = self.interval - (self.clock() - self.last)
            if remaining > 0:
                self.sleep(remaining)
        self.last = self.clock()

    def download(self, *, id, output, timeout, retries, **_):
        output = Path(output)
        partial = output.with_name(output.name + ".part")
        error = None
        for attempt in range(retries + 1):
            if attempt:
                self.sleep(self.backoff * 2 ** (attempt - 1))
            self._wait()
            try:
                with self.session.get(self.URL, params={"id": id, "export": "download", "confirm": "t"},
                                      stream=True, timeout=timeout) as response:
                    kind = response.headers.get("Content-Type", "")
                    if response.status_code != 200 or kind.startswith("text/html"):
                        raise IngestionError(f"Drive returned HTTP {response.status_code} ({kind or 'no type'}) for {id}.")
                    with partial.open("wb") as stream:
                        for chunk in response.iter_content(1 << 16):
                            stream.write(chunk)
                if partial.stat().st_size <= 0:
                    raise IngestionError(f"Drive returned an empty file for {id}.")
                os.replace(partial, output)
                return str(output)
            except Exception as caught:
                error = caught
                partial.unlink(missing_ok=True)
        raise IngestionError(f"Direct Drive transfer failed after {retries + 1} attempts: {error}")


def _check_current_metadata(root, rows, previous, downloader, timeout, retries):
    check_root = _target(root, f"manifests/source-check/{uuid.uuid4().hex}")
    check_root.mkdir(parents=True)
    observed = {}
    for row in rows:
        if row["kind"] != "metadata":
            continue
        destination = _target(check_root, row["path"])
        result = downloader.download(
            id=row["driveId"], output=str(destination), quiet=True, use_cookies=False,
            resume=True, timeout=(min(30, timeout), timeout), retries=retries,
        )
        if not isinstance(result, str) or Path(result).resolve() != destination.resolve():
            raise IngestionError(f"Could not refresh source metadata: {row['path']}")
        _quick_check(_source(check_root, row["path"]), "metadata")
        observed[row["path"]] = digest(destination)
    _atomic_json(_target(root, "manifests/metadata-check.json"), {
        "checkedAt": _now(), "previousSha256": previous["metadataSha256"],
        "observedSha256": observed, "snapshot": check_root.relative_to(root).as_posix(),
    })
    if observed != previous["metadataSha256"]:
        raise IngestionError("Source metadata changed under the same Drive IDs; see metadata-check.json and use a reviewed new dataset root.")


def _load_document(root, relative, kind):
    document = read_json(_source(root, relative))
    if document.get("schemaVersion") != 1 or document.get("kind") != kind:
        raise IngestionError(f"Unsupported local sidecar: {relative}")
    return document


def _verify_receipt(root, row):
    path = _source(root, row["path"])
    if (
        type(row.get("bytes")) is not int or row["bytes"] <= 0
        or path.stat().st_size != row["bytes"] or digest(path) != row.get("sha256")
    ):
        raise ContentError("checksum_mismatch", f"Size/SHA256 differs from transfer receipt: {row['path']}")
    return path


def download_dataset(dataset_root, *, inventory_only=False, expectations=None,
                     timeout=60, retries=3, downloader=None):
    """Inventory-only still recovers the two text metadata files, never images."""
    expected = _expectations(expectations)
    if not math.isfinite(timeout) or timeout <= 0 or timeout > 600 or type(retries) is not int or not 0 <= retries <= 10:
        raise IngestionError("timeout must be in (0, 600]; retries must be an integer in [0, 10].")
    with _workspace(dataset_root) as root:
        raw, incoming = _target(root, "raw"), _target(root, "incoming")
        if raw.exists():
            if incoming.exists():
                raise IngestionError("Both raw and incoming exist; refusing to overwrite either.")
            original = _load_document(root, "manifests/raw-inventory.json", "raw-image-inventory")
            complete = _load_document(root, "manifests/normalized-inventory.json", "normalized-image-inventory")
            if complete.get("status") != "complete" or complete.get("rawInventorySha256") != digest(
                _source(root, "manifests/raw-inventory.json")
            ):
                raise IngestionError("Raw promotion was interrupted; run audit_raw_dataset to finish.")
            if original.get("expectations") != expected:
                raise IngestionError("Existing immutable raw uses different expectations.")
            issues, present = _tree_issues(raw, [row["path"] for row in original["images"] + original["metadata"]])
            for row in original["images"] + original["metadata"]:
                _verify_receipt(raw, row)
            if issues or len(present) != len(original["images"]) + len(original["metadata"]):
                raise IngestionError(f"Immutable raw tree changed: {issues}")
            return {"status": "already-promoted", "images": len(original["images"]), "rawRoot": str(raw)}
        progress_path = _target(root, "manifests/download-progress.json")
        state = _load_document(root, "manifests/download-progress.json", "drive-download-progress") if progress_path.exists() else {
            "schemaVersion": 1, "kind": "drive-download-progress", "startedAt": _now(), "files": {},
        }
        if not progress_path.exists() and incoming.exists() and any(incoming.iterdir()):
            raise IngestionError("Nonempty incoming has no local progress sidecar; use a fresh dataset root.")
        incoming.mkdir(exist_ok=True)
        state.update(status="discovering", expectedImages=sum(expected["imageCounts"].values()), errors=[])
        _save_progress(root, state)
        try:
            downloader = downloader if downloader is not None else _get_downloader()
            # local_path is deliberately ignored: all destinations are derived and checked here.
            rows = _listing(downloader.download_folder(
                id=FOLDER_ID, output=str(incoming), skip_download=True, quiet=True,
                use_cookies=False, timeout=(min(30, timeout), timeout),
            ))
            _atomic_json(_target(root, "manifests/drive-discovery.json"), {"schemaVersion": 1, "files": rows})
            preliminary = reconcile_inventory(rows, None, expected)
            _atomic_json(_target(root, "manifests/reconciliation.json"), preliminary)
            if preliminary["status"] != "matched":
                raise IngestionError("Drive class counts/metadata differ; see manifests/reconciliation.json (50-file listings are not complete).")
            previous_path = _target(root, "manifests/drive-inventory.json")
            previous = _load_document(root, "manifests/drive-inventory.json", "drive-image-inventory") if previous_path.exists() else None
            if state.get("inventorySha256") and not previous:
                raise IngestionError("Progress references a missing Drive inventory; restore it or use a new dataset root.")
            if previous:
                old_identity = [(row["path"], row["driveId"]) for row in previous["files"]]
                new_identity = [(row["path"], row["driveId"]) for row in rows]
                if old_identity != new_identity or previous["expectations"] != expected:
                    raise IngestionError("Drive IDs/paths or expectations changed; use a reviewed new dataset root.")
                _check_current_metadata(root, rows, previous, downloader, timeout, retries)
            issues, _ = _tree_issues(incoming, [row["path"] for row in rows])
            unsafe = [issue for issue in issues if not issue.startswith("Unexpected/partial file: ")]
            if unsafe:
                raise IngestionError("; ".join(unsafe))
            state["status"] = "metadata"
            for row in rows:
                if row["kind"] == "metadata":
                    _transfer(root, row, state, downloader, timeout, retries)
            mapping = parse_mapping(_read_metadata(_source(incoming, "rename_mapping.txt")))
            reconciliation = reconcile_inventory(rows, mapping, expected)
            _atomic_json(_target(root, "manifests/reconciliation.json"), reconciliation)
            if reconciliation["status"] != "matched":
                raise IngestionError("Drive paths/provenance do not reconcile with rename_mapping.txt; see manifests/reconciliation.json.")
            files = [
                {**row, "source": mapping[row["path"]]} if row["kind"] == "image" else row
                for row in rows
            ]
            inventory = {
                "schemaVersion": 1, "kind": "drive-image-inventory", "labelsVersion": taxonomy()["version"],
                "folderId": FOLDER_ID, "folderUrl": f"https://drive.google.com/drive/folders/{FOLDER_ID}",
                "downloader": getattr(downloader, "DESCRIPTION", "gdown==6.4.0"), "useCookies": False,
                "remoteContentChecksums": "not-provided",
                "integrityScope": "Locally recovered bytes; stable Drive IDs do not prove immutable remote file revisions.",
                "expectations": expected, "expectationsOverridden": expectations is not None,
                "metadataSha256": {name: state["files"][name]["sha256"] for name in METADATA_IDS},
                "files": files,
            }
            if previous and previous != inventory:
                raise IngestionError("Source metadata/inventory changed during resume; use a new dataset root.")
            _atomic_json(previous_path, inventory)
            state["inventorySha256"] = digest(previous_path)
            state["status"] = "inventory-only" if inventory_only else "downloading"
            _save_progress(root, state)
            if inventory_only:
                return {"status": "inventory-only", "images": preliminary["listedImages"],
                        "inventorySha256": state["inventorySha256"], "requiresContentAudit": True}
            failures = []
            for row in files:
                if row["kind"] != "image":
                    continue
                try:
                    _transfer(root, row, state, downloader, timeout, retries)
                except Exception as error:
                    failures.append({"path": row["path"], "error": str(error)})
            issues, present = _tree_issues(incoming, [row["path"] for row in files])
            missing = sorted({row["path"] for row in files} - present)
            if failures or issues or missing:
                state["errors"] = failures + [{"error": issue} for issue in issues] + [
                    {"path": path, "error": "missing"} for path in missing
                ]
                raise IngestionError("Recovery incomplete; inspect download-progress.json, then rerun the same command.")
            complete_rows = [
                {**row, "bytes": state["files"][row["path"]]["bytes"],
                 "sha256": state["files"][row["path"]]["sha256"]}
                for row in files
            ]
            raw_inventory = {
                "schemaVersion": 1, "kind": "raw-image-inventory", "labelsVersion": taxonomy()["version"],
                "status": "downloaded-pending-audit", "folderId": FOLDER_ID, "expectations": expected,
                "driveInventorySha256": state["inventorySha256"],
                "images": [row for row in complete_rows if row["kind"] == "image"],
                "metadata": [row for row in complete_rows if row["kind"] == "metadata"],
            }
            raw_path = _target(root, "manifests/raw-inventory.json")
            _atomic_json(raw_path, raw_inventory)
            state.update(status="downloaded-pending-audit", rawInventorySha256=digest(raw_path))
            _save_progress(root, state)
            return {"status": state["status"], "images": len(raw_inventory["images"]),
                    "rawInventorySha256": state["rawInventorySha256"], "requiresContentAudit": True}
        except (Exception, KeyboardInterrupt) as error:
            state["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            state.setdefault("errors", []).append({"error": str(error) or "interrupted"})
            _save_progress(root, state)
            raise


def _dhash(image):
    thumbnail = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(thumbnail.getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return f"{value:016x}"


def _normalize(source, destination):
    _quick_check(source, "image")
    if ImageFile.LOAD_TRUNCATED_IMAGES:
        raise ContentError("unsafe_decoder", "Pillow LOAD_TRUNCATED_IMAGES must remain False.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source) as check:
                if check.format != IMAGE_FORMATS[source.suffix.lower()]:
                    raise ContentError("extension_content_mismatch", "Pillow format differs from filename extension.")
                if getattr(check, "n_frames", 1) != 1:
                    raise ContentError("multiple_frames", "Animated/multi-frame images require explicit review.")
                check.verify()
            with Image.open(source) as original:
                original.load()
                if min(original.size) < 16:
                    raise ContentError("image_too_small", f"Image must be at least 16x16: {original.size}")
                orientation = original.getexif().get(274, 1)
                if orientation not in range(1, 9):
                    raise ContentError("invalid_exif", f"Invalid EXIF orientation: {orientation}")
                upright = ImageOps.exif_transpose(original)
                if "A" in upright.getbands() or "transparency" in upright.info:
                    alpha = upright.convert("RGBA").getchannel("A")
                    if alpha.getextrema() != (255, 255):
                        raise ContentError("transparent_image", "Nonopaque pixels require an explicitly reviewed background policy.")
                image = upright.convert("RGB")
                image.info.clear()
                info = {
                    "width": image.width, "height": image.height,
                    "originalWidth": original.width, "originalHeight": original.height,
                    "exifOrientation": orientation, "dHash": _dhash(image),
                    "pixelSha256": hashlib.sha256(
                        f"RGB:{image.width}:{image.height}:".encode("ascii") + image.tobytes()
                    ).hexdigest(),
                }
                destination.parent.mkdir(parents=True, exist_ok=True)
                staging = _target(destination.parent, destination.name + ".writing")
                try:
                    image.save(staging, format="PNG")
                    with staging.open("r+b") as stream:
                        os.fsync(stream.fileno())
                    staging.replace(destination)
                finally:
                    if staging.exists():
                        staging.unlink()
                return info
    except ContentError:
        raise
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ContentError("image_decode_or_normalization", str(error)) from error


def _duplicate_clusters(rows, field):
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row[field]].append(row)
    return [
        {"sha256": sha, "paths": [row["originalPath"] for row in group],
         "classConflict": len({row["classProposal"]["classId"] for row in group}) > 1,
         "status": "pending"}
        for sha, group in sorted(by_hash.items()) if len(group) > 1
    ]


def _near_duplicates(rows, enabled, max_distance, max_pairs):
    result = {
        "status": "proposals" if enabled else "not-run", "method": "dhash64",
        "maxDistance": max_distance, "candidateCount": 0, "pairs": [], "truncated": False,
        "reviewStatus": "pending",
        "limitation": "Heuristic similarity only; absence of a match does not prove independent groups.",
    }
    if not enabled:
        return result
    hashes = [(row, int(row["dHash"], 16)) for row in rows]
    for index, (left, left_hash) in enumerate(hashes):
        for right, right_hash in hashes[index + 1:]:
            if left["sha256"] == right["sha256"]:
                continue
            distance = (left_hash ^ right_hash).bit_count()
            if distance <= max_distance:
                result["candidateCount"] += 1
                if len(result["pairs"]) < max_pairs:
                    result["pairs"].append({
                        "left": left["originalPath"], "right": right["originalPath"],
                        "distance": distance, "status": "pending",
                    })
    result["truncated"] = result["candidateCount"] > len(result["pairs"])
    return result


def _cached_image(cached, row, destination):
    if (
        not cached or cached.get("originalSha256") != row["sha256"]
        or cached.get("path") != destination[1] or not destination[0].is_file()
        or cached.get("sha256") != digest(destination[0])
    ):
        return None
    with Image.open(destination[0]) as image:
        if (
            image.format != "PNG" or image.mode != "RGB" or min(image.size) < 16
            or image.size != (cached.get("width"), cached.get("height"))
            or image.getexif().get(274, 1) != 1
        ):
            return None
    keys = ("width", "height", "originalWidth", "originalHeight", "exifOrientation", "dHash", "pixelSha256")
    if (
        any(key not in cached for key in keys) or not re.fullmatch(r"[0-9a-f]{16}", cached["dHash"])
        or not re.fullmatch(r"[0-9a-f]{64}", cached["pixelSha256"])
    ):
        return None
    return {key: cached[key] for key in keys}


def audit_raw_dataset(dataset_root, *, near_duplicates=False, max_distance=4, max_pairs=10000):
    """Return a persistent report; status other than complete must block annotation."""
    if type(max_distance) is not int or not 0 <= max_distance <= 16 or type(max_pairs) is not int or max_pairs <= 0:
        raise IngestionError("Near-duplicate distance must be 0..16; max_pairs must be positive.")
    with _workspace(dataset_root) as root:
        report = {
            "schemaVersion": 1, "kind": "raw-dataset-audit", "status": "auditing",
            "startedAt": _now(), "expectedImages": 0, "recoveredImages": 0, "validImages": 0,
            "excludedImages": 0, "exclusions": [], "errors": [], "trainingReady": False,
            "pendingReviews": PENDING_REVIEWS, "readyForAnnotation": False,
            "duplicateAnalysis": "pending",
            "exactDuplicateScope": "SHA256-verified originals, including image-decode failures",
        }
        normalized = None
        state = None

        def save_partial():
            report["updatedAt"] = _now()
            report["validImages"] = len(normalized["images"]) if normalized else 0
            report["excludedImages"] = len(report["exclusions"])
            _atomic_json(_target(root, "manifests/raw-audit.json"), report)
            if normalized is not None:
                _atomic_json(_target(root, "manifests/normalized-inventory.partial.json"), {**normalized, "status": "partial"})
            published = _target(root, "manifests/normalized-inventory.json")
            if published.exists() and report["status"] != "complete":
                previous = _load_document(root, "manifests/normalized-inventory.json", "normalized-image-inventory")
                if previous.get("status") != report["status"]:
                    _atomic_json(published, {**previous, "status": report["status"]})

        try:
            drive = _load_document(root, "manifests/drive-inventory.json", "drive-image-inventory")
            drive_sha = digest(_source(root, "manifests/drive-inventory.json"))
            expected = _expectations(drive["expectations"])
            report["expectedImages"] = sum(expected["imageCounts"].values())
            incoming, raw = _target(root, "incoming"), _target(root, "raw")
            if incoming.exists() and raw.exists():
                raise IngestionError("Both incoming and raw exist; refusing ambiguous originals.")
            originals = raw if raw.exists() else incoming
            raw_path = _target(root, "manifests/raw-inventory.json")
            inventory = _load_document(root, "manifests/raw-inventory.json", "raw-image-inventory") if raw_path.exists() else None
            state = _load_document(root, "manifests/download-progress.json", "drive-download-progress")
            if state.get("inventorySha256") != drive_sha:
                raise IngestionError("Progress does not belong to this Drive inventory.")
            raw_sha = digest(raw_path) if inventory else None
            if inventory:
                if inventory.get("driveInventorySha256") != drive_sha or inventory.get("expectations") != expected:
                    raise IngestionError("Raw inventory does not belong to the reconciled Drive inventory.")
                receipts = {row["path"]: row for row in inventory["images"] + inventory["metadata"]}
                if len(receipts) != len(drive["files"]):
                    raise IngestionError("Raw inventory has missing or duplicate records.")
                for row in drive["files"]:
                    receipt = receipts.get(row["path"], {})
                    if any(receipt.get(key) != value for key, value in row.items()):
                        raise IngestionError(f"Raw inventory provenance/identity differs: {row['path']}")
            else:
                receipts = state["files"]
                report["errors"].append("Download is not finalized; rerun download_dataset before promotion.")
            issues, present = _tree_issues(originals, [row["path"] for row in drive["files"]])
            report["errors"].extend(issues)
            report["rawFilesPresent"] = len(present - set(METADATA_IDS))
            for name, drive_id in METADATA_IDS.items():
                receipt = {"path": name, **receipts.get(name, {})}
                if receipt.get("driveId") != drive_id:
                    raise IngestionError(f"Missing metadata transfer receipt: {name}")
                metadata_path = _verify_receipt(originals, receipt)
                if digest(metadata_path) != drive["metadataSha256"].get(name):
                    raise IngestionError(f"Source metadata SHA256 changed: {name}")
                _read_metadata(metadata_path)
            mapping = parse_mapping(_read_metadata(_source(originals, "rename_mapping.txt")))
            reconciliation = reconcile_inventory(drive["files"], mapping, expected)
            if reconciliation["status"] != "matched":
                raise IngestionError("Local originals no longer reconcile with the approved inventory.")
            for row in drive["files"]:
                if row["kind"] == "image" and row.get("source") != mapping[row["path"]]:
                    raise IngestionError(f"Mapping provenance differs from local inventory: {row['path']}")
            images, staging = _target(root, "images"), _target(root, "images.incoming")
            if images.exists() and staging.exists():
                raise IngestionError("Both images and images.incoming exist; refusing ambiguous derivatives.")
            if images.exists() and not raw.exists():
                raise IngestionError("Published images exist without immutable raw.")
            derivatives = images if images.exists() else staging
            complete_path = _target(root, "manifests/normalized-inventory.json")
            partial_path = _target(root, "manifests/normalized-inventory.partial.json")
            cache = None
            cache_path = complete_path if complete_path.exists() else partial_path
            if cache_path.exists():
                cache = _load_document(root, cache_path.relative_to(root).as_posix(), "normalized-image-inventory")
                if cache.get("driveInventorySha256") != drive_sha or cache.get("normalization") != NORMALIZATION:
                    raise IngestionError("Existing derivatives use another inventory/normalization environment.")
                if images.exists() and cache.get("rawInventorySha256") != raw_sha:
                    raise IngestionError("Published derivatives are bound to another raw inventory.")
            elif derivatives.exists() and any(derivatives.iterdir()):
                raise IngestionError("Nonempty derivatives have no local resume inventory.")
            derivatives.mkdir(exist_ok=True)
            cached = {row["originalPath"]: row for row in cache["images"]} if cache else {}
            normalized = {
                "schemaVersion": 1, "kind": "normalized-image-inventory", "status": "partial",
                "labelsVersion": taxonomy()["version"], "imagesRoot": "../images",
                "driveInventorySha256": drive_sha, "rawInventorySha256": raw_sha,
                "normalization": NORMALIZATION, "reviewStatus": "pending", "images": [],
            }
            save_partial()
            recovered_rows = []
            for row in drive["files"]:
                if row["kind"] != "image":
                    continue
                try:
                    receipt = receipts.get(row["path"], {})
                    if receipt.get("driveId") != row["driveId"] or (not inventory and receipt.get("status") != "complete"):
                        raise ContentError("missing_transfer_receipt", "No completed download receipt for this Drive ID.")
                    actual = {**row, "bytes": receipt.get("bytes"), "sha256": receipt.get("sha256")}
                    path = _verify_receipt(originals, actual)
                    report["recoveredImages"] += 1
                    recovered_rows.append({
                        "originalPath": row["path"], "originalSha256": actual["sha256"],
                        "classProposal": row["classProposal"],
                    })
                    relative = f"{row['classProposal']['slug']}/{Path(row['path']).name}.png"
                    destination = _target(derivatives, relative)
                    info = _cached_image(cached.get(row["path"]), actual, (destination, relative))
                    if info is None:
                        if images.exists():
                            raise ContentError("changed_derivative", "Published derivative is missing or changed; not overwritten.")
                        info = _normalize(path, destination)
                    normalized["images"].append({
                        "path": relative, "sha256": digest(destination), **info,
                        "driveId": row["driveId"],
                        "originalPath": row["path"], "originalSha256": actual["sha256"],
                        "classProposal": row["classProposal"], "source": row["source"],
                        "groupSuggestion": None, "reviewStatus": "pending",
                    })
                except Exception as error:
                    report["exclusions"].append({
                        "originalPath": row["path"], "reason": getattr(error, "reason", "invalid_or_missing_image"),
                        "detail": str(error), "status": "excluded-pending-recovery-or-review",
                    })
                    if not raw.exists() and row["path"] in state["files"]:
                        state["files"][row["path"]].update(status="invalid", error=str(error))
                if (len(normalized["images"]) + len(report["exclusions"])) % 25 == 0:
                    save_partial()
            rows = normalized["images"]
            report["exactDuplicates"] = _duplicate_clusters(recovered_rows, "originalSha256")
            report["normalizedDuplicates"] = _duplicate_clusters(rows, "sha256")
            report["pixelDuplicates"] = _duplicate_clusters(rows, "pixelSha256")
            report["duplicateAnalysis"] = "complete-for-verified-files"
            duplicate_hashes = {group["sha256"] for group in report["exactDuplicates"]}
            for row in rows:
                if row["originalSha256"] in duplicate_hashes:
                    row["groupSuggestion"] = {
                        "kind": "exact-original-sha256", "key": row["originalSha256"], "status": "pending",
                    }
            report["nearDuplicates"] = _near_duplicates(rows, near_duplicates, max_distance, max_pairs)
            report["imagesPerClass"] = dict(Counter(row["classProposal"]["slug"] for row in rows))
            report["sourceCounts"] = dict(Counter(row["source"]["id"] for row in rows))
            report["normalizedBytes"] = sum(_source(derivatives, row["path"]).stat().st_size for row in rows)
            derivative_issues, derivative_paths = _tree_issues(derivatives, [row["path"] for row in rows])
            report["errors"].extend(derivative_issues)
            if report["exclusions"] or report["errors"] or len(rows) != report["expectedImages"] or len(derivative_paths) != len(rows):
                report["status"] = "incomplete"
                save_partial()
                if state and not raw.exists():
                    state["status"] = "audit-incomplete"
                    _save_progress(root, state)
                return report
            # Raw bytes are never rewritten. Both renames remain on the dataset filesystem.
            save_partial()
            if not raw.exists():
                incoming.rename(raw)
            if not images.exists():
                staging.rename(images)
            normalized["status"] = "complete"
            report.update(status="complete", readyForAnnotation=True, rawInventorySha256=raw_sha)
            save_partial()
            _atomic_json(complete_path, normalized)
            state["status"] = "promoted"
            _save_progress(root, state)
            return report
        except (Exception, KeyboardInterrupt) as error:
            report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            report["errors"].append(str(error) or "interrupted")
            save_partial()
            if state is not None:
                _save_progress(root, state)
            return report
