"""A file-only bridge for human review in Label Studio Community Edition.

The sidecar, not an exported task's editable data, is the source of image identity.
Keep it with the source snapshot. No Label Studio package, API, or server is needed.
"""

import hashlib
import json
import math
import re
import shutil
import unicodedata
import uuid
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

from PIL import Image

from training.common import digest, reject_constant, source_path, taxonomy, write_json
from training.dataset import audit_manifest


IMAGE_URL_PREFIX = "/data/local-files/?d="
ROW_FIELDS = {"path", "sourceId", "proposedClassId", "sha256", "width", "height"}
SIDECAR_FIELDS = {
    "schemaVersion", "labelsVersion", "taxonomySha256", "configSha256",
    "inventorySha256", "bundleId", "tasks",
}
PRIVATE_SERVICE_HELP = """\
Use an isolated Python 3.11 environment with training/requirements-annotations.txt.
Set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true,
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT to the normalized image root (DATA/images)
ONLY, and LABEL_STUDIO_COLLECT_ANALYTICS=false. Start Label Studio with
--internal-host 127.0.0.1 (setting --host alone does not bind the listener).
Use an SSH loopback tunnel; keep service state/credentials private and outside Git.
In Label Studio CE, paste label-config.xml into the labeling interface and import
tasks.json. Neither class suggestions nor predictions are completed annotations.
Export full JSON, not JSON_MIN; export all tasks when possible. Missing tasks
remain pending. Never rotate/replace normalized images after annotation.
"""


class PendingReview(ValueError):
    """A task still requires an explicit human action."""


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def read_annotation_json(path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream, parse_constant=reject_constant, object_pairs_hook=unique_keys)


def _fields(value, required, optional, context):
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object.")
    missing, extra = required - value.keys(), value.keys() - required - optional
    if missing or extra:
        raise ValueError(f"{context}: missing fields {sorted(missing)}, unexpected fields {sorted(extra)}.")


def _text(value, context, multiline=False):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a nonblank string.")
    if any(unicodedata.category(char).startswith("C") and not (multiline and char in "\n\r\t")
           for char in value):
        raise ValueError(f"{context} contains control characters.")
    return unicodedata.normalize("NFC", value.strip())


def _integer(value, context, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}.")
    return value


def _number(value, context):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{context} must be a finite number, not a boolean/string.")
    return value


def _relative_path(value):
    _text(value, "Image path")
    relative = value.replace("\\", "/")
    if ":" in relative or any(part in ("", ".", "..") for part in relative.split("/")):
        raise ValueError(f"Image path must be relative without traversal: {value}")
    return relative


def _labels():
    value = taxonomy()
    labels = value["labels"]
    if (type(value.get("version")) is not int or value["version"] != 1
            or [row["id"] for row in labels] != list(range(10))
            or any(type(row["id"]) is not int for row in labels)
            or len({row["slug"] for row in labels}) != 10):
        raise ValueError("Expected shared/labels.json version 1 with ordered integer IDs 0..9.")
    return value


def annotation_config():
    """Generate labels in the shared taxonomy's ID order, never folder order."""
    view = ET.Element("View")
    ET.SubElement(view, "Header", value="Review manusia: anotasi setiap instance rimpang yang terlihat.")
    ET.SubElement(view, "Text", name="pathDisplay", value="$path")
    ET.SubElement(view, "Text", name="sourceDisplay", value="$sourceId")
    ET.SubElement(view, "Text", name="proposalDisplay", value="$proposedClass")
    ET.SubElement(view, "Header", value="Kelas sumber hanya SARAN, bukan anotasi. Jangan buat bbox otomatis seluruh gambar.")
    ET.SubElement(view, "Image", name="image", value="$image", zoom="true",
                  zoomControl="true", rotateControl="false")
    boxes = ET.SubElement(view, "RectangleLabels", name="bbox", toName="image",
                          choice="single", canRotate="false", strokeWidth="2")
    for row in _labels()["labels"]:
        ET.SubElement(boxes, "Label", value=row["slug"], model_index=str(row["id"]))
    ET.SubElement(view, "Header", value="include: semua instance diberi bbox; negative: tidak ada target; exclude: jangan gunakan.")
    choices = ET.SubElement(view, "Choices", name="decision", toName="image", choice="single-radio",
                            required="true", showInline="true")
    for decision in ("include", "negative", "exclude"):
        ET.SubElement(choices, "Choice", value=decision)
    ET.SubElement(view, "Header", value="groupId wajib untuk include/negative. Pakai ID global yang sama untuk spesimen/sesi, frame, crop, dan augmentasi terkait; jangan jadikan setiap nama file sebagai grup.")
    ET.SubElement(view, "TextArea", name="groupId", toName="image", rows="1",
                  maxSubmissions="1", editable="true", perRegion="false",
                  placeholder="ID kelompok yang sudah diverifikasi manusia")
    ET.SubElement(view, "Header", value="Untuk exclude, hapus bbox dan isi alasan. Untuk negative, pastikan benar-benar tidak ada target dan tidak ada bbox.")
    ET.SubElement(view, "TextArea", name="excludeReason", toName="image", rows="2",
                  maxSubmissions="1", editable="true", perRegion="false",
                  placeholder="Alasan eksklusi (wajib untuk exclude)")
    ET.SubElement(view, "Header", value="Centang reviewed hanya setelah kelas, semua bbox, keputusan, sumber, dan kelompok selesai diperiksa; lalu Submit/Update. Draft atau Skip bukan review selesai.")
    review = ET.SubElement(view, "Choices", name="reviewed", toName="image", choice="multiple",
                           required="true", showInline="true")
    ET.SubElement(review, "Choice", value="reviewed")
    ET.indent(view, space="  ")
    return ET.tostring(view, encoding="unicode") + "\n"


def _inventory_rows(inventory):
    _fields(inventory, {"schemaVersion", "labelsVersion", "images"}, set(), "Inventory")
    if (type(inventory["schemaVersion"]) is not int or inventory["schemaVersion"] != 1
            or type(inventory["labelsVersion"]) is not int
            or inventory["labelsVersion"] != _labels()["version"]):
        raise ValueError("Inventory schemaVersion/labelsVersion must both be 1.")
    if not isinstance(inventory["images"], list) or not inventory["images"]:
        raise ValueError("Inventory images must be a nonempty list.")
    rows, paths, hashes = [], set(), {}
    for index, item in enumerate(inventory["images"]):
        _fields(item, ROW_FIELDS, {"license"}, f"Inventory images[{index}]")
        row = dict(item)
        row["path"] = _relative_path(row["path"])
        row["sourceId"] = _text(row["sourceId"], "sourceId")
        if _integer(row["proposedClassId"], "proposedClassId") not in range(10):
            raise ValueError("proposedClassId must follow shared labels IDs 0..9.")
        if not isinstance(row["sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", row["sha256"]):
            raise ValueError(f"Invalid sha256: {row['path']}")
        for dimension in ("width", "height"):
            _integer(row[dimension], dimension, 16)
        if "license" in row and not isinstance(row["license"], str):
            raise ValueError("Optional license must be a string.")
        if row["path"] in paths:
            raise ValueError(f"Duplicate inventory path: {row['path']}")
        if row["sha256"] in hashes:
            raise ValueError(f"Exact duplicate inventory images: {hashes[row['sha256']]} and {row['path']}; reconcile before annotation.")
        paths.add(row["path"])
        hashes[row["sha256"]] = row["path"]
        rows.append(row)
    return sorted(rows, key=lambda row: row["path"])


def adapt_normalized_inventory(document):
    """Convert the ingestion inventory to annotation rows.

    Byte-identical normalized images are annotated once; aliases are reported
    so they cannot leak into different splits.
    """
    if not isinstance(document, dict) or document.get("kind") != "normalized-image-inventory":
        return document, None
    if document.get("status") != "complete" or document.get("schemaVersion") != 1:
        raise ValueError("Normalized inventory must be complete schemaVersion 1; finish audit_raw_dataset first.")
    if not isinstance(document.get("images"), list) or not document["images"]:
        raise ValueError("Normalized inventory images must be a nonempty list.")
    groups = defaultdict(list)
    for index, item in enumerate(document["images"]):
        try:
            row = {
                "path": item["path"], "sourceId": "drive:" + _text(item["driveId"], "driveId"),
                "proposedClassId": item["classProposal"]["classId"], "sha256": item["sha256"],
                "width": item["width"], "height": item["height"],
            }
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Normalized inventory images[{index}] is missing {exc}.") from exc
        groups[row["sha256"]].append((row, item))
    images, clusters = [], []
    for sha256 in sorted(groups):
        members = sorted(groups[sha256], key=lambda pair: pair[0]["path"])
        images.append(members[0][0])
        if len(members) > 1:
            clusters.append({
                "sha256": sha256, "representative": members[0][0]["path"],
                "excludedAliases": [pair[0]["path"] for pair in members[1:]],
                "originalPaths": [pair[1].get("originalPath") for pair in members],
                "classConflict": len({pair[0]["proposedClassId"] for pair in members}) > 1,
            })
    report = {
        "schemaVersion": 1, "kind": "annotation-duplicate-report",
        "inputImages": len(document["images"]), "annotationTasks": len(images),
        "excludedAliases": sum(len(row["excludedAliases"]) for row in clusters), "clusters": clusters,
    }
    return {"schemaVersion": 1, "labelsVersion": document.get("labelsVersion"), "images": images}, report


def _verify_image(row, image_root):
    path = source_path(image_root, row["path"])
    if digest(path) != row["sha256"]:
        raise ValueError(f"Image checksum mismatch: {row['path']}; annotations are stale.")
    try:
        with Image.open(path) as image:
            if image.size != (row["width"], row["height"]):
                raise ValueError(f"Image dimensions changed: {row['path']}")
            if min(image.size) < 16:
                raise ValueError(f"Image too small: {row['path']}")
            if image.getexif().get(274, 1) not in (1, None):
                raise ValueError(f"Normalize EXIF orientation BEFORE annotation: {row['path']}")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError(f"Normalize animated/multipage images BEFORE annotation: {row['path']}")
            image.load()
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Cannot decode normalized image {row['path']}: {exc}") from exc
    return path


def _task_key(row):
    return _fingerprint(["rimpang-annotation-v1", row["path"], row["sourceId"], row["sha256"]])


def _sidecar(rows):
    labels = _labels()
    value = {
        "schemaVersion": 1, "labelsVersion": labels["version"],
        "taxonomySha256": _fingerprint(labels),
        "configSha256": hashlib.sha256(annotation_config().encode("utf-8")).hexdigest(),
        "inventorySha256": _fingerprint({"schemaVersion": 1, "labelsVersion": labels["version"], "images": rows}),
        "tasks": [{**row, "taskKey": _task_key(row)} for row in rows],
    }
    return {**value, "bundleId": _fingerprint(value)}


def _task_data(row, sidecar):
    label = _labels()["labels"][row["proposedClassId"]]
    return {
        "image": IMAGE_URL_PREFIX + quote(row["path"], safe="/"),
        "taskKey": _task_key(row), "bundleId": sidecar["bundleId"],
        "path": row["path"], "sourceId": row["sourceId"], "sha256": row["sha256"],
        "width": row["width"], "height": row["height"], "proposedClassId": row["proposedClassId"],
        "proposedClass": f"SARAN SAJA — {label['id']}: {label['name']} ({label['slug']})",
    }


def prepare_annotation_tasks(inventory, image_root):
    """Return (tasks, XML config, authoritative sidecar); never invent annotations."""
    rows = _inventory_rows(inventory)
    seen = set()
    for row in rows:
        path = _verify_image(row, image_root)
        if path in seen:
            raise ValueError(f"Same image resolved more than once: {row['path']}")
        seen.add(path)
    sidecar = _sidecar(rows)
    return ([{"data": _task_data(row, sidecar)} for row in rows], annotation_config(), sidecar)


def _sidecar_rows(sidecar):
    _fields(sidecar, SIDECAR_FIELDS, set(), "Sidecar")
    if not isinstance(sidecar["tasks"], list):
        raise ValueError("Sidecar tasks must be a list.")
    rows = []
    for item in sidecar["tasks"]:
        _fields(item, ROW_FIELDS | {"taskKey"}, {"license"}, "Sidecar task")
        row = {key: value for key, value in item.items() if key != "taskKey"}
        if item["taskKey"] != _task_key(row):
            raise ValueError("Sidecar taskKey mismatch; use the original authoritative sidecar.")
        rows.append(row)
    rows = _inventory_rows({"schemaVersion": sidecar["schemaVersion"],
                            "labelsVersion": sidecar["labelsVersion"], "images": rows})
    if _canonical(sidecar) != _canonical(_sidecar(rows)):
        raise ValueError("Sidecar checksum/taxonomy/config mismatch; use the original authoritative sidecar.")
    return rows


def _ls_id(value, context):
    if type(value) is int and value > 0:
        return value
    if isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value):
        return int(value)
    raise ValueError(f"{context} must be a positive Label Studio ID.")


def _state_flag(value, name):
    if name not in value:
        return False
    if type(value[name]) is not bool:
        raise ValueError(f"{name} must be a boolean.")
    return value[name]


def _choices(result, name, permitted):
    if result is None:
        raise PendingReview(f"Explicit {name} decision is missing.")
    _fields(result["value"], {"choices"}, set(), name)
    choices = result["value"]["choices"]
    if choices == []:
        raise PendingReview(f"Explicit {name} decision is missing.")
    if (not isinstance(choices, list) or len(choices) != 1
            or not isinstance(choices[0], str) or choices[0] not in permitted):
        raise ValueError(f"Invalid {name} choice.")
    return choices[0]


def _textarea(result, name):
    if result is None:
        return ""
    _fields(result["value"], {"text"}, set(), name)
    text = result["value"]["text"]
    if text == []:
        return ""
    if not isinstance(text, list) or len(text) != 1 or not isinstance(text[0], str):
        raise ValueError(f"{name} must be one global TextArea submission.")
    if not text[0].strip():
        return ""
    return _text(text[0], name, multiline=name == "excludeReason")


def _rectangle(result, row):
    for key, expected in (("original_width", row["width"]), ("original_height", row["height"])):
        if _number(result.get(key), key) != expected:
            raise ValueError(f"Bbox {key} does not match the normalized image.")
    if _number(result.get("image_rotation"), "image_rotation") != 0:
        raise ValueError("Image rotation must be zero.")
    value = result["value"]
    _fields(value, {"x", "y", "width", "height", "rectanglelabels"}, {"rotation"}, "Bbox value")
    # Label Studio omits value.rotation in some zero-rotation results.
    if _number(value.get("rotation", 0), "bbox rotation") != 0:
        raise ValueError("Bbox rotation must be zero.")
    labels = value["rectanglelabels"]
    by_slug = {label["slug"]: label["id"] for label in _labels()["labels"]}
    if not isinstance(labels, list) or len(labels) != 1 or not isinstance(labels[0], str) or labels[0] not in by_slug:
        raise ValueError("Each bbox must have exactly one shared taxonomy slug.")
    coordinates = [_number(value.get(key), f"bbox {key}") for key in ("x", "y", "width", "height")]
    if any(number < 0 or number > 100 for number in coordinates):
        raise ValueError("Bbox percentages must be within 0..100.")
    x, y, width, height = [number / 100 + 0.0 for number in coordinates]
    if width <= 0 or height <= 0 or x + width > 1 + 1e-9 or y + height > 1 + 1e-9:
        raise ValueError("Bbox must have positive size and stay within the image.")
    return {"classId": by_slug[labels[0]], "bbox": [x, y, width, height]}


def _review_annotation(annotation, row, task_id):
    annotation_id = _ls_id(annotation.get("id"), "Annotation id")
    if "task" in annotation and _ls_id(annotation["task"], "Annotation task") != task_id:
        raise ValueError("Annotation belongs to a different task.")
    reviewer = annotation.get("completed_by")
    if isinstance(reviewer, dict):
        reviewer = reviewer.get("id")
    reviewer = _ls_id(reviewer, "Human completed_by")
    results = annotation.get("result")
    if not isinstance(results, list):
        raise ValueError("Annotation result must be a list.")
    controls, objects, result_ids, boxes = {}, [], set(), set()
    kinds = {"bbox": "rectanglelabels", "decision": "choices", "reviewed": "choices",
             "groupId": "textarea", "excludeReason": "textarea"}
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("Each annotation result must be an object.")
        name, result_id = result.get("from_name"), result.get("id")
        if (not isinstance(name, str) or name not in kinds or result.get("type") != kinds[name]
                or result.get("to_name") != "image"):
            raise ValueError("Unexpected result type/control/target; use the generated label config.")
        if not isinstance(result_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", result_id):
            raise ValueError("Result is missing a valid region/control ID.")
        if result_id in result_ids:
            raise ValueError("Duplicate result ID or per-region global review field.")
        result_ids.add(result_id)
        if not isinstance(result.get("value"), dict):
            raise ValueError("Result value must be an object.")
        if result.get("parentID") or result.get("parent_id"):
            raise ValueError("Review fields and boxes must not be nested/per-region.")
        if name == "bbox":
            obj = _rectangle(result, row)
            key = tuple(obj["bbox"])
            if key in boxes:
                raise ValueError("Duplicate bounding box; review the instances.")
            boxes.add(key)
            objects.append(obj)
        else:
            if name in controls:
                raise ValueError(f"Multiple global {name} results.")
            controls[name] = result
    _choices(controls.get("reviewed"), "reviewed", {"reviewed"})
    if controls["reviewed"].get("origin", "manual") != "manual":
        raise PendingReview("The reviewed checkbox must be a human decision, not a prediction.")
    decision = _choices(controls.get("decision"), "decision", {"include", "negative", "exclude"})
    group = _textarea(controls.get("groupId"), "groupId")
    reason = _textarea(controls.get("excludeReason"), "excludeReason")
    if decision == "exclude":
        if not reason:
            raise PendingReview("An excluded image requires a human exclusion reason.")
        if objects:
            raise ValueError("An excluded image must not retain bounding boxes.")
    else:
        if not group:
            raise PendingReview("Included/negative images require a verified global groupId.")
        if reason:
            raise ValueError("excludeReason is only valid with the exclude decision.")
        if decision == "include" and not objects:
            raise PendingReview("include requires instance bboxes; an empty task is not a negative.")
        if decision == "negative" and objects:
            raise ValueError("An explicit negative must not contain target bboxes.")
    return {
        "decision": decision, "groupId": group if decision != "exclude" else "",
        "reason": reason, "objects": sorted(objects, key=lambda obj: (obj["classId"], *obj["bbox"])),
    }, annotation_id, reviewer


def _review_task(task, row, task_id):
    for flag in ("was_cancelled", "skipped", "is_skipped", "is_draft", "draft"):
        if _state_flag(task, flag):
            raise PendingReview(f"Task is currently {flag}.")
    if "is_labeled" in task and not _state_flag(task, "is_labeled"):
        raise PendingReview("Task is not currently labeled.")
    for key in ("annotations", "drafts", "predictions"):
        if key in task and not isinstance(task[key], list):
            raise ValueError(f"Task {key} must be a list.")
    if task.get("drafts"):
        raise PendingReview("Task has an unsubmitted draft; submit or remove it before export.")
    completed, ids, reviewers = [], [], []
    annotation_ids = set()
    for annotation in task.get("annotations", []):
        if not isinstance(annotation, dict):
            raise ValueError("Annotation must be an object.")
        annotation_id = _ls_id(annotation.get("id"), "Annotation id")
        if annotation_id in annotation_ids:
            raise ValueError("Duplicate annotation ID.")
        annotation_ids.add(annotation_id)
        if "was_cancelled" not in annotation:
            raise ValueError("Annotation is missing explicit was_cancelled completion status.")
        if _state_flag(annotation, "was_cancelled"):
            continue
        for flag in ("skipped", "is_skipped", "is_draft", "draft"):
            if _state_flag(annotation, flag):
                raise PendingReview(f"Annotation is currently {flag}.")
        review, annotation_id, reviewer = _review_annotation(annotation, row, task_id)
        completed.append(review)
        ids.append(annotation_id)
        reviewers.append(reviewer)
    if not completed:
        raise PendingReview("No completed human annotation (drafts, predictions and cancelled work do not count).")
    if any(value != completed[0] for value in completed[1:]):
        raise ValueError("Conflicting completed annotations; resolve the disagreement in Label Studio and export again.")
    return completed[0], sorted(ids), sorted(set(reviewers))


def invalid_review_report(message):
    return {
        "schemaVersion": 1, "status": "invalid",
        "counts": {"expected": 0, "exported": 0, "accepted": 0, "included": 0, "negative": 0,
                   "excluded": 0, "pending": 0, "errorTasks": 0, "errors": 1},
        "accepted": [], "excluded": [], "pending": [],
        "errors": [{"status": "error", "reason": str(message)}],
    }


def import_annotation_tasks(export, sidecar, image_root):
    """Return (manifest or None, report). A partial review never yields a manifest."""
    rows = _sidecar_rows(sidecar)
    if not isinstance(export, list):
        raise ValueError("Expected a full Label Studio JSON export list, not JSON_MIN or a results object.")
    expected = {_task_key(row): row for row in rows}
    by_key, id_counts, annotation_owners = defaultdict(list), Counter(), defaultdict(set)
    report = {
        "schemaVersion": 1, "labelsVersion": 1, "bundleId": sidecar["bundleId"],
        "status": "incomplete", "accepted": [], "excluded": [], "pending": [], "errors": [],
        "groupingPolicy": "Reviewer-supplied global groupId; never inferred from filenames or sources. Run audit_manifest/assign_splits before training.",
    }
    for index, task in enumerate(export):
        if not isinstance(task, dict) or not isinstance(task.get("data"), dict):
            report["errors"].append({"status": "error", "exportIndex": index, "reason": "Task must contain a data object."})
            continue
        key = task["data"].get("taskKey")
        if not isinstance(key, str) or key not in expected:
            report["errors"].append({"status": "error", "exportIndex": index, "reason": "Missing/unknown taskKey; cannot bind task to the sidecar."})
            continue
        by_key[key].append(task)
        try:
            id_counts[_ls_id(task.get("id"), "Task id")] += 1
        except ValueError:
            pass
        annotations = task.get("annotations", [])
        if isinstance(annotations, list):
            for annotation in annotations:
                if isinstance(annotation, dict):
                    try:
                        annotation_owners[_ls_id(annotation.get("id"), "Annotation id")].add(key)
                    except ValueError:
                        pass
    accepted_rows, seen_paths, error_tasks = [], set(), 0
    for key, row in expected.items():
        identity = {"taskKey": key, "path": row["path"], "sourceId": row["sourceId"]}
        try:
            path = _verify_image(row, image_root)
            if path in seen_paths:
                raise ValueError("Same image resolved more than once.")
            seen_paths.add(path)
            candidates = by_key.get(key, [])
            if not candidates:
                raise PendingReview("Task is missing from the export.")
            if len(candidates) != 1:
                raise ValueError("Duplicate exported taskKey; export each sidecar task exactly once.")
            task = candidates[0]
            task_id = _ls_id(task.get("id"), "Task id")
            identity["taskId"] = task_id
            if id_counts[task_id] != 1:
                raise ValueError("Duplicate Label Studio task ID.")
            if _canonical(task["data"]) != _canonical(_task_data(row, sidecar)):
                raise ValueError("Export task data differs from the authoritative sidecar (path/hash/proposal/dimensions/bundle).")
            annotations = task.get("annotations", [])
            if isinstance(annotations, list):
                for annotation in annotations:
                    if isinstance(annotation, dict):
                        annotation_id = _ls_id(annotation.get("id"), "Annotation id")
                        if len(annotation_owners[annotation_id]) > 1:
                            raise ValueError("Annotation ID is reused across different tasks.")
            review, annotation_ids, reviewers = _review_task(task, row, task_id)
            identity.update(annotationIds=annotation_ids, reviewerIds=reviewers, decision=review["decision"])
            if review["decision"] == "exclude":
                report["excluded"].append({**identity, "status": "excluded", "reason": review["reason"]})
            else:
                record = {name: row[name] for name in ("path", "sourceId", "sha256")}
                if "license" in row:
                    record["license"] = row["license"]
                accepted_rows.append({**record, "groupId": review["groupId"], "objects": review["objects"]})
                report["accepted"].append({**identity, "status": "accepted", "groupId": review["groupId"],
                                           "objects": len(review["objects"])})
        except PendingReview as exc:
            report["pending"].append({**identity, "status": "pending", "reason": str(exc)})
        except (ValueError, OSError) as exc:
            error_tasks += 1
            report["errors"].append({**identity, "status": "error", "reason": str(exc)})
    group_spellings = defaultdict(set)
    for row in accepted_rows:
        group_spellings[unicodedata.normalize("NFKC", row["groupId"]).casefold()].add(row["groupId"])
    for spellings in group_spellings.values():
        if len(spellings) > 1:
            report["errors"].append({"status": "error", "reason": f"Inconsistent global groupId spelling: {sorted(spellings)}"})
    manifest = None
    if not report["pending"] and not report["errors"]:
        if not accepted_rows:
            report["errors"].append({"status": "error", "reason": "All tasks were excluded; a master manifest requires at least one included image."})
        else:
            candidate = {"schemaVersion": 1, "labelsVersion": 1, "images": accepted_rows}
            try:
                audited, report["audit"] = audit_manifest(candidate, image_root)
                manifest = {**candidate, "images": audited}
                report["status"] = "ready"
            except (ValueError, OSError) as exc:
                report["errors"].append({"status": "error", "reason": f"Manifest audit failed: {exc}"})
    if report["errors"]:
        report["status"] = "invalid"
    report["counts"] = {
        "expected": len(rows), "exported": len(export), "accepted": len(report["accepted"]),
        "included": sum(item["decision"] == "include" for item in report["accepted"]),
        "negative": sum(item["decision"] == "negative" for item in report["accepted"]),
        "excluded": len(report["excluded"]), "pending": len(report["pending"]),
        "errorTasks": error_tasks, "errors": len(report["errors"]),
    }
    return manifest, report


def publish_annotation_files(output, image_root, files):
    """Publish a new versioned result outside the local-file serving root."""
    destination, root = Path(output).absolute(), Path(image_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise ValueError("Output already exists; use a new versioned directory. Nothing is overwritten.")
    if destination.resolve().is_relative_to(root):
        raise ValueError("Annotation metadata output must be outside the served image root.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.staging"
    stage.mkdir()
    try:
        for name, value in files.items():
            if Path(name).name != name:
                raise ValueError("Output file name must not contain directories.")
            if isinstance(value, str):
                (stage / name).write_text(value, encoding="utf-8", newline="\n")
            else:
                write_json(stage / name, value)
        if destination.exists() or destination.is_symlink():
            raise ValueError("Output appeared during processing; refusing to overwrite it.")
        stage.rename(destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return destination
