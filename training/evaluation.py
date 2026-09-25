"""Static-frame evaluation; temporal tracking and browser interpolation need separate checks."""

import hashlib
import io
import itertools
import json
import math
import os
import re
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from training.common import SPLITS, digest, read_json, taxonomy


PROTOCOL_VERSION = "rimpang-static-frame-v1"
CLASS_COUNT = 10
MAX_OBJECTS = 5
NMS_CANDIDATES = 300
NMS_LIMIT = 30
MATCH_IOU = 0.5
THRESHOLD_NAMES = (
    "detectorScoreThreshold", "iouThreshold", "classifierScoreThreshold", "minMargin",
)
DEFAULT_GRID = {
    "detectorScoreThreshold": [0.15, 0.25, 0.35, 0.5],
    "iouThreshold": [0.3, 0.45, 0.6],
    "classifierScoreThreshold": [0.5, 0.65, 0.8],
    "minMargin": [0.0, 0.1, 0.2],
}
LIMITATIONS = [
    "Static independent images only: browser tracking, score smoothing, two-frame confirmation, "
    "temporal continuity, latency and physical-phone performance are not evaluated.",
    "Pillow BILINEAR RGB resizing is not claimed equivalent to canvas high-quality interpolation. "
    "Real-image PyTorch/ONNX numeric checks and browser golden-image geometry/tensor/output checks "
    "are required before activation; random-tensor export checks alone are insufficient.",
    "Group-disjointness is checked against declared groupId values, not proof of biological or "
    "capture-session independence. Human annotation/provenance review remains required.",
    "objects: [] measures a combined annotated negative/unknown-frame false-accept rate. The "
    "schema cannot distinguish unknown rhizomes from non-rhizomes or measure unseen populations.",
    "No occlusion, capture-domain or temporal labels exist in the master schema. Small holdout "
    "group counts and missing scenario strata limit generalization; no accuracy target is implied.",
]


def json_fingerprint(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()


def atomic_write(path, writer):
    """Commit a complete neighboring file, retaining the previous file on write failure."""
    path = Path(path)
    pending = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    try:
        with pending.open("xb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def atomic_json(path, value):
    payload = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    atomic_write(path, lambda stream: stream.write(payload))


def atomic_copy(source, destination):
    with Path(source).open("rb") as stream:
        atomic_write(destination, lambda output: shutil.copyfileobj(stream, output))


def finite_number(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} is outside [{minimum}, {maximum}].")
    return float(value)


def class_id(value):
    if type(value) is not int or not 0 <= value < CLASS_COUNT:
        raise ValueError("classId must be an integer in the existing 0..9 taxonomy.")
    return value


def checked_box(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("A box must be normalized top-left x,y,width,height.")
    x, y, width, height = (finite_number(v, "bbox coordinate", 0, 1) for v in value)
    if width <= 0 or height <= 0 or x + width > 1 + 1e-9 or y + height > 1 + 1e-9:
        raise ValueError("Invalid normalized box extent.")
    return x, y, width, height


def checked_logits(values):
    if not isinstance(values, (list, tuple)) or len(values) != CLASS_COUNT:
        raise ValueError("Classifier must produce exactly 10 finite logits, in shared label order.")
    return tuple(finite_number(value, "logit") for value in values)


def checked_thresholds(value):
    if not isinstance(value, dict) or set(value) != set(THRESHOLD_NAMES):
        raise ValueError(f"Thresholds must contain exactly {THRESHOLD_NAMES}.")
    return {key: finite_number(value[key], key, 0, 1) for key in THRESHOLD_NAMES}


def protocol():
    return {
        "version": PROTOCOL_VERSION,
        "selectionSplit": "valid",
        "testPolicy": "Evaluate once with persisted frozen thresholds; never select on test.",
        "grid": {key: list(values) for key, values in DEFAULT_GRID.items()},
        "objective": "maximum endToEnd.macroF1 over all 10 fixed classes",
        "tieBreak": [
            "maximum accepted correctly classified GT-object coverage",
            "minimum accepted negative/unknown frames (when present)",
            "first candidate in declared detector/iou/classifier/margin grid order",
        ],
        "matching": {
            "iouThreshold": MATCH_IOU, "classAgnostic": True,
            "method": "descending detector score, highest-IoU unused GT, GT-index tie break",
        },
        "browserContract": {
            "detectorInputSize": 416, "classifierInputSize": 224,
            "detectorOutput": "1x5xN", "classifierOutput": "1x10 logits",
            "dtype": "float32", "layout": "RGB NCHW", "padValue": 114,
            "detectorMean": [0, 0, 0], "detectorStd": [1, 1, 1],
            "classifierMean": [0.5, 0.5, 0.5], "classifierStd": [0.5, 0.5, 0.5],
            "cropMargin": 0.1, "nmsCandidateLimit": NMS_CANDIDATES,
            "nmsLimit": NMS_LIMIT, "maxObjects": MAX_OBJECTS,
            "rounding": "positive Math.round letterbox; floor/ceil padded crop",
            "resampling": "Pillow BILINEAR; canvas golden-image parity still required",
        },
        "ultralytics": {
            "confidenceFloor": 0.001, "maxDetections": 300,
            "note": "Separate standard AP evaluation on prepared JPEGs using Ultralytics "
                    "preprocessing. Its P/R operating point is not the deployed frozen threshold. "
                    "Browser-style detector P/R is reported separately on original normalized images.",
        },
        "eligibilityMinimum": {
            "allClassesIn": list(SPLITS), "negativeFramesIn": ["valid", "test"],
            "objectCountsInEachHoldout": [2, 3, 4, 5],
            "note": "At least one actual frame/group per stratum is a technical minimum, "
                    "not statistical sufficiency or production quality approval.",
        },
    }


def inspect_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get("schemaVersion") != 1 or manifest.get("labelsVersion") != 1:
        raise ValueError("Require master manifest schema/labels version 1.")
    rows = manifest.get("images")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Prepared manifest must contain images.")
    paths, hashes, group_splits = set(), set(), {}
    slugs = [label["slug"] for label in taxonomy()["labels"]]
    summaries, unmet = {}, []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Manifest image must be an object.")
        path, group, source, split = (row.get(key) for key in ("path", "groupId", "sourceId", "split"))
        if not all(isinstance(value, str) and value.strip() for value in (path, group, source)):
            raise ValueError("Every prepared image requires nonblank path/groupId/sourceId.")
        if group != group.strip() or source != source.strip():
            raise ValueError("Prepared group/source identities must already be normalized.")
        if split not in SPLITS:
            raise ValueError("Every prepared image needs an explicit train/valid/test split.")
        portable_path = path.replace("\\", "/")
        if ":" in portable_path or portable_path.startswith("/") or ".." in portable_path.split("/"):
            raise ValueError(f"Unsafe image path: {path}")
        if portable_path in paths:
            raise ValueError(f"Duplicate manifest path: {path}")
        paths.add(portable_path)
        sha = row.get("sha256")
        if not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{64}", sha):
            raise ValueError(f"Prepared image lacks a SHA256: {path}")
        if sha in hashes:
            raise ValueError(f"Duplicate image checksum in prepared manifest: {path}")
        hashes.add(sha)
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"Group crosses splits: {group}")
        group_splits[group] = split
        if not isinstance(row.get("objects"), list):
            raise ValueError(f"Missing explicit reviewed objects list: {path}")
        for obj in row["objects"]:
            if not isinstance(obj, dict):
                raise ValueError("Ground-truth object must be a mapping.")
            class_id(obj.get("classId"))
            checked_box(obj.get("bbox"))
    for split in SPLITS:
        subset = [row for row in rows if row["split"] == split]
        counts = Counter(obj["classId"] for row in subset for obj in row["objects"])
        groups_by_class = {
            slug: len({row["groupId"] for row in subset
                       if any(obj["classId"] == index for obj in row["objects"])})
            for index, slug in enumerate(slugs)
        }
        cardinality = Counter(len(row["objects"]) for row in subset)
        summary = {
            "images": len(subset), "groups": len({row["groupId"] for row in subset}),
            "sources": dict(sorted(Counter(row["sourceId"] for row in subset).items())),
            "objects": sum(counts.values()),
            "objectsPerClass": {slug: counts[index] for index, slug in enumerate(slugs)},
            "groupsPerClass": groups_by_class,
            "negativeImages": cardinality[0],
            "negativeGroups": len({row["groupId"] for row in subset if not row["objects"]}),
            "imagesByObjectCount": {str(count): cardinality[count] for count in range(6)},
            "imagesAboveObjectCap": sum(count for size, count in cardinality.items() if size > MAX_OBJECTS),
            "groupsByObjectCount": {
                str(count): len({row["groupId"] for row in subset if len(row["objects"]) == count})
                for count in range(6)
            },
            "mixedClassImages": sum(len({obj["classId"] for obj in row["objects"]}) > 1 for row in subset),
        }
        summaries[split] = summary
        missing = [slug for slug, count in summary["objectsPerClass"].items() if count == 0]
        if missing:
            unmet.append(f"{split}: add reviewed independent groups covering missing classes: {', '.join(missing)}.")
        if split != "train":
            if not cardinality[0]:
                unmet.append(f"{split}: add reviewed negative/unknown frames (explicit objects: []); false accept is unmeasured.")
            absent_counts = [str(count) for count in range(2, 6) if not cardinality[count]]
            if absent_counts:
                unmet.append(f"{split}: add real reviewed multiobject frames with these instance counts: {', '.join(absent_counts)}.")
    return {
        "groupDisjoint": True, "totalImages": len(rows), "totalGroups": len(group_splits),
        "splits": summaries, "eligible": not unmet, "unmetGates": unmet,
    }


def prepared_dataset_binding(prepared):
    """Bind the manifest AND actual generated training files, not the audit summary alone."""
    prepared = Path(prepared).resolve()
    manifest_path, report_path = prepared / "manifest.json", prepared / "report.json"
    manifest, report = read_json(manifest_path), read_json(report_path)
    inspection = inspect_manifest(manifest)
    margin = finite_number(report.get("cropMargin"), "prepared cropMargin", 0, 0.5)
    labels = taxonomy()["labels"]
    expected = {prepared / "detector" / "data.yaml"}
    for index, row in enumerate(manifest["images"]):
        split, stem = row["split"], f"{index:06d}_{row['sha256'][:12]}"
        expected.add(prepared / "detector" / "images" / split / f"{stem}.jpg")
        label_file = prepared / "detector" / "labels" / split / f"{stem}.txt"
        expected.add(label_file)
        actual_labels = label_file.read_text(encoding="ascii").splitlines()
        if len(actual_labels) != len(row["objects"]):
            raise ValueError(f"Prepared YOLO instance count differs from manifest: {label_file}")
        for instance, (obj, line) in enumerate(zip(row["objects"], actual_labels)):
            tokens = line.split()
            x, y, width, height = obj["bbox"]
            target = [x + width / 2, y + height / 2, width, height]
            if len(tokens) != 5 or tokens[0] != "0":
                raise ValueError(f"Prepared detector must use one generic rimpang class: {label_file}")
            coordinates = [float(token) for token in tokens[1:]]
            if any(not math.isfinite(value) or abs(value - wanted) > 1e-8
                   for value, wanted in zip(coordinates, target)):
                raise ValueError(f"Prepared detector boxes differ from manifest: {label_file}")
            expected.add(prepared / "classifier" / split / labels[obj["classId"]]["slug"] / f"{stem}_{instance}.jpg")
    label_caches = {prepared / "detector" / "labels" / f"{split}.cache" for split in SPLITS}
    actual = {file for folder in ("detector", "classifier") for file in (prepared / folder).rglob("*")
              if file.is_file() and file not in label_caches}
    if actual != expected:
        difference = sorted(str(file.relative_to(prepared)) for file in actual ^ expected)
        raise ValueError(f"Prepared inventory differs from manifest: {difference[:10]}")
    entries = []
    for file in sorted(expected):
        if file.is_symlink() or not file.resolve().is_relative_to(prepared) or file.stat().st_size == 0:
            # Empty YOLO labels are the required representation for reviewed negative frames.
            if file.suffix != ".txt" or file.is_symlink() or not file.resolve().is_relative_to(prepared):
                raise ValueError(f"Missing, empty or unsafe prepared file: {file}")
        entries.append({"path": file.relative_to(prepared).as_posix(), "sha256": digest(file)})
    data_yaml = read_json(prepared / "detector" / "data.yaml")
    if (data_yaml.get("names") != ["rimpang"] or data_yaml.get("train") != "images/train"
            or data_yaml.get("val") != "images/valid" or data_yaml.get("test") != "images/test"
            or Path(data_yaml.get("path", "")).resolve() != prepared / "detector"):
        raise ValueError("Prepared data.yaml must point to this exact one-class, grouped prepared dataset.")
    identity = {
        "preparedManifestSha256": digest(manifest_path), "preparedReportSha256": digest(report_path),
        "preparedFilesSha256": json_fingerprint(entries), "preparedFileCount": len(entries),
        "cropMargin": margin,
    }
    return {**identity, "datasetFingerprint": json_fingerprint(identity)}, manifest, inspection


@dataclass(frozen=True)
class Detection:
    box: tuple
    score: float
    anchor: int
    logits: tuple | None = None

    def __post_init__(self):
        object.__setattr__(self, "box", checked_box(self.box))
        object.__setattr__(self, "score", finite_number(self.score, "detector score", 0, 1))
        if type(self.anchor) is not int or self.anchor < 0:
            raise ValueError("Detection anchor must be a nonnegative integer.")
        if self.logits is not None:
            object.__setattr__(self, "logits", checked_logits(self.logits))


@dataclass(frozen=True)
class GroundTruth:
    box: tuple
    class_id: int
    logits: tuple

    def __post_init__(self):
        object.__setattr__(self, "box", checked_box(self.box))
        class_id(self.class_id)
        object.__setattr__(self, "logits", checked_logits(self.logits))


@dataclass(frozen=True)
class FramePrediction:
    path: str
    group_id: str
    split: str
    truth: tuple
    detections: tuple
    candidate_count: int

    def __post_init__(self):
        if not self.path or not self.group_id or self.split not in ("valid", "test"):
            raise ValueError("Prediction must identify a valid/test image and its group.")
        if not all(isinstance(obj, GroundTruth) for obj in self.truth):
            raise ValueError("Invalid ground-truth predictions.")
        if not all(isinstance(obj, Detection) for obj in self.detections):
            raise ValueError("Invalid detector predictions.")
        if len({obj.anchor for obj in self.detections}) != len(self.detections):
            raise ValueError("Detector anchor indices must be unique within an image.")
        if type(self.candidate_count) is not int or self.candidate_count < len(self.detections):
            raise ValueError("Candidate count must include every stored proposal.")
        if len(self.detections) > NMS_CANDIDATES:
            raise ValueError("Store at most the browser's 300 pre-NMS candidates.")


def letterbox_geometry(width, height, size=416):
    if any(type(value) is not int or value <= 0 for value in (width, height, size)):
        raise ValueError("Image dimensions and tensor size must be positive integers.")
    ratio = min(size / width, size / height)
    resized_width = max(1, math.floor(width * ratio + 0.5))
    resized_height = max(1, math.floor(height * ratio + 0.5))
    return {
        "resizedWidth": resized_width, "resizedHeight": resized_height,
        "left": (size - resized_width) // 2, "top": (size - resized_height) // 2,
    }


def padded_crop(box, width, height, margin=0.1):
    x, y, box_width, box_height = checked_box(box)
    finite_number(margin, "crop margin", 0, 0.5)
    if any(type(value) is not int or value <= 0 for value in (width, height)):
        raise ValueError("Crop image dimensions must be positive integers.")
    left = max(0, math.floor((x - box_width * margin) * width))
    top = max(0, math.floor((y - box_height * margin) * height))
    right = min(width, math.ceil((x + box_width * (1 + margin)) * width))
    bottom = min(height, math.ceil((y + box_height * (1 + margin)) * height))
    if right <= left or bottom <= top:
        raise ValueError("Predicted crop is empty.")
    return left, top, right, bottom


def image_tensor(image, size, *, detector=False, box=None, margin=0.1,
                 mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    import numpy as np
    from PIL import Image
    if image.mode != "RGB":
        raise ValueError("Input image must already be normalized, oriented RGB.")
    if detector:
        geometry = letterbox_geometry(*image.size, size)
        resized = image.resize((geometry["resizedWidth"], geometry["resizedHeight"]), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        canvas.paste(resized, (geometry["left"], geometry["top"]))
        pixels, mean, std = canvas, (0, 0, 0), (1, 1, 1)
    else:
        pixels = image.crop(padded_crop(box, *image.size, margin)).resize((size, size), Image.Resampling.BILINEAR)
        geometry = None
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("RGB normalization requires three mean/std values.")
    for value in mean:
        finite_number(value, "mean")
    for value in std:
        finite_number(value, "std", 1e-12)
    # Canvas uses double arithmetic then writes Float32Array; cast after normalization.
    array = np.asarray(pixels, dtype=np.float64) / 255
    array = (array - np.asarray(mean)) / np.asarray(std)
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None], dtype=np.float32), geometry


def intersection_over_union(a, b):
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[0] + a[2], b[0] + b[2])
    bottom = min(a[1] + a[3], b[1] + b[3])
    area = max(0, right - left) * max(0, bottom - top)
    union = a[2] * a[3] + b[2] * b[3] - area
    return area / union if union > 0 else 0.0


def decode_detector(values, geometry, score_floor):
    import numpy as np
    values = np.asarray(values)
    finite_number(score_floor, "score floor", 0, 1)
    if values.dtype != np.float32 or values.ndim != 3 or values.shape[:2] != (1, 5) or values.shape[2] < 1:
        raise ValueError("Detector output must be float32 1x5xN, without embedded NMS.")
    if not np.isfinite(values).all():
        raise ValueError("Detector output contains non-finite values.")
    if np.any(values[0, 4] < 0) or np.any(values[0, 4] > 1):
        raise ValueError("Detector output contains invalid confidence probabilities.")
    width, height = geometry["resizedWidth"], geometry["resizedHeight"]
    if width <= 0 or height <= 0:
        raise ValueError("Invalid letterbox geometry.")
    detections = []
    for index in range(values.shape[2]):
        cx, cy, w, h, score = map(float, values[0, :, index])
        if score < score_floor or w <= 0 or h <= 0:
            continue
        x1 = max(0, min(1, (cx - w / 2 - geometry["left"]) / width))
        y1 = max(0, min(1, (cy - h / 2 - geometry["top"]) / height))
        x2 = max(0, min(1, (cx + w / 2 - geometry["left"]) / width))
        y2 = max(0, min(1, (cy + h / 2 - geometry["top"]) / height))
        if x2 > x1 and y2 > y1:
            detections.append(Detection((x1, y1, x2 - x1, y2 - y1), score, index))
    detections.sort(key=lambda item: (-item.score, item.anchor))
    return tuple(detections[:NMS_CANDIDATES]), len(detections)


def non_max_suppression(detections, score_threshold, iou_threshold):
    finite_number(score_threshold, "detector threshold", 0, 1)
    finite_number(iou_threshold, "NMS IoU", 0, 1)
    pending = sorted((item for item in detections if item.score >= score_threshold),
                     key=lambda item: (-item.score, item.anchor))[:NMS_CANDIDATES]
    selected = []
    for item in pending:
        if not any(intersection_over_union(item.box, other.box) > iou_threshold for other in selected):
            selected.append(item)
            if len(selected) == NMS_LIMIT:
                break
    return tuple(selected)


def classify(logits, score_threshold, min_margin):
    logits = checked_logits(logits)
    finite_number(score_threshold, "classifier threshold", 0, 1)
    finite_number(min_margin, "top-two margin", 0, 1)
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    total = sum(exponentials)
    probabilities = [value / total for value in exponentials]
    ranking = sorted(range(CLASS_COUNT), key=lambda index: (-probabilities[index], index))
    top, second = ranking[:2]
    score, margin = probabilities[top], probabilities[top] - probabilities[second]
    return top, score >= score_threshold and margin >= min_margin


def match_detections(detections, truth, iou_threshold=MATCH_IOU):
    finite_number(iou_threshold, "matching IoU", 0, 1)
    used, matches = set(), {}
    for index, detection in enumerate(detections):
        candidates = [(intersection_over_union(detection.box, obj.box), target)
                      for target, obj in enumerate(truth) if target not in used]
        candidates.sort(key=lambda value: (-value[0], value[1]))
        if candidates and candidates[0][0] >= iou_threshold:
            target = candidates[0][1]
            used.add(target)
            matches[index] = target
    return matches


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def confusion_metrics(matrix, row_labels, column_labels):
    if len(matrix) != len(row_labels) or any(len(row) != len(column_labels) for row in matrix):
        raise ValueError("Confusion-matrix dimensions do not match labels.")
    if len(row_labels) < CLASS_COUNT or len(column_labels) < CLASS_COUNT:
        raise ValueError("All ten taxonomy classes must be represented.")
    if any(type(value) is not int or value < 0 for row in matrix for value in row):
        raise ValueError("Confusion counts must be nonnegative integers.")
    slugs = [label["slug"] for label in taxonomy()["labels"]]
    per_class = {}
    for index, slug in enumerate(slugs):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        denominator = support + predicted
        per_class[slug] = {
            "support": support, "predicted": predicted, "truePositive": tp,
            "falsePositive": predicted - tp, "falseNegative": support - tp,
            "precision": tp / predicted if predicted else 0.0,
            "recall": tp / support if support else 0.0,
            "f1": 2 * tp / denominator if denominator else 0.0,
        }
    support = sum(value["support"] for value in per_class.values())
    return {
        "confusion": {"rows": row_labels, "columns": column_labels, "counts": matrix},
        "perClass": per_class,
        "macroF1": sum(value["f1"] for value in per_class.values()) / CLASS_COUNT,
        "correctObjectCoverage": ratio(sum(matrix[index][index] for index in range(CLASS_COUNT)), support),
        "support": support,
        "zeroDivision": "0 for unsupported per-class precision/recall/F1; macro includes all ten classes.",
    }


def _crop_metrics(samples, thresholds, slugs):
    matrix = [[0] * CLASS_COUNT for _ in range(CLASS_COUNT)]
    thresholded = [[0] * (CLASS_COUNT + 1) for _ in range(CLASS_COUNT)]
    accepted, correct = 0, 0
    for target, logits in samples:
        predicted, accept = classify(logits, thresholds["classifierScoreThreshold"], thresholds["minMargin"])
        matrix[target][predicted] += 1
        thresholded[target][predicted if accept else CLASS_COUNT] += 1
        accepted += accept
        correct += accept and target == predicted
    return {
        "unconditional": confusion_metrics(matrix, slugs, slugs),
        "thresholded": confusion_metrics(thresholded, slugs, slugs + ["rejected"]),
        "crops": len(samples), "acceptedCrops": accepted,
        "acceptedCoverage": ratio(accepted, len(samples)),
        "acceptedAccuracy": ratio(correct, accepted),
    }


def _detection_metrics(tp, predicted, actual):
    return {"truePositive": tp, "falsePositive": predicted - tp, "falseNegative": actual - tp,
            "predictions": predicted, "groundTruth": actual,
            "precision": ratio(tp, predicted), "recall": ratio(tp, actual), "matchingIou": MATCH_IOU}


def evaluate_frames(frames, thresholds, *, include_cases=True):
    thresholds = checked_thresholds(thresholds)
    frames = tuple(frames)
    if not frames or len({frame.split for frame in frames}) != 1:
        raise ValueError("Evaluate exactly one nonempty holdout split at a time.")
    if len({frame.path for frame in frames}) != len(frames):
        raise ValueError("Duplicate prediction image.")
    slugs = [label["slug"] for label in taxonomy()["labels"]]
    matrix = [[0] * (CLASS_COUNT + 2) for _ in range(CLASS_COUNT + 1)]
    gt_samples, matched_samples, cases = [], [], []
    all_tp = all_predictions = selected_tp = selected_predictions = total_gt = 0
    for frame in frames:
        detections = non_max_suppression(frame.detections, thresholds["detectorScoreThreshold"], thresholds["iouThreshold"])
        all_matches = match_detections(detections, frame.truth)
        selected = detections[:MAX_OBJECTS]
        matches = match_detections(selected, frame.truth)
        all_tp += len(all_matches)
        all_predictions += len(detections)
        selected_tp += len(matches)
        selected_predictions += len(selected)
        total_gt += len(frame.truth)
        accepted = accepted_matched = correct = false_positive = 0
        for obj in frame.truth:
            gt_samples.append((obj.class_id, obj.logits))
        for index, detection in enumerate(selected):
            if detection.logits is None:
                raise ValueError(f"Missing real classifier prediction for selected crop: {frame.path}, anchor {detection.anchor}")
            predicted, accept = classify(detection.logits, thresholds["classifierScoreThreshold"], thresholds["minMargin"])
            accepted += accept
            target = frame.truth[matches[index]].class_id if index in matches else CLASS_COUNT
            matrix[target][predicted if accept else CLASS_COUNT] += 1
            if index in matches:
                matched_samples.append((target, detection.logits))
                accepted_matched += accept
                correct += accept and predicted == target
            elif accept:
                false_positive += 1
        for index, obj in enumerate(frame.truth):
            if index not in matches.values():
                matrix[obj.class_id][CLASS_COUNT + 1] += 1
        cases.append({
            "path": frame.path, "groupId": frame.group_id, "objects": len(frame.truth),
            "truthClassIds": [obj.class_id for obj in frame.truth],
            "detectionsAfterNms": len(detections), "selectedDetections": len(selected),
            "matchedObjects": len(matches), "acceptedPredictions": accepted,
            "acceptedMatchedObjects": accepted_matched, "acceptedCorrectObjects": correct,
            "acceptedBackgroundPredictions": false_positive,
            "fullyRecognized": bool(frame.truth) and correct == len(frame.truth) and false_positive == 0,
            "overflow": len(detections) > MAX_OBJECTS,
            "preNmsCandidates": frame.candidate_count,
        })
    negatives = [case for case in cases if case["objects"] == 0]
    negative_accepts = sum(case["acceptedPredictions"] > 0 for case in negatives)
    total_accepted = sum(case["acceptedPredictions"] for case in cases)
    accepted_matched = sum(case["acceptedMatchedObjects"] for case in cases)
    accepted_correct = sum(case["acceptedCorrectObjects"] for case in cases)
    multiobject = {}
    for count in range(2, 6):
        subset = [case for case in cases if case["objects"] == count]
        support = count * len(subset)
        multiobject[str(count)] = {
            "frames": len(subset), "groups": len({case["groupId"] for case in subset}),
            "groundTruthObjects": support, "measured": bool(subset),
            "detectionCoverage": ratio(sum(case["matchedObjects"] for case in subset), support),
            "acceptedCorrectCoverage": ratio(sum(case["acceptedCorrectObjects"] for case in subset), support),
            "fullyRecognizedFrames": sum(case["fullyRecognized"] for case in subset),
            "fullyRecognizedFrameRate": ratio(sum(case["fullyRecognized"] for case in subset), len(subset)),
        }
    end_to_end = confusion_metrics(matrix, slugs + ["background"], slugs + ["rejected", "missed"])
    end_to_end.update({
        "acceptedPredictions": total_accepted,
        "acceptedPredictionFraction": ratio(total_accepted, selected_predictions),
        "acceptedGtObjectCoverage": ratio(accepted_matched, total_gt),
        "acceptedCorrectObjectCoverage": ratio(accepted_correct, total_gt),
        "acceptedPrecision": ratio(accepted_correct, total_accepted),
        "note": "Wrong types count as FP/FN, rejected/missed GT as FN, accepted unmatched boxes as FP. "
                "Background/rejected/missed are accounting bins, not additional model classes.",
    })
    result = {
        "split": frames[0].split, "frames": len(frames), "groups": len({frame.group_id for frame in frames}),
        "detectorAfterNms": _detection_metrics(all_tp, all_predictions, total_gt),
        "detectorAfterObjectCap": _detection_metrics(selected_tp, selected_predictions, total_gt),
        "classifierGroundTruthCrops": {
            **_crop_metrics(gt_samples, thresholds, slugs),
            "note": "Ground-truth boxes cropped from original normalized images with the same "
                    "padding/resize/normalization as predicted crops; not recompressed training JPEG crops.",
        },
        "classifierMatchedPredictedCrops": {
            **_crop_metrics(matched_samples, thresholds, slugs),
            "note": "Conditional on class-agnostic IoU-matched predicted crops. Excludes missed GT "
                    "and unmatched predictions; not an end-to-end accuracy estimate.",
        },
        "endToEnd": end_to_end,
        "negativeOrUnknown": {
            "measured": bool(negatives), "frames": len(negatives),
            "groups": len({case["groupId"] for case in negatives}),
            "acceptedFrames": negative_accepts, "frameFalseAcceptRate": ratio(negative_accepts, len(negatives)),
            "acceptedBoxes": sum(case["acceptedPredictions"] for case in negatives),
            "note": "Unmeasured strata use null rates, never zero-risk claims.",
        },
        "multiobject": multiobject,
        "overflowFrames": sum(case["overflow"] for case in cases),
        "framesAboveObjectCap": sum(case["objects"] > MAX_OBJECTS for case in cases),
    }
    if include_cases:
        result["cases"] = cases
    return result


def calibrate_validation(frames):
    frames = tuple(frames)
    if not frames or any(frame.split != "valid" for frame in frames):
        raise ValueError("Threshold selection is allowed only on the validation split.")
    candidates, best_key, best_thresholds = [], None, None
    for values in itertools.product(*(DEFAULT_GRID[key] for key in THRESHOLD_NAMES)):
        thresholds = dict(zip(THRESHOLD_NAMES, values))
        metrics = evaluate_frames(frames, thresholds, include_cases=False)
        end_to_end, negative = metrics["endToEnd"], metrics["negativeOrUnknown"]
        key = (end_to_end["macroF1"], end_to_end["acceptedCorrectObjectCoverage"] or 0,
               -negative["acceptedFrames"])
        candidates.append({
            "thresholds": thresholds, "macroF1": end_to_end["macroF1"],
            "acceptedGtObjectCoverage": end_to_end["acceptedGtObjectCoverage"],
            "acceptedCorrectObjectCoverage": end_to_end["acceptedCorrectObjectCoverage"],
            "negativeFrameFalseAcceptRate": negative["frameFalseAcceptRate"],
        })
        if best_key is None or key > best_key:
            best_key, best_thresholds = key, thresholds
    return {
        "thresholds": best_thresholds, "candidates": candidates,
        "metrics": evaluate_frames(frames, best_thresholds),
    }


def needed_anchors(detections, thresholds=None):
    pairs = ((thresholds["detectorScoreThreshold"], thresholds["iouThreshold"]),) if thresholds else itertools.product(
        DEFAULT_GRID["detectorScoreThreshold"], DEFAULT_GRID["iouThreshold"])
    return {item.anchor for score, iou in pairs
            for item in non_max_suppression(detections, score, iou)[:MAX_OBJECTS]}


def predict_split(records, split, data_root, backend, thresholds=None):
    from PIL import Image
    from training.common import source_path
    if split not in ("valid", "test") or (split == "test" and thresholds is None):
        raise ValueError("Test predictions require already frozen thresholds.")
    if thresholds is not None:
        thresholds = checked_thresholds(thresholds)
    floor = thresholds["detectorScoreThreshold"] if thresholds else min(DEFAULT_GRID["detectorScoreThreshold"])
    frames = []
    for row in records:
        if row["split"] != split:
            continue
        contents = source_path(data_root, row["path"]).read_bytes()
        if hashlib.sha256(contents).hexdigest() != row["sha256"]:
            raise ValueError(f"Source image changed since preparation: {row['path']}")
        with Image.open(io.BytesIO(contents)) as original:
            original.load()
            if original.getexif().get(274, 1) not in (1, None):
                raise ValueError(f"Normalize EXIF before annotation/evaluation: {row['path']}")
            image = original.convert("RGB")
            output, geometry = backend.detect(image)
            detections, candidate_count = decode_detector(output, geometry, floor)
            anchors = needed_anchors(detections, thresholds)
            detections = tuple(
                replace(item, logits=tuple(backend.classify(image, item.box))) if item.anchor in anchors else item
                for item in detections
            )
            truth = tuple(GroundTruth(tuple(obj["bbox"]), obj["classId"],
                                     tuple(backend.classify(image, obj["bbox"]))) for obj in row["objects"])
        frames.append(FramePrediction(row["path"], row["groupId"], split, truth, detections, candidate_count))
    if not frames:
        raise ValueError(f"No {split} images are available for evaluation.")
    return tuple(frames)


def predictions_document(frames, input_fingerprint):
    return {
        "schemaVersion": 1, "inputFingerprint": input_fingerprint,
        "frames": [{
            "path": frame.path, "groupId": frame.group_id, "split": frame.split,
            "candidateCount": frame.candidate_count,
            "truth": [{"box": obj.box, "classId": obj.class_id, "logits": obj.logits} for obj in frame.truth],
            "detections": [{"box": obj.box, "score": obj.score, "anchor": obj.anchor, "logits": obj.logits}
                           for obj in frame.detections],
        } for frame in frames],
    }


def load_predictions(document, records, split, input_fingerprint):
    if document.get("schemaVersion") != 1 or document.get("inputFingerprint") != input_fingerprint:
        raise ValueError("Cached predictions belong to different checkpoints/data/protocol.")
    expected = {row["path"]: row for row in records if row["split"] == split}
    frames, seen = [], set()
    for item in document["frames"]:
        path = item["path"]
        if path not in expected or path in seen:
            raise ValueError("Cached prediction paths do not match this holdout.")
        row = expected[path]
        if item["split"] != split or item["groupId"] != row["groupId"]:
            raise ValueError("Cached prediction split/group differs from manifest.")
        truth = tuple(GroundTruth(tuple(obj["box"]), obj["classId"], tuple(obj["logits"])) for obj in item["truth"])
        if (len(truth) != len(row["objects"]) or
                any(obj.class_id != original["classId"] or obj.box != tuple(original["bbox"])
                    for obj, original in zip(truth, row["objects"]))):
            raise ValueError("Cached ground truth differs from manifest.")
        detections = tuple(Detection(tuple(obj["box"]), obj["score"], obj["anchor"],
                                     tuple(obj["logits"]) if obj["logits"] is not None else None)
                           for obj in item["detections"])
        frames.append(FramePrediction(path, item["groupId"], split, truth, detections, item["candidateCount"]))
        seen.add(path)
    if seen != set(expected):
        raise ValueError("Cached predictions omit holdout images.")
    return tuple(frames)


def detector_checkpoint_binding(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    root = checkpoint.parent.parent
    binding_file, inventory_file = root / "training-binding.json", root / "checkpoint-inventory.json"
    if not binding_file.is_file() or not inventory_file.is_file():
        return None
    binding, inventory = read_json(binding_file), read_json(inventory_file)
    relative = checkpoint.relative_to(root).as_posix()
    if inventory.get("bindingSha256") != digest(binding_file) or inventory.get("files", {}).get(relative) != digest(checkpoint):
        raise ValueError("Detector checkpoint does not match its completed training provenance receipt.")
    return binding


def checkpoint_eligibility(classifier_checkpoint, detector_binding, dataset_binding, export_config):
    unmet = []
    if classifier_checkpoint.get("datasetBinding") != dataset_binding:
        unmet.append("Classifier checkpoint lacks this exact prepared dataset fingerprint; train/recover with bound provenance.")
    if detector_binding is None or detector_binding.get("datasetBinding") != dataset_binding:
        unmet.append("Detector checkpoint lacks this exact prepared dataset fingerprint and checkpoint receipt.")
    elif detector_binding.get("config", {}).get("model") != "yolo11n.pt":
        unmet.append("Detector training provenance is not the approved YOLO11n configuration.")
    config = classifier_checkpoint["config"]
    if (config.get("architecture") != "tf_efficientnetv2_b0.in1k" or config.get("cbam") is not True
            or config.get("inputSize") != 224 or config.get("mean") != [0.5] * 3 or config.get("std") != [0.5] * 3):
        unmet.append("Classifier does not match the approved EfficientNetV2-B0 + CBAM browser configuration.")
    if export_config.get("detectorInputSize") != 416 or export_config.get("cropMargin") != 0.1:
        unmet.append("Evaluation requires the approved detector 416/crop margin 0.1 configuration.")
    if export_config.get("cropMargin") != dataset_binding["cropMargin"]:
        unmet.append("Prepared crops and evaluation use different crop margins.")
    return unmet


def verify_calibration_evidence(calibration_path, detector_sha, classifier_sha, export_config, slugs):
    calibration_path = Path(calibration_path)
    calibration = read_json(calibration_path)
    if (calibration.get("schemaVersion") != 1 or calibration.get("status") != "validated"
            or calibration.get("split") != "valid" or calibration.get("protocol") != "group-disjoint"
            or calibration.get("protocolVersion") != PROTOCOL_VERSION):
        raise ValueError("Require measured, group-disjoint validation calibration from evaluate_models.")
    if (calibration.get("detectorCheckpointSha256") != detector_sha
            or calibration.get("classifierCheckpointSha256") != classifier_sha):
        raise ValueError("Calibration was not performed for these exact checkpoints.")
    if calibration.get("classSlugs") != slugs:
        raise ValueError("Calibration class order must match shared labels.")
    if calibration.get("eligibility") != {"eligible": True, "unmetGates": []}:
        raise ValueError("Calibration has unmet eligibility gates.")
    if (calibration.get("detectorInputSize") != export_config["detectorInputSize"]
            or calibration.get("cropMargin") != export_config["cropMargin"]
            or calibration.get("exportConfigSha256") != json_fingerprint(export_config)):
        raise ValueError("Export configuration differs from calibration.")
    thresholds = checked_thresholds({key: calibration.get(key) for key in THRESHOLD_NAMES})
    if any(value not in DEFAULT_GRID[key] for key, value in thresholds.items()):
        raise ValueError("Calibration thresholds are outside the declared finite grid.")
    report_path = calibration_path.parent / "evaluation-report.json"
    frozen_path = calibration_path.parent / "frozen-thresholds.json"
    if (calibration.get("evaluationReportSha256") != digest(report_path)
            or calibration.get("frozenThresholdsSha256") != digest(frozen_path)):
        raise ValueError("Calibration evaluation/frozen-threshold evidence was modified or is missing.")
    report, frozen = read_json(report_path), read_json(frozen_path)
    if (report.get("schemaVersion") != 1 or report.get("protocolVersion") != PROTOCOL_VERSION
            or report.get("status") != "evaluated" or report.get("eligibility") != {"eligible": True, "unmetGates": []}
            or frozen.get("thresholds") != thresholds or frozen.get("selectionSplit") != "valid"
            or frozen.get("protocol") != protocol() or report.get("protocol") != protocol()
            or report.get("frozenThresholdsSha256") != digest(frozen_path)
            or report.get("datasetInspection", {}).get("groupDisjoint") is not True
            or report["datasetInspection"].get("eligible") is not True):
        raise ValueError("Calibration has no eligible measured report with the declared frozen validation protocol.")
    for key in ("datasetBinding", "detectorCheckpointSha256", "classifierCheckpointSha256",
                "classSlugs", "classifierConfigSha256", "exportConfigSha256"):
        if calibration.get(key) is None or calibration[key] != report["inputs"].get(key) or calibration[key] != frozen["inputs"].get(key):
            raise ValueError(f"Calibration evidence binding differs: {key}")
    for key in ("datasetFingerprint", "preparedManifestSha256", "preparedReportSha256", "preparedFilesSha256"):
        value = calibration["datasetBinding"].get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
            raise ValueError(f"Missing dataset fingerprint component: {key}")
    for split in ("valid", "test"):
        metrics = report["metrics"][split]
        negative = metrics["negativeOrUnknown"]
        if metrics["split"] != split or negative["frames"] < 1 or not negative["measured"]:
            raise ValueError(f"{split} negative/unknown false accept is unmeasured.")
        finite_number(negative["frameFalseAcceptRate"], "negative false accept", 0, 1)
        for count in range(2, 6):
            stratum = metrics["multiobject"][str(count)]
            if stratum["frames"] < 1 or not stratum["measured"] or stratum["groups"] < 1:
                raise ValueError(f"{split} lacks real {count}-object evaluation.")
        if (set(metrics["endToEnd"]["perClass"]) != set(slugs)
                or any(value["support"] < 1 for value in metrics["endToEnd"]["perClass"].values())):
            raise ValueError(f"{split} is missing taxonomy coverage.")
        finite_number(metrics["endToEnd"]["macroF1"], "end-to-end macro F1", 0, 1)
    return calibration, report, frozen
