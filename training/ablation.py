"""Ablation study helpers: measure what YOLOv11 localization and CBAM attention contribute.

Variants share the same prepared, specimen-disjoint split so differences come from the model only:
- full: YOLO11n crop -> EfficientNetV2-B0 + CBAM (deployed pipeline)
- tanpa-cbam: YOLO11n crop -> EfficientNetV2-B0 (configs/classifier.no-cbam.json)
- tanpa-yolo: whole image -> EfficientNetV2-B0 + CBAM (trained on ablation-whole-image)
All variants are compared on image-level accuracy over single-species frames, the only metric
that a whole-image classifier can also produce.
"""

import math
from pathlib import Path

from training.common import SPLITS, taxonomy
from training.evaluation import (
    CLASS_COUNT, MAX_OBJECTS, checked_logits, checked_thresholds, classify, confusion_metrics, non_max_suppression,
)


WHOLE_IMAGE_FOLDER = "ablation-whole-image"
WHOLE_IMAGE_BOX = (0.0, 0.0, 1.0, 1.0)
WHOLE_IMAGE_MAX_SIDE = 448
VARIANTS = ("full", "tanpa-cbam", "tanpa-yolo")


def single_species(row):
    classes = {obj["classId"] for obj in row["objects"]}
    return classes.pop() if len(classes) == 1 else None


def build_whole_image_folder(records, prepared):
    """Copy uncropped single-species detector images into an ImageFolder tree, reusing the frozen splits."""
    from PIL import Image, ImageOps
    prepared = Path(prepared)
    destination = prepared / WHOLE_IMAGE_FOLDER
    if destination.exists():
        raise ValueError(f"{destination} exists; whole-image ablation data is write-once.")
    slugs = [label["slug"] for label in taxonomy()["labels"]]
    counts = {split: dict.fromkeys(slugs, 0) for split in SPLITS}
    planned = []
    for index, row in enumerate(records):
        target = single_species(row)
        if target is None:
            continue
        stem = f"{index:06d}_{row['sha256'][:12]}"
        planned.append((prepared / "detector" / "images" / row["split"] / f"{stem}.jpg",
                        destination / row["split"] / slugs[target] / f"{stem}.jpg"))
        counts[row["split"]][slugs[target]] += 1
    missing = [f"{split}/{slug}" for split, values in counts.items() for slug, count in values.items() if not count]
    if missing:
        raise ValueError(f"Whole-image ablation lacks single-species images for: {', '.join(missing[:10])}")
    for split in SPLITS:
        for slug in slugs:
            (destination / split / slug).mkdir(parents=True)
    for source, target in planned:
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((WHOLE_IMAGE_MAX_SIDE, WHOLE_IMAGE_MAX_SIDE), Image.Resampling.LANCZOS)
        image.save(target, quality=95)
    return counts


def softmax_score(logits):
    logits = checked_logits(logits)
    peak = max(logits)
    exponentials = [math.exp(value - peak) for value in logits]
    return max(exponentials) / sum(exponentials)


def pipeline_image_prediction(frame, thresholds):
    """Image label from the detector pipeline: most confident accepted crop, or None if nothing accepted."""
    thresholds = checked_thresholds(thresholds)
    selected = non_max_suppression(frame.detections, thresholds["detectorScoreThreshold"],
                                   thresholds["iouThreshold"])[:MAX_OBJECTS]
    best = None
    for detection in selected:
        if detection.logits is None:
            raise ValueError(f"Missing classifier logits for selected crop: {frame.path}")
        predicted, accept = classify(detection.logits, thresholds["classifierScoreThreshold"], thresholds["minMargin"])
        if accept:
            score = softmax_score(detection.logits)
            if best is None or score > best[0]:
                best = (score, predicted)
    return None if best is None else best[1]


def whole_image_prediction(logits, thresholds):
    thresholds = checked_thresholds(thresholds)
    predicted, accept = classify(logits, thresholds["classifierScoreThreshold"], thresholds["minMargin"])
    return predicted if accept else None


def image_level_metrics(samples):
    """samples: (target classId, predicted classId or None). Unanswered images count as errors."""
    samples = list(samples)
    if not samples:
        raise ValueError("Image-level ablation needs at least one single-species image.")
    slugs = [label["slug"] for label in taxonomy()["labels"]]
    matrix = [[0] * (CLASS_COUNT + 1) for _ in range(CLASS_COUNT)]
    for target, predicted in samples:
        if type(target) is not int or not 0 <= target < CLASS_COUNT:
            raise ValueError("Ablation targets must be taxonomy class ids.")
        if predicted is not None and (type(predicted) is not int or not 0 <= predicted < CLASS_COUNT):
            raise ValueError("Ablation predictions must be class ids or None.")
        matrix[target][CLASS_COUNT if predicted is None else predicted] += 1
    metrics = confusion_metrics(matrix, slugs, slugs + ["tidak-dijawab"])
    correct = sum(matrix[index][index] for index in range(CLASS_COUNT))
    return {
        "images": len(samples), "accuracy": correct / len(samples), "macroF1": metrics["macroF1"],
        "unanswered": sum(row[CLASS_COUNT] for row in matrix),
        "perClassF1": {slug: values["f1"] for slug, values in metrics["perClass"].items()},
        "confusion": metrics["confusion"],
    }


def summarize(results):
    """results: {variant: image_level_metrics}. Returns rows with deltas against the full model."""
    if "full" not in results:
        raise ValueError("Ablation summary needs the full model as reference.")
    unknown = set(results) - set(VARIANTS)
    if unknown:
        raise ValueError(f"Unknown ablation variants: {sorted(unknown)}")
    reference = results["full"]
    rows = []
    for variant in VARIANTS:
        if variant not in results:
            continue
        current = results[variant]
        if current["images"] != reference["images"]:
            raise ValueError("All ablation variants must be scored on the same images.")
        rows.append({
            "variant": variant, "images": current["images"],
            "accuracy": current["accuracy"], "macroF1": current["macroF1"],
            "deltaAccuracy": current["accuracy"] - reference["accuracy"],
            "deltaMacroF1": current["macroF1"] - reference["macroF1"],
        })
    return rows


def summary_markdown(rows):
    names = {"full": "YOLOv11 + EfficientNetV2 + CBAM", "tanpa-cbam": "YOLOv11 + EfficientNetV2 (tanpa CBAM)",
             "tanpa-yolo": "EfficientNetV2 + CBAM (tanpa YOLOv11)"}
    lines = ["| Varian | Citra | Akurasi | Macro F1 | Δ Akurasi | Δ Macro F1 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {names[row['variant']]} | {row['images']} | {row['accuracy']:.1%} | {row['macroF1']:.1%} "
                     f"| {row['deltaAccuracy']:+.1%} | {row['deltaMacroF1']:+.1%} |")
    return "\n".join(lines) + "\n"
