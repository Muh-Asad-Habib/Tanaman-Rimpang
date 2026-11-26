"""Robustness evaluation: per-capture-condition strata and synthetic photo degradations.

Both reuse the frozen validation thresholds on the test split; neither is used for tuning.
Strata need capture metadata (shared/dataset-manifest.schema.json); unrecorded images are
reported under "tidak-dicatat" so missing field data stays visible instead of being hidden.
"""

import hashlib

from training.dataset import CAPTURE_KEYS
from training.evaluation import evaluate_frames, ratio


UNRECORDED = "tidak-dicatat"
PERTURBATIONS = {
    "gelap": "brightness x0.5 (underexposed field photo)",
    "terang": "brightness x1.6 (harsh sunlight)",
    "kontras-rendah": "contrast x0.5 (hazy or shaded)",
    "buram": "Gaussian blur radius 2 (camera shake/out of focus)",
    "noise": "Gaussian noise sigma 12 (low-light sensor noise)",
    "jpeg-rendah": "JPEG quality 25 (chat-app recompression)",
}


def perturb(image, name, key=""):
    """Deterministic degradation; noise is seeded by the image key so runs are reproducible."""
    import io
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter
    if name not in PERTURBATIONS:
        raise ValueError(f"Unknown perturbation: {name}")
    if name == "gelap":
        return ImageEnhance.Brightness(image).enhance(0.5)
    if name == "terang":
        return ImageEnhance.Brightness(image).enhance(1.6)
    if name == "kontras-rendah":
        return ImageEnhance.Contrast(image).enhance(0.5)
    if name == "buram":
        return image.filter(ImageFilter.GaussianBlur(2))
    if name == "noise":
        seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big")
        pixels = np.asarray(image, dtype=np.float64)
        noisy = pixels + np.random.default_rng(seed).normal(0, 12, pixels.shape)
        return Image.fromarray(np.clip(noisy + 0.5, 0, 255).astype(np.uint8), "RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=25)
    buffer.seek(0)
    with Image.open(buffer) as degraded:
        return degraded.convert("RGB")


def summary(metrics):
    end_to_end = metrics["endToEnd"]
    return {
        "frames": metrics["frames"], "groups": metrics["groups"],
        "groundTruthObjects": end_to_end["support"],
        "macroF1": end_to_end["macroF1"],
        "acceptedPrecision": end_to_end["acceptedPrecision"],
        "acceptedCorrectObjectCoverage": end_to_end["acceptedCorrectObjectCoverage"],
    }


def capture_strata(frames, records, thresholds):
    """End-to-end metrics per capture condition value, e.g. lighting=redup vs lighting=terang."""
    by_path = {row["path"]: row for row in records}
    strata = {}
    for key in CAPTURE_KEYS:
        buckets = {}
        for frame in frames:
            value = (by_path[frame.path].get("capture") or {}).get(key) or UNRECORDED
            buckets.setdefault(value, []).append(frame)
        strata[key] = {}
        for value, subset in sorted(buckets.items()):
            row = summary(evaluate_frames(subset, thresholds, include_cases=False))
            row["classesPresent"] = len({obj.class_id for frame in subset for obj in frame.truth})
            row["note"] = "macroF1 averages all ten classes; compare coverage/precision when classes are missing."
            strata[key][value] = row
    recorded = sum(1 for frame in frames if by_path[frame.path].get("capture"))
    return {"recordedFrames": recorded, "recordedFraction": ratio(recorded, len(frames)), "strata": strata}


def degradation_table(baseline, degraded):
    """degraded: {perturbation: evaluate_frames result}. Deltas are relative to clean test images."""
    clean = summary(baseline)
    rows = {"bersih": {**clean, "description": "original test images"}}
    for name, metrics in degraded.items():
        current = summary(metrics)
        rows[name] = {
            **current, "description": PERTURBATIONS[name],
            "deltaMacroF1": current["macroF1"] - clean["macroF1"],
            "deltaAcceptedCorrectObjectCoverage": (
                None if current["acceptedCorrectObjectCoverage"] is None or clean["acceptedCorrectObjectCoverage"] is None
                else current["acceptedCorrectObjectCoverage"] - clean["acceptedCorrectObjectCoverage"]),
        }
    return rows
