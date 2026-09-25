import math
import random
from collections import Counter, defaultdict

from PIL import Image, ImageOps

from training.common import SPLITS, digest, source_path, taxonomy, validate_schema


def _inspect_image(image_path):
    if image_path.stat().st_size == 0:
        raise ValueError(f"Empty image: {image_path}")
    with Image.open(image_path) as original:
        original.load()
        orientation = original.getexif().get(274, 1)
        size = original.size
    return digest(image_path), orientation, size


def audit_manifest(manifest, data_root, workers=8):
    validate_schema(manifest, "dataset-manifest.schema.json")
    seen_paths, seen_hashes, group_splits = set(), {}, {}
    records, counts = [], Counter()
    paths = [source_path(data_root, item["path"]) for item in manifest["images"]]
    if workers > 1 and len(paths) > 64:
        from multiprocessing import get_context
        with get_context("spawn").Pool(workers) as pool:
            inspected = pool.map(_inspect_image, paths, chunksize=16)
    else:
        inspected = [_inspect_image(path) for path in paths]
    for item, image_path, (sha, orientation, size) in zip(manifest["images"], paths, inspected):
        if image_path in seen_paths:
            raise ValueError(f"Image referenced more than once: {item['path']}")
        seen_paths.add(image_path)
        if sha in seen_hashes:
            raise ValueError(f"Exact duplicate image: {item['path']} and {seen_hashes[sha]}")
        seen_hashes[sha] = item["path"]
        if item.get("sha256", sha) != sha:
            raise ValueError(f"Checksum mismatch: {item['path']}")
        if orientation not in (1, None):
            raise ValueError(f"Normalize EXIF orientation BEFORE annotation: {item['path']}")
        if min(size) < 16:
            raise ValueError(f"Image too small: {item['path']}")
        group = item["groupId"].strip()
        if not group or not item["sourceId"].strip():
            raise ValueError("sourceId/groupId must not be blank.")
        split = item.get("split")
        if split:
            if group in group_splits and group_splits[group] != split:
                raise ValueError(f"Group crosses splits: {group}")
            group_splits[group] = split
        for obj in item["objects"]:
            x, y, width, height = obj["bbox"]
            if not all(math.isfinite(v) for v in obj["bbox"]) or width <= 0 or height <= 0 or x + width > 1 + 1e-9 or y + height > 1 + 1e-9:
                raise ValueError(f"Invalid bbox: {item['path']}: {obj['bbox']}")
            counts[obj["classId"]] += 1
        records.append({**item, "groupId": group, "sourceId": item["sourceId"].strip(), "sha256": sha})
    return records, {
        "images": len(records), "groups": len({item["groupId"] for item in records}),
        "objectsPerClass": {label["slug"]: counts[label["id"]] for label in taxonomy()["labels"]},
        "negativeImages": sum(not item["objects"] for item in records),
    }


def assign_splits(records, seed=42):
    supplied = ["split" in record for record in records]
    if any(supplied) and not all(supplied):
        raise ValueError("Provide splits for every image or for none.")
    groups = sorted({record["groupId"] for record in records})
    if len(groups) < 3:
        raise ValueError("At least three independent groups are required.")
    if all(supplied):
        result = records
    else:
        by_class = defaultdict(set)
        for record in records:
            for obj in record["objects"]:
                by_class[obj["classId"]].add(record["groupId"])
        missing = [label["slug"] for label in taxonomy()["labels"] if len(by_class[label["id"]]) < 3]
        if missing:
            raise ValueError(f"Need >=3 independent groups per class: {', '.join(missing)}")
        rng = random.Random(seed)
        result = None
        for _ in range(1000):
            order = groups.copy()
            rng.shuffle(order)
            n_valid = max(1, round(len(order) * 0.15))
            mapping = {group: "train" for group in order}
            mapping.update({group: "valid" for group in order[:n_valid]})
            mapping.update({group: "test" for group in order[n_valid:2 * n_valid]})
            trial = [{**record, "split": mapping[record["groupId"]]} for record in records]
            if coverage(trial):
                result = trial
                break
        if result is None:
            raise ValueError("Cannot create 10-class group-disjoint holdouts. Add data or provide reviewed group splits.")
    if not coverage(result):
        raise ValueError("Each train/valid/test split must contain all 10 classes.")
    groups_by_split = defaultdict(set)
    for record in result:
        groups_by_split[record["groupId"]].add(record["split"])
    if any(len(splits) > 1 for splits in groups_by_split.values()):
        raise ValueError("A group spans multiple splits.")
    return result


def coverage(records):
    return all({obj["classId"] for row in records if row.get("split") == split for obj in row["objects"]} == set(range(10)) for split in SPLITS)


def crop_box(box, size, margin):
    x, y, width, height = box
    w, h = size
    return (
        max(0, math.floor((x - width * margin) * w)),
        max(0, math.floor((y - height * margin) * h)),
        min(w, math.ceil((x + width * (1 + margin)) * w)),
        min(h, math.ceil((y + height * (1 + margin)) * h)),
    )


DETECTOR_MAX_SIDE = 1280
CROP_MAX_SIDE = 448


def _prepare_one(job):
    row, index, data_root, destination, margin, slugs = job
    split, stem = row["split"], f"{index:06d}_{row['sha256'][:12]}"
    with Image.open(source_path(data_root, row["path"])) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    # Labels are normalized, so downscaling keeps them valid while making every epoch cheaper to decode.
    detector_image = image.copy()
    detector_image.thumbnail((DETECTOR_MAX_SIDE, DETECTOR_MAX_SIDE), Image.Resampling.LANCZOS)
    detector_image.save(destination / "detector" / "images" / split / f"{stem}.jpg", quality=95)
    yolo = []
    for instance, obj in enumerate(row["objects"]):
        x, y, width, height = obj["bbox"]
        yolo.append(f"0 {x + width / 2:.9f} {y + height / 2:.9f} {width:.9f} {height:.9f}")
        crop = image.crop(crop_box(obj["bbox"], image.size, margin))
        if min(crop.size) < 8:
            raise ValueError(f"ROI too small in {row['path']}; review annotation.")
        crop.thumbnail((CROP_MAX_SIDE, CROP_MAX_SIDE), Image.Resampling.LANCZOS)
        crop.save(destination / "classifier" / split / slugs[obj["classId"]] / f"{stem}_{instance}.jpg", quality=95)
    (destination / "detector" / "labels" / split / f"{stem}.txt").write_text("\n".join(yolo) + ("\n" if yolo else ""), encoding="ascii")


def prepare_images(records, data_root, destination, margin, workers=8):
    labels = taxonomy()["labels"]
    for split in SPLITS:
        for sub in ("images", "labels"):
            (destination / "detector" / sub / split).mkdir(parents=True)
        for label in labels:
            (destination / "classifier" / split / label["slug"]).mkdir(parents=True)
    slugs = [label["slug"] for label in labels]
    jobs = [(row, index, data_root, destination, margin, slugs) for index, row in enumerate(records)]
    if workers <= 1:
        for job in jobs:
            _prepare_one(job)
        return
    from multiprocessing import get_context
    with get_context("spawn").Pool(workers) as pool:
        for _ in pool.imap_unordered(_prepare_one, jobs, chunksize=8):
            pass
