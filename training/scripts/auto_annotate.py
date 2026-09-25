"""Experimental automatic annotation: OWLv2 boxes, folder class, leak-safe groups.

Produces a master manifest for prepare_dataset WITHOUT human review. Models trained
from it must be labelled experimental; boxes and folder classes are unverified.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from training.common import read_json, source_path, write_json

QUERIES = ["a photo of a rhizome", "a ginger root", "a turmeric root", "a galangal root", "a root vegetable"]
MAX_SIDE = 960


class Groups:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def build_groups(rows, max_distance):
    """Union rows sharing a Roboflow parent, original photo, pixels, or near-identical dHash."""
    groups = Groups()
    for index, row in enumerate(rows):
        node = f"row:{index:08d}"
        groups.find(node)
        source = row["source"]
        keys = [f"orig:{source['originalPath']}", f"pix:{row['pixelSha256']}"]
        if source.get("parentSuggestion"):
            keys.append(f"parent:{source['id']}:{source['parentSuggestion']['key']}")
        for key in keys:
            groups.union(node, key)
    hashes = [int(row["dHash"], 16) for row in rows]
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if (hashes[i] ^ hashes[j]).bit_count() <= max_distance:
                groups.union(f"row:{i:08d}", f"row:{j:08d}")
    roots = {}
    return [f"g{roots.setdefault(groups.find(f'row:{index:08d}'), len(roots)):05d}" for index in range(len(rows))]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = lambda box: (box[2] - box[0]) * (box[3] - box[1])
    return inter / (area(a) + area(b) - inter + 1e-9), inter / (min(area(a), area(b)) + 1e-9)


def select_boxes(candidates, width, height, *, floor, relative, fallback, limit=5):
    """candidates: [(score, [x1,y1,x2,y2])] in pixels. Returns normalized xywh boxes."""
    candidates = sorted(candidates, key=lambda item: -item[0])
    if not candidates or candidates[0][0] < fallback:
        return [], "none"
    best = candidates[0][0]
    threshold = max(floor, relative * best)
    kept = []
    for score, box in candidates:
        box = [max(0.0, box[0]), max(0.0, box[1]), min(float(width), box[2]), min(float(height), box[3])]
        w, h = box[2] - box[0], box[3] - box[1]
        if w < 0.03 * width or h < 0.03 * height:
            continue
        if kept and score < threshold:
            break
        # Drop overlapping duplicates and parts nested inside an already kept rhizome.
        if any(overlap > 0.5 or contained > 0.85 for overlap, contained in (iou(box, other) for _, other in kept)):
            continue
        kept.append((score, box))
        if len(kept) == limit:
            break
    boxes = [[b[0] / width, b[1] / height, (b[2] - b[0]) / width, (b[3] - b[1]) / height] for _, b in kept]
    boxes = [[round(min(max(v, 0.0), 1.0), 6) for v in box] for box in boxes]
    boxes = [[x, y, min(w, 1 - x), min(h, 1 - y)] for x, y, w, h in boxes]
    return boxes, "confident" if best >= floor else "fallback"


def detect(rows, image_root, device, batch_size, workers):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    name = "google/owlv2-base-patch16-ensemble"
    processor = Owlv2Processor.from_pretrained(name)
    model = Owlv2ForObjectDetection.from_pretrained(name).to(device).eval()

    class Images(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            with Image.open(source_path(image_root, rows[index]["path"])) as image:
                image = image.convert("RGB")
                image.thumbnail((MAX_SIDE, MAX_SIDE))
                return index, image

    loader = DataLoader(Images(), batch_size=batch_size, num_workers=workers,
                        collate_fn=lambda batch: batch)
    results = [None] * len(rows)
    with torch.inference_mode():
        for step, batch in enumerate(loader):
            indices, images = zip(*batch)
            inputs = processor(text=[QUERIES] * len(images), images=list(images), return_tensors="pt").to(device)
            outputs = model(**inputs)
            # OWLv2 predicts on the square padded canvas.
            sizes = torch.tensor([[max(image.size)] * 2 for image in images], device=device)
            processed = processor.post_process_object_detection(outputs, threshold=0.02, target_sizes=sizes)
            for index, image, item in zip(indices, images, processed):
                scale = rows[index]["width"] / image.size[0]
                results[index] = [(float(score), [float(v) * scale for v in box])
                                  for score, box in zip(item["scores"].tolist(), item["boxes"].tolist())]
            if step % 20 == 0:
                print(f"detected {min((step + 1) * batch_size, len(rows))}/{len(rows)}", flush=True)
    return results


def preview(rows, manifest_rows, image_root, output, count=48):
    thumbs = []
    step = max(1, len(manifest_rows) // count)
    for item in manifest_rows[::step][:count]:
        with Image.open(source_path(image_root, item["path"])) as image:
            image = image.convert("RGB")
            image.thumbnail((256, 256))
            draw = ImageDraw.Draw(image)
            for obj in item["objects"]:
                x, y, w, h = obj["bbox"]
                draw.rectangle([x * image.width, y * image.height, (x + w) * image.width, (y + h) * image.height],
                               outline=(255, 0, 0), width=3)
            draw.text((4, 4), item["path"].split("/")[0], fill=(255, 255, 0))
            thumbs.append(image)
    sheet = Image.new("RGB", (8 * 256, math.ceil(len(thumbs) / 8) * 256), (30, 30, 30))
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % 8) * 256, (index // 8) * 256))
    sheet.save(output, quality=85)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path, help="Complete normalized-inventory.json.")
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New directory for manifest.json/report.json/preview.jpg.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--score-floor", type=float, default=0.2)
    parser.add_argument("--relative", type=float, default=0.5)
    parser.add_argument("--fallback", type=float, default=0.05)
    parser.add_argument("--max-hash-distance", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a new versioned directory.")
    inventory = read_json(args.inventory)
    if inventory.get("kind") != "normalized-image-inventory" or inventory.get("status") != "complete":
        parser.error("Inventory must be the complete normalized-image-inventory from audit_raw_dataset.")
    seen, rows, duplicates = set(), [], 0
    for row in sorted(inventory["images"], key=lambda item: item["path"]):
        if row["sha256"] in seen:
            duplicates += 1
            continue
        seen.add(row["sha256"])
        rows.append(row)
    group_ids = build_groups(rows, args.max_hash_distance)
    detections = detect(rows, args.image_root, args.device, args.batch_size, args.workers)
    images, modes, objects = [], Counter(), Counter()
    for row, group, candidates in zip(rows, group_ids, detections):
        boxes, mode = select_boxes(candidates, row["width"], row["height"], floor=args.score_floor,
                                   relative=args.relative, fallback=args.fallback)
        modes[mode] += 1
        if not boxes:
            continue
        objects[len(boxes)] += 1
        images.append({
            "path": row["path"], "sourceId": "drive:" + row["driveId"], "groupId": group,
            "sha256": row["sha256"],
            "objects": [{"classId": row["classProposal"]["classId"], "bbox": box} for box in boxes],
        })
    args.output.mkdir(parents=True)
    write_json(args.output / "manifest.json", {"schemaVersion": 1, "labelsVersion": 1, "images": images})
    report = {
        "schemaVersion": 1, "kind": "automatic-annotation-report", "status": "experimental-unreviewed",
        "detector": "google/owlv2-base-patch16-ensemble", "queries": QUERIES,
        "settings": {key: getattr(args, key) for key in ("score_floor", "relative", "fallback", "max_hash_distance")},
        "inventoryImages": len(inventory["images"]), "exactDuplicatesDropped": duplicates,
        "annotatedImages": len(images), "noDetection": modes["none"], "fallbackImages": modes["fallback"],
        "objectsPerImage": dict(sorted(objects.items())), "groups": len(set(group_ids)),
        "imagesPerClass": dict(Counter(row["path"].split("/")[0] for row in images)),
        "warning": "Boxes come from a zero-shot detector and classes from folder names; nothing was human reviewed.",
    }
    write_json(args.output / "report.json", report)
    preview(rows, images, args.image_root, args.output / "preview.jpg")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
