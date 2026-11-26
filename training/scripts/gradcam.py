import argparse
import hashlib
import io
from pathlib import Path

from training.common import ROOT, digest, read_json, source_path, taxonomy
from training.evaluation import atomic_json


def main():
    parser = argparse.ArgumentParser(
        description="Grad-CAM explanations: heatmap overlays and in-box activation share per class.")
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--classifier", required=True, type=Path, help="Trusted project classifier checkpoint (pickle).")
    parser.add_argument("--output", required=True, type=Path, help="New directory for overlays and gradcam.json.")
    parser.add_argument("--split", default="test", choices=("valid", "test"))
    parser.add_argument("--per-class", type=int, default=5, help="Overlays saved per class; statistics use all objects.")
    parser.add_argument("--whole-image", action="store_true",
                        help="Explain a no-YOLO ablation model on whole images; in-box share uses all GT boxes.")
    parser.add_argument("--config", type=Path, default=ROOT / "training" / "configs" / "export.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a new Grad-CAM directory.")

    import torch
    from PIL import Image
    from training.evaluation import image_tensor, padded_crop
    from training.gradcam import activation_inside, grad_cam, object_box_in_crop, overlay
    from training.models.efficientnetv2_cbam import RimpangClassifier
    margin = read_json(args.config)["cropMargin"]
    checkpoint = torch.load(args.classifier, map_location="cpu", weights_only=False)
    slugs = [label["slug"] for label in taxonomy()["labels"]]
    if checkpoint.get("classSlugs") != slugs:
        raise ValueError("Classifier class order differs from shared/labels.json.")
    config = checkpoint["config"]
    device = torch.device(args.device)
    model = RimpangClassifier(config, pretrained=False).to(device).float().eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    records = [row for row in read_json(args.prepared / "manifest.json")["images"] if row["split"] == args.split]
    args.output.mkdir(parents=True)
    saved = dict.fromkeys(slugs, 0)
    shares = {slug: [] for slug in slugs}
    correct = {slug: [] for slug in slugs}
    for row in records:
        contents = source_path(args.data_root, row["path"]).read_bytes()
        if hashlib.sha256(contents).hexdigest() != row["sha256"]:
            raise ValueError(f"Source image changed since preparation: {row['path']}")
        with Image.open(io.BytesIO(contents)) as original:
            image = original.convert("RGB")
        if args.whole_image:
            classes = {obj["classId"] for obj in row["objects"]}
            if len(classes) != 1:
                continue
            items = [(classes.pop(), (0.0, 0.0, 1.0, 1.0), [tuple(obj["bbox"]) for obj in row["objects"]])]
        else:
            items = [(obj["classId"], tuple(obj["bbox"]), [object_box_in_crop(obj["bbox"], *image.size, margin)])
                     for obj in row["objects"]]
        for number, (target, box, inside_boxes) in enumerate(items):
            tensor, _ = image_tensor(image, config["inputSize"], box=box, margin=margin,
                                     mean=config["mean"], std=config["std"])
            cam, predicted, probabilities = grad_cam(model, torch.from_numpy(tensor).to(device), target)
            share = activation_inside(cam, inside_boxes)
            slug = slugs[target]
            if share is not None:
                shares[slug].append(share)
            correct[slug].append(max(range(10), key=probabilities.__getitem__) == target)
            if saved[slug] < args.per_class:
                view = image if args.whole_image else image.crop(padded_crop(box, *image.size, margin))
                name = f"{slug}_{row['sha256'][:12]}_{number}.jpg"
                overlay(view, cam).save(args.output / name, quality=90)
                saved[slug] += 1
    per_class = {
        slug: {"objects": len(correct[slug]),
               "meanInBoxActivation": sum(shares[slug]) / len(shares[slug]) if shares[slug] else None,
               "top1Accuracy": sum(correct[slug]) / len(correct[slug]) if correct[slug] else None}
        for slug in slugs
    }
    everything = [value for values in shares.values() for value in values]
    atomic_json(args.output / "gradcam.json", {
        "schemaVersion": 1, "split": args.split, "mode": "whole-image" if args.whole_image else "crop",
        "classifierSha256": digest(args.classifier), "cbam": bool(config.get("cbam")),
        "targetLayer": "attention output (CBAM) before global pooling",
        "meanInBoxActivation": sum(everything) / len(everything) if everything else None,
        "perClass": per_class,
        "note": "In-box activation is the share of Grad-CAM mass on the annotated rhizome; higher means the "
                "decision relies on the rhizome rather than background. Compare CBAM vs no-CBAM checkpoints.",
    })
    print(f"Grad-CAM saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
