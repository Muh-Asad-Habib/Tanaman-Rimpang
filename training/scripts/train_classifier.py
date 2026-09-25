import argparse
import random
from pathlib import Path

from training.common import ROOT, digest, read_json, taxonomy
from training.evaluation import atomic_json, atomic_write, finite_number, prepared_dataset_binding


def validate_resume(checkpoint, config, class_slugs, dataset_binding, config_sha):
    if (checkpoint.get("formatVersion") != 1 or checkpoint.get("config") != config
            or checkpoint.get("classSlugs") != class_slugs
            or checkpoint.get("datasetBinding") != dataset_binding
            or checkpoint.get("configFileSha256") != config_sha):
        raise ValueError("Resume configuration/classes/prepared dataset differ or provenance is missing; create a new experiment.")
    epoch, history = checkpoint.get("epoch"), checkpoint.get("history")
    if type(epoch) is not int or epoch < 0 or not isinstance(history, list) or len(history) != epoch + 1:
        raise ValueError("Resume epoch/history is incomplete.")
    best, stale = -1.0, 0
    for index, row in enumerate(history):
        if row.get("epoch") != index + 1:
            raise ValueError("Resume history is not contiguous.")
        score = finite_number(row.get("validMacroF1"), "history validation macro-F1", 0, 1)
        stale = 0 if score > best else stale + 1
        best = max(best, score)
    if checkpoint.get("best") != best or type(checkpoint.get("stale")) is not int or checkpoint["stale"] != stale:
        raise ValueError("Resume best/early-stop state does not match history.")


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNetV2-B0 +/- CBAM on grouped ROI data.")
    parser.add_argument("--data", type=Path, required=True, help="prepared/classifier directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "training" / "configs" / "classifier.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path, help="Trusted last.pt from this experiment; restores optimizer and RNG. Pickles are executable.")
    args = parser.parse_args()
    config = read_json(args.config)
    class_slugs = [label["slug"] for label in taxonomy()["labels"]]
    if args.output.exists() and not args.resume:
        parser.error("Output exists; use a new experiment directory.")
    if args.resume and (args.resume.resolve().parent != args.output.resolve() or args.resume.name != "last.pt"):
        parser.error("Resume must use last.pt in the original output directory.")
    if args.data.resolve() != args.data.parent.resolve() / "classifier":
        parser.error("--data must be the classifier directory inside a frozen prepared dataset.")
    dataset_binding, _, inspection = prepared_dataset_binding(args.data.parent)
    if dataset_binding["cropMargin"] != 0.1:
        parser.error("Prepared crops must use the approved margin 0.1.")
    if any(not count for split in inspection["splits"].values() for count in split["objectsPerClass"].values()):
        parser.error("Prepared train/valid/test must all cover the existing ten classes.")
    config_sha = digest(args.config)
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from training.models.efficientnetv2_cbam import RimpangClassifier
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    device = torch.device(args.device)
    size = config["inputSize"]
    normalize = transforms.Normalize(config["mean"], config["std"])
    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.15, 0.15, 0.1, 0.02),
        transforms.ToTensor(), normalize,
    ])
    evaluation = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), normalize])
    train = datasets.ImageFolder(args.data / "train", transform=train_transform)
    valid = datasets.ImageFolder(args.data / "valid", transform=evaluation)
    expected = {slug: index for index, slug in enumerate(class_slugs)}
    if train.class_to_idx != expected or valid.class_to_idx != expected:
        raise ValueError("Folder labels must exactly match shared/labels.json. Missing classes are not allowed.")
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(train, batch_size=config["batchSize"], shuffle=True, num_workers=config["workers"], generator=generator)
    valid_loader = DataLoader(valid, batch_size=config["batchSize"], shuffle=False, num_workers=config["workers"])
    model = RimpangClassifier(config, pretrained=not bool(args.resume)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learningRate"], weight_decay=config["weightDecay"])
    counts = np.bincount(train.targets, minlength=10)
    if np.any(counts == 0):
        raise ValueError("Training set must cover 10 classes.")
    weights = torch.tensor(counts.sum() / (10 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    start, best, stale = 0, -1.0, 0
    history = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume(checkpoint, config, class_slugs, dataset_binding, config_sha)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, best, stale, history = checkpoint["epoch"] + 1, checkpoint["best"], checkpoint["stale"], checkpoint["history"]
        generator.set_state(checkpoint["loaderRng"])
        torch.set_rng_state(checkpoint["torchRng"])
        random.setstate(checkpoint["pythonRng"])
        np.random.set_state(checkpoint["numpyRng"])
        if checkpoint.get("cudaRng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cudaRng"])
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "config.json", config)
    if args.resume:
        if stale == 0:
            # last.pt commits first; repair best.pt after an interrupted best/history write.
            atomic_write(args.output / "best.pt", lambda stream: torch.save(checkpoint, stream))
        else:
            best_checkpoint = torch.load(args.output / "best.pt", map_location="cpu", weights_only=False)
            validate_resume(best_checkpoint, config, class_slugs, dataset_binding, config_sha)
            if (best_checkpoint["best"] != best or best_checkpoint["epoch"] > checkpoint["epoch"]
                    or best_checkpoint["history"] != history[:best_checkpoint["epoch"] + 1]):
                raise ValueError("Best checkpoint does not belong to this resume history.")
            del best_checkpoint
        atomic_json(args.output / "history.json", history)
        if start >= config["epochs"] or stale >= config["patience"]:
            print("Experiment already reached its epoch/early-stopping limit; no extra training performed.", flush=True)
            return
    for epoch in range(start, config["epochs"]):
        model.train()
        total_loss = 0.0
        for batch, targets in train_loader:
            batch, targets = batch.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), targets)
            if not torch.isfinite(loss).item():
                raise ValueError("Non-finite training loss; refusing to publish a corrupt checkpoint.")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
        model.eval()
        confusion = torch.zeros((10, 10), dtype=torch.int64)
        with torch.inference_mode():
            for batch, targets in valid_loader:
                logits = model(batch.to(device))
                if not torch.isfinite(logits).all().item():
                    raise ValueError("Non-finite validation logits.")
                predictions = logits.argmax(1).cpu()
                confusion += torch.bincount(targets * 10 + predictions, minlength=100).reshape(10, 10)
        matrix = confusion.numpy()
        denominator = matrix.sum(0) + matrix.sum(1)
        macro_f1 = float(np.mean(np.divide(2 * matrix.diagonal(), denominator, out=np.zeros(10), where=denominator > 0)))
        improved = macro_f1 > best
        best = max(best, macro_f1)
        stale = 0 if improved else stale + 1
        history.append({"epoch": epoch + 1, "trainLoss": total_loss / len(train), "validMacroF1": macro_f1, "confusion": matrix.tolist()})
        checkpoint = {
            "formatVersion": 1, "config": config, "classSlugs": class_slugs,
            "datasetBinding": dataset_binding, "configFileSha256": config_sha,
            "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "best": best, "stale": stale, "history": history,
            "loaderRng": generator.get_state(), "torchRng": torch.get_rng_state(),
            "pythonRng": random.getstate(), "numpyRng": np.random.get_state(),
            "cudaRng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        atomic_write(args.output / "last.pt", lambda stream: torch.save(checkpoint, stream))
        if improved:
            atomic_write(args.output / "best.pt", lambda stream: torch.save(checkpoint, stream))
        atomic_json(args.output / "history.json", history)
        print(f"epoch={epoch + 1} valid_macro_f1={macro_f1:.4f}", flush=True)
        if stale >= config["patience"]:
            break


if __name__ == "__main__":
    main()
