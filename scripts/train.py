#!/usr/bin/env python3
"""Train a three-class ResNet50 research classifier from ImageFolder data.

Expected directory layout:
  data/processed/glioma/*.png
  data/processed/meningioma/*.png
  data/processed/pituitary/*.png

For rigorous evaluation, provide --groups-csv with filename,patient_id columns
and split by patient, not by slice. Without it, the script uses stratified image
splits and prints a warning that metrics may be optimistic due to patient leakage.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.ml.model import build_resnet50  # noqa: E402

EXPECTED_CLASSES = {"glioma", "meningioma", "pituitary"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="ImageFolder dataset directory.")
    parser.add_argument("--output", type=Path, default=Path("models/brain_tumor_resnet50.pt"), help="Output checkpoint path.")
    parser.add_argument("--metrics-output", type=Path, default=Path("models/metrics.json"), help="JSON test-metrics path.")
    parser.add_argument("--groups-csv", type=Path, help="Optional CSV with filename and patient_id columns for patient-independent splits.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--backbone-lr", type=float, default=0.001)
    parser.add_argument("--head-lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not initialize from ImageNet weights (avoids first-run download).")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def build_splits(targets, seed: int, groups=None):
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    indices = np.arange(len(targets))
    labels = np.asarray(targets)
    if groups is None:
        train_val, test = train_test_split(
            indices, test_size=0.15, random_state=seed, stratify=labels
        )
        train, val = train_test_split(
            train_val, test_size=(0.15 / 0.85), random_state=seed + 1,
            stratify=labels[train_val],
        )
        return np.asarray(train), np.asarray(val), np.asarray(test), "stratified_image_split"

    groups = np.asarray(groups)
    # Retry shuffled group splits until every partition has all classes. Small
    # datasets may not support this; failing closed is safer than a misleading split.
    for attempt in range(100):
        first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed + attempt)
        train_val, test = next(first.split(indices, labels, groups))
        second = GroupShuffleSplit(n_splits=1, test_size=(0.15 / 0.85), random_state=seed + 1000 + attempt)
        train_local, val_local = next(second.split(train_val, labels[train_val], groups[train_val]))
        train, val = train_val[train_local], train_val[val_local]
        expected = set(np.unique(labels))
        if all(set(np.unique(labels[part])) == expected for part in (train, val, test)):
            return np.asarray(train), np.asarray(val), np.asarray(test), "patient_group_split"
    raise ValueError("Could not create train/validation/test groups containing every class. Review patient IDs or dataset size.")


def load_groups_csv(groups_path: Path, image_folder):
    lookup = {}
    with groups_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"filename", "patient_id"}.issubset(reader.fieldnames):
            raise ValueError("groups CSV needs filename and patient_id columns")
        for row in reader:
            lookup[row["filename"].replace("\\", "/")] = row["patient_id"]
    groups = []
    missing = []
    for path, _label in image_folder.samples:
        relative = Path(path).resolve().relative_to(Path(image_folder.root).resolve()).as_posix()
        patient_id = lookup.get(relative) or lookup.get(Path(relative).name)
        if not patient_id:
            missing.append(relative)
        else:
            groups.append(patient_id)
    if missing:
        raise ValueError(f"groups CSV is missing patient IDs for {len(missing)} image(s), e.g. {missing[0]}")
    return groups


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32:
        raise SystemExit("epochs and batch-size must be positive; image-size must be at least 32")

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Subset
        from torchvision import datasets, transforms
        from sklearn.metrics import classification_report, confusion_matrix
    except (ImportError, RuntimeError) as exc:
        raise SystemExit(f"Optional ML packages could not be imported. Install requirements-ml.txt. Details: {exc}") from exc

    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {data_dir}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available. Use --device cpu or auto.")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size + 32, args.image_size + 32)),
        transforms.RandomResizedCrop(args.image_size, scale=(0.84, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.15),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.12, contrast=0.12),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_dataset = datasets.ImageFolder(str(data_dir), transform=train_transform)
    eval_dataset = datasets.ImageFolder(str(data_dir), transform=eval_transform)
    classes = train_dataset.classes
    if set(classes) != EXPECTED_CLASSES or len(classes) != 3:
        raise SystemExit(f"Expected exactly these class folders: {sorted(EXPECTED_CLASSES)}; found {classes}")
    if len(train_dataset) < 30:
        raise SystemExit("At least 30 readable images are required for this training script.")
    labels = [label for _path, label in train_dataset.samples]
    group_ids = load_groups_csv(args.groups_csv.expanduser().resolve(), train_dataset) if args.groups_csv else None
    train_indices, val_indices, test_indices, split_kind = build_splits(labels, args.seed, group_ids)
    if not args.groups_csv:
        print("WARNING: using a stratified image-level split. Multiple slices from one patient may leak across sets; reported test metrics may be optimistic.", file=sys.stderr)

    use_pretrained = not args.no_pretrained
    try:
        model = build_resnet50(num_classes=len(classes), pretrained=use_pretrained).to(device)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    backbone_params = [parameter for name, parameter in model.named_parameters() if not name.startswith("fc.")]
    head_params = list(model.fc.parameters())
    optimizer = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.head_lr},
        ]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=pin_memory)
    val_loader = DataLoader(Subset(eval_dataset, val_indices), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(Subset(eval_dataset, test_indices), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin_memory)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    for epoch in range(args.epochs):
        # A short head warm-up precedes full-network fine-tuning.
        backbone_trainable = epoch >= 1
        for parameter in backbone_params:
            parameter.requires_grad = backbone_trainable
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * targets.size(0)
            train_correct += (logits.argmax(1) == targets).sum().item()
            train_count += targets.size(0)

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_count = 0
        with torch.inference_mode():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                logits = model(images)
                loss = criterion(logits, targets)
                val_loss_sum += loss.item() * targets.size(0)
                val_correct += (logits.argmax(1) == targets).sum().item()
                val_count += targets.size(0)
        val_loss = val_loss_sum / max(1, val_count)
        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"train loss {train_loss_sum / max(1, train_count):.4f} "
            f"acc {train_correct / max(1, train_count):.3f} | "
            f"val loss {val_loss:.4f} acc {val_correct / max(1, val_count):.3f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        scheduler.step()

    if best_state is None:
        raise SystemExit("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    model.to(device).eval()
    y_true, y_pred = [], []
    with torch.inference_mode():
        for images, targets in test_loader:
            logits = model(images.to(device))
            y_true.extend(targets.numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())

    report = classification_report(y_true, y_pred, labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))).tolist()
    output_path = args.output.expanduser().resolve()
    metrics_path = args.metrics_output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "architecture": "resnet50_mlp_v1",
        "state_dict": best_state,
        "class_names": classes,
        "input_size": args.image_size,
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training": {
            "epochs_requested": args.epochs,
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "batch_size": args.batch_size,
            "pretrained_imagenet": use_pretrained,
            "split_kind": split_kind,
            "train_images": int(len(train_indices)),
            "validation_images": int(len(val_indices)),
            "test_images": int(len(test_indices)),
            "seed": args.seed,
        },
    }
    torch.save(checkpoint, output_path)
    metrics = {
        "warning": "Research-only metrics. Patient independence is only assured when patient group IDs were supplied.",
        "split_kind": split_kind,
        "best_epoch": best_epoch,
        "class_names": classes,
        "confusion_matrix": matrix,
        "classification_report": report,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {output_path}")
    print(f"Saved test metrics: {metrics_path}")
    print("Test accuracy: {:.3f}".format(report.get("accuracy", 0.0)))
    print(f"Split: {split_kind}; best validation epoch: {best_epoch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
