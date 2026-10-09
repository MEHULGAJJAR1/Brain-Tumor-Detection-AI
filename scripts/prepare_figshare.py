#!/usr/bin/env python3
"""Convert Jun Cheng Figshare .mat files into ImageFolder-compatible PNGs.

Label mapping follows the supplied project reference: 1=meningioma, 2=glioma,
3=pituitary. Tumor masks are not exported or used as labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat


LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}


def get_field(value, field: str):
    """Read a MATLAB struct field across scipy's common return representations."""
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    if hasattr(value, field):
        return getattr(value, field)
    if isinstance(value, np.void) and value.dtype.names and field in value.dtype.names:
        return value[field]
    if isinstance(value, dict) and field in value:
        return value[field]
    raise ValueError(f"MAT file is missing cjdata.{field}")


def to_uint8(array: np.ndarray) -> np.ndarray:
    pixels = np.asarray(array).squeeze()
    pixels = np.nan_to_num(pixels, nan=0.0, posinf=0.0, neginf=0.0)
    if pixels.ndim == 3 and pixels.shape[-1] in (3, 4):
        pixels = pixels[..., :3].astype(np.float32)
        low, high = np.percentile(pixels, (1, 99))
        if high <= low:
            return np.zeros(pixels.shape, dtype=np.uint8)
        return np.clip((pixels - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    if pixels.ndim != 2:
        raise ValueError(f"Expected a 2D image or RGB array, got shape {pixels.shape}")
    pixels = pixels.astype(np.float32)
    low, high = np.percentile(pixels, (1, 99))
    if high <= low:
        return np.zeros(pixels.shape, dtype=np.uint8)
    return np.clip((pixels - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)


def convert_file(mat_path: Path, input_root: Path, output_root: Path) -> dict:
    contents = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "cjdata" not in contents:
        raise ValueError("missing cjdata struct")
    cjdata = contents["cjdata"]
    label_value = int(np.asarray(get_field(cjdata, "label")).squeeze())
    class_name = LABELS.get(label_value)
    if not class_name:
        raise ValueError(f"unexpected label {label_value}")
    pixels = to_uint8(get_field(cjdata, "image"))

    relative = mat_path.relative_to(input_root).as_posix()
    suffix = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:8]
    safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in mat_path.stem)[:80]
    filename = f"{safe_stem}_{suffix}.png"
    destination_dir = output_root / class_name
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / filename
    image = Image.fromarray(pixels)
    if image.mode not in {"L", "RGB"}:
        image = image.convert("RGB")
    image.save(destination, format="PNG", optimize=True)
    return {
        "filename": f"{class_name}/{filename}",
        "class_name": class_name,
        "source_file": relative,
        "source_label": label_value,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing the extracted Figshare .mat files (searched recursively).")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Output directory with glioma/meningioma/pituitary subfolders.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output PNGs and manifest.")
    args = parser.parse_args()

    input_root = args.input_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    if not input_root.is_dir():
        parser.error(f"Input directory does not exist: {input_root}")
    files = sorted(input_root.rglob("*.mat"))
    if not files:
        parser.error(f"No .mat files found under {input_root}")

    manifest_path = output_root / "manifest.csv"
    if manifest_path.exists() and not args.overwrite:
        parser.error(f"{manifest_path} already exists. Use --overwrite to replace processed outputs.")
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    for index, mat_path in enumerate(files, 1):
        try:
            rows.append(convert_file(mat_path, input_root, output_root))
        except Exception as exc:  # Continue so one malformed archive member does not discard all work.
            failures.append((mat_path, str(exc)))
        if index % 250 == 0 or index == len(files):
            print(f"Processed {index}/{len(files)} files…", file=sys.stderr)

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("filename", "class_name", "source_file", "source_label"))
        writer.writeheader()
        writer.writerows(rows)

    counts = {name: sum(row["class_name"] == name for row in rows) for name in LABELS.values()}
    print(f"Saved {len(rows)} image(s) to {output_root}")
    print("Class counts: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    print(f"Manifest: {manifest_path}")
    if failures:
        print(f"Skipped {len(failures)} invalid file(s):", file=sys.stderr)
        for path, reason in failures[:20]:
            print(f"  {path}: {reason}", file=sys.stderr)
        if len(failures) > 20:
            print(f"  …and {len(failures) - 20} more", file=sys.stderr)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
