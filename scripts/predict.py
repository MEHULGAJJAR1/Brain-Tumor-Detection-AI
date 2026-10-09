#!/usr/bin/env python3
"""Run one saved research checkpoint on a local image and print JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.ml.inference import ModelService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="JPEG, PNG or WebP image to analyze.")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "brain_tumor_resnet50.pt", help="Compatible checkpoint path.")
    args = parser.parse_args()
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        parser.error(f"Image file does not exist: {image_path}")
    try:
        result = ModelService(args.model.expanduser().resolve()).predict(image_path)
    except Exception as exc:
        print(f"Prediction unavailable: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        **result,
        "disclaimer": "Research output only; not a diagnosis or a substitute for a clinician.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
