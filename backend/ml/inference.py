"""Lazy CPU-compatible loading and inference for a trained ResNet50 artifact."""
from __future__ import annotations

import logging
import math
from pathlib import Path

from PIL import Image, ImageOps


logger = logging.getLogger(__name__)
CLASS_NAMES = ("glioma", "meningioma", "pituitary")
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ModelUnavailable(RuntimeError):
    def __init__(self, status: dict):
        super().__init__(status.get("message", "Model is not ready."))
        self.status = status


class ModelService:
    """Loads a checkpoint on demand and caches it until the file changes."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self._model = None
        self._torch = None
        self._transform = None
        self._class_names = CLASS_NAMES
        self._input_size = IMAGE_SIZE
        self._mean = IMAGENET_MEAN
        self._std = IMAGENET_STD
        self._loaded_mtime = None
        self._load_error = None
        self._error_mtime = None

    def status(self) -> dict:
        if not self.model_path.is_file():
            self._model = None
            self._loaded_mtime = None
            return {
                "ready": False,
                "state": "missing_artifact",
                "engine": "ResNet50",
                "classes": list(CLASS_NAMES),
                "artifact": self.model_path.name,
                "message": "No trained model weights are installed. Uploads can be saved, but no classification will be generated.",
                "setup_hint": "Train a checkpoint with scripts/train.py or set MODEL_PATH to a compatible checkpoint.",
            }

        try:
            self._ensure_loaded()
        except Exception as exc:
            message = str(exc).lower()
            missing_dependencies = (
                isinstance(exc, (ModuleNotFoundError, ImportError))
                or "no module named" in message
                or "optional ml dependencies are unavailable" in message
            )
            if missing_dependencies:
                return {
                    "ready": False,
                    "state": "dependencies_missing",
                    "engine": "ResNet50",
                    "classes": list(CLASS_NAMES),
                    "artifact": self.model_path.name,
                    "message": "PyTorch inference dependencies are not installed or are incompatible.",
                    "setup_hint": "Install a compatible PyTorch/torchvision pair and requirements-ml.txt.",
                }
            logger.warning("Model checkpoint is not ready: %s", exc)
            return {
                "ready": False,
                "state": "invalid_artifact",
                "engine": "ResNet50",
                "classes": list(CLASS_NAMES),
                "artifact": self.model_path.name,
                "message": "The configured model artifact could not be loaded.",
                "setup_hint": "Use a checkpoint produced by scripts/train.py; see models/README.md.",
            }

        return {
            "ready": True,
            "state": "ready",
            "engine": "ResNet50",
            "classes": list(self._class_names),
            "input_size": self._input_size,
            "device": "cpu",
            "message": "A compatible research checkpoint is loaded for CPU inference.",
            "setup_hint": None,
        }

    def _ensure_loaded(self) -> None:
        mtime = self.model_path.stat().st_mtime_ns
        if self._model is not None and self._loaded_mtime == mtime:
            return
        if self._error_mtime == mtime and self._load_error is not None:
            raise RuntimeError(self._load_error)

        try:
            import torch
            from torchvision import transforms
            from .model import build_resnet50

            try:
                checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
            except TypeError:  # PyTorch versions before weights_only was introduced.
                checkpoint = torch.load(self.model_path, map_location="cpu")

            if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
                raise ValueError("Checkpoint needs a state_dict and class_names metadata.")
            class_names = tuple(checkpoint.get("class_names", ()))
            if len(class_names) != len(CLASS_NAMES) or set(class_names) != set(CLASS_NAMES):
                raise ValueError("Checkpoint class_names must be glioma, meningioma, and pituitary.")
            architecture = checkpoint.get("architecture", "resnet50_mlp_v1")
            if architecture != "resnet50_mlp_v1":
                raise ValueError("Checkpoint architecture is not resnet50_mlp_v1.")
            input_size = int(checkpoint.get("input_size", IMAGE_SIZE))
            if input_size < 32 or input_size > 1024:
                raise ValueError("Checkpoint input_size must be between 32 and 1024.")
            normalization = checkpoint.get("normalization", {})
            if not isinstance(normalization, dict):
                raise ValueError("Checkpoint normalization metadata is invalid.")
            mean = tuple(float(value) for value in normalization.get("mean", IMAGENET_MEAN))
            std = tuple(float(value) for value in normalization.get("std", IMAGENET_STD))
            if len(mean) != 3 or len(std) != 3 or not all(math.isfinite(value) for value in mean + std) or any(value <= 0 for value in std):
                raise ValueError("Checkpoint normalization metadata is invalid.")

            state_dict = checkpoint["state_dict"]
            if not isinstance(state_dict, dict):
                raise ValueError("Checkpoint state_dict is invalid.")
            if state_dict and all(key.startswith("module.") for key in state_dict):
                state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

            model = build_resnet50(num_classes=len(class_names), pretrained=False)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            transform = transforms.Compose(
                [
                    transforms.Resize((input_size, input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

            self._torch = torch
            self._model = model
            self._transform = transform
            self._class_names = class_names
            self._input_size = input_size
            self._mean = mean
            self._std = std
            self._loaded_mtime = mtime
            self._load_error = None
            self._error_mtime = None
        except Exception as exc:
            self._model = None
            self._loaded_mtime = None
            self._load_error = f"{type(exc).__name__}: {exc}"
            self._error_mtime = mtime
            raise

    def predict(self, image_path: str | Path) -> dict:
        status = self.status()
        if not status["ready"]:
            raise ModelUnavailable(status)
        assert self._model is not None and self._torch is not None and self._transform is not None

        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self._transform(image).unsqueeze(0)

        with self._torch.inference_mode():
            logits = self._model(tensor)
            probabilities = self._torch.softmax(logits, dim=1)[0].cpu().tolist()

        if len(probabilities) != len(self._class_names) or not all(
            math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0 for value in probabilities
        ):
            raise RuntimeError("The model returned invalid probability values.")
        ranked = {name: float(probabilities[index]) for index, name in enumerate(self._class_names)}
        label = max(ranked, key=ranked.get)
        return {
            "prediction_class": label,
            "confidence": ranked[label],
            "probabilities": ranked,
        }
