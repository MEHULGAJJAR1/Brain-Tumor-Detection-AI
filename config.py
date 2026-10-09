"""Application configuration sourced from environment variables and .env."""
from __future__ import annotations

import os
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _path_setting(name: str, default: str) -> str:
    value = Path(os.environ.get(name, default)).expanduser()
    if not value.is_absolute():
        value = ROOT / value
    return str(value.resolve())


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
    DATABASE_PATH = _path_setting("DATABASE_PATH", "instance/neurovista.sqlite3")
    UPLOAD_FOLDER = _path_setting("UPLOAD_FOLDER", "instance/uploads")
    MODEL_PATH = _path_setting("MODEL_PATH", "models/brain_tumor_resnet50.pt")
    MAX_UPLOAD_MB = max(1, int(os.environ.get("MAX_UPLOAD_MB", "12")))
    MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
    # Leave room for multipart boundaries/headers while enforcing the actual file limit separately.
    MAX_CONTENT_LENGTH = MAX_UPLOAD_BYTES + 256 * 1024
    MAX_IMAGE_PIXELS = max(1_000_000, int(os.environ.get("MAX_IMAGE_PIXELS", "40000000")))
    JSON_SORT_KEYS = False
    TESTING = False
