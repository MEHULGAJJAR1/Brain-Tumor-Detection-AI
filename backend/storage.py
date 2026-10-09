"""Safe local image ingestion; metadata is stripped before a scan is stored."""
from __future__ import annotations

import hashlib
import io
import warnings
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.utils import secure_filename


ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class UploadError(ValueError):
    def __init__(self, message: str, status_code: int = 400, code: str = "INVALID_UPLOAD"):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


def _display_filename(name: str | None) -> str:
    # secure_filename strips separators and control characters; cap database/UI length.
    cleaned = secure_filename(Path(name or "").name)
    if not cleaned:
        raise UploadError("Choose a file with a valid name.")
    if Path(cleaned).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise UploadError("Upload a JPEG, PNG, or WebP image.", 415, "UNSUPPORTED_FILE_TYPE")
    return cleaned[:120]


def save_image_upload(file_storage, upload_folder: str | Path, max_bytes: int, max_pixels: int) -> dict:
    """Validate, normalize, and persist one image. Returns safe metadata and a UUID."""
    if file_storage is None:
        raise UploadError("Select an image before submitting.")
    original_filename = _display_filename(file_storage.filename)
    raw = file_storage.stream.read(max_bytes + 1)
    if not raw:
        raise UploadError("The selected file is empty.")
    if len(raw) > max_bytes:
        raise UploadError(
            f"Image is larger than the {max_bytes // (1024 * 1024)} MB upload limit.",
            413,
            "FILE_TOO_LARGE",
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                actual_format = (image.format or "").upper()
                if actual_format not in ALLOWED_FORMATS:
                    raise UploadError("The file contents are not a supported JPEG, PNG, or WebP image.", 415, "UNSUPPORTED_FILE_TYPE")
                if getattr(image, "n_frames", 1) != 1:
                    raise UploadError("Animated and multi-frame images are not supported.", 415, "MULTIFRAME_NOT_SUPPORTED")
                width, height = image.size
                if width < 16 or height < 16:
                    raise UploadError("Image dimensions must be at least 16 × 16 pixels.")
                if width * height > max_pixels:
                    raise UploadError("Image dimensions exceed the configured pixel limit.", 413, "IMAGE_TOO_LARGE")
                image.verify()

            with Image.open(io.BytesIO(raw)) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                width, height = image.size
                output = io.BytesIO()
                # Re-encoding strips embedded EXIF/GPS metadata and creates one predictable format.
                image.save(output, format="PNG", optimize=True)
                sanitized = output.getvalue()
    except UploadError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning, OSError, ValueError) as exc:
        raise UploadError("This image could not be decoded. Try a valid, non-animated JPEG, PNG, or WebP file.") from exc

    scan_id = str(uuid4())
    stored_filename = f"{scan_id.replace('-', '')}.png"
    destination = Path(upload_folder)
    destination.mkdir(parents=True, exist_ok=True)
    final_path = destination / stored_filename
    temporary_path = destination / f".{stored_filename}.tmp"
    try:
        temporary_path.write_bytes(sanitized)
        temporary_path.replace(final_path)
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        raise UploadError("The image could not be saved to local storage.", 500, "STORAGE_ERROR") from exc

    return {
        "id": scan_id,
        "original_filename": original_filename,
        "stored_filename": stored_filename,
        "mime_type": "image/png",
        "size_bytes": len(raw),
        "width": width,
        "height": height,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "path": final_path,
    }
