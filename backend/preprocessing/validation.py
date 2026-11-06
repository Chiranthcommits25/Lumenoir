"""Upload validation for JPG/JPEG/PNG images.

This module is foundation code, not research code: it enforces file type, file
size, image dimensions and integrity, so corrupted or spoofed uploads never
reach the enhancement pipeline.

Validation order (cheapest checks first):
    1. filename / extension
    2. declared content type
    3. byte size (empty, too large)
    4. real decoding + integrity (catches spoofed or corrupted files)
    5. extension vs. detected format
    6. dimensions and megapixel ceiling

``validate_image_upload`` returns metadata only. ``validate_and_decode`` also
returns the decoded RGB image so callers do not decode the same payload twice.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from ..config import Settings

# Accepted extensions per *detected* format. Guards against a .jpg file that
# is really a PNG (or any other payload renamed with a supported extension).
EXTENSIONS_BY_FORMAT: dict[str, frozenset[str]] = {
    "JPEG": frozenset({".jpg", ".jpeg"}),
    "PNG": frozenset({".png"}),
}

CONTENT_TYPE_BY_FORMAT: dict[str, str] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
}


class ImageValidationError(Exception):
    """A rejected upload, carrying the HTTP status and a stable error code."""

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class ImageMetadata:
    """Basic information about a validated image."""

    filename: str
    content_type: str
    format: str
    width: int
    height: int
    size_bytes: int
    colour_mode: str

    @property
    def megapixels(self) -> float:
        return round((self.width * self.height) / 1_000_000, 2)

    @property
    def aspect_ratio(self) -> float:
        return round(self.width / self.height, 4) if self.height else 0.0


@dataclass(frozen=True)
class ValidatedUpload:
    """Metadata plus a decoded, detached RGB image."""

    metadata: ImageMetadata
    image: Image.Image


def _probe(data: bytes) -> tuple[str, int, int, str, Image.Image]:
    """Fully decode the payload and return (format, width, height, mode, image).

    ``verify()`` catches most corruption, but it does not decode pixel data,
    so the file is reopened and fully loaded afterwards. Truncated files fail
    the load step instead of producing a half-grey image.
    """
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
    except UnidentifiedImageError as exc:
        raise ImageValidationError(
            "unsupported_format",
            "The uploaded file is not a readable image. Use JPG, JPEG or PNG.",
            415,
        ) from exc
    except Image.DecompressionBombError as exc:
        raise ImageValidationError(
            "image_too_large",
            "The image is too large to process safely.",
            422,
        ) from exc
    except Exception as exc:
        raise ImageValidationError(
            "corrupted_image",
            "The image file appears to be corrupted or incomplete.",
            400,
        ) from exc

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            info = ((image.format or "").upper(), image.width, image.height, image.mode)
            # convert() copies the pixel data, so the result stays valid after
            # the file handle is closed and has no alpha channel.
            detached = image.convert("RGB")
            return (*info, detached)
    except Image.DecompressionBombError as exc:
        raise ImageValidationError(
            "image_too_large",
            "The image is too large to process safely.",
            422,
        ) from exc
    except Exception as exc:
        raise ImageValidationError(
            "corrupted_image",
            "The image file appears to be corrupted or incomplete.",
            400,
        ) from exc


def _validate(
    *,
    filename: str | None,
    content_type: str | None,
    data: bytes,
    settings: Settings,
) -> ValidatedUpload:
    clean_name = Path(filename or "").name
    if not clean_name:
        raise ImageValidationError(
            "missing_filename", "No filename was provided with the upload."
        )

    extension = Path(clean_name).suffix.lower()
    if extension not in settings.allowed_extensions:
        supported = ", ".join(
            sorted(ext.lstrip(".").upper() for ext in settings.allowed_extensions)
        )
        raise ImageValidationError(
            "unsupported_extension",
            f"Unsupported file type '{extension or clean_name}'. Supported formats: {supported}.",
            415,
        )

    declared_type = (content_type or "").split(";")[0].strip().lower()
    if declared_type and declared_type not in settings.allowed_mime_types:
        raise ImageValidationError(
            "unsupported_media_type",
            f"Unsupported content type '{declared_type}'. Supported types: image/jpeg, image/png.",
            415,
        )

    size_bytes = len(data)
    if size_bytes == 0:
        raise ImageValidationError("empty_file", "The uploaded file is empty.")
    if size_bytes > settings.max_upload_bytes:
        limit_mb = settings.max_upload_bytes / (1024 * 1024)
        raise ImageValidationError(
            "file_too_large",
            f"The file exceeds the {limit_mb:.0f} MB upload limit.",
            413,
        )

    detected_format, width, height, colour_mode, image = _probe(data)

    try:
        if detected_format not in EXTENSIONS_BY_FORMAT:
            raise ImageValidationError(
                "unsupported_format",
                f"Detected format '{detected_format or 'unknown'}' is not supported. "
                "Use JPG, JPEG or PNG.",
                415,
            )

        if extension not in EXTENSIONS_BY_FORMAT[detected_format]:
            raise ImageValidationError(
                "format_mismatch",
                f"The file extension '{extension}' does not match the detected "
                f"{detected_format} image.",
                415,
            )

        if width < settings.min_width or height < settings.min_height:
            raise ImageValidationError(
                "image_too_small",
                f"The image is {width}x{height}px. Minimum accepted size is "
                f"{settings.min_width}x{settings.min_height}px.",
                422,
            )

        if width > settings.max_width or height > settings.max_height:
            raise ImageValidationError(
                "image_too_large",
                f"The image is {width}x{height}px. Maximum accepted size is "
                f"{settings.max_width}x{settings.max_height}px.",
                422,
            )

        megapixels = (width * height) / 1_000_000
        if megapixels > settings.max_megapixels:
            raise ImageValidationError(
                "image_too_large",
                f"The image is {megapixels:.1f} MP. Maximum accepted size is "
                f"{settings.max_megapixels:.0f} MP.",
                422,
            )
    except Exception:
        image.close()
        raise

    return ValidatedUpload(
        metadata=ImageMetadata(
            filename=clean_name,
            content_type=declared_type or CONTENT_TYPE_BY_FORMAT[detected_format],
            format=detected_format,
            width=width,
            height=height,
            size_bytes=size_bytes,
            colour_mode=colour_mode,
        ),
        image=image,
    )


def validate_image_upload(
    *,
    filename: str | None,
    content_type: str | None,
    data: bytes,
    settings: Settings,
) -> ImageMetadata:
    """Validate an upload and return its metadata (image is discarded).

    Raises:
        ImageValidationError: with a ``status_code`` and stable ``code``.
    """
    validated = _validate(
        filename=filename, content_type=content_type, data=data, settings=settings
    )
    validated.image.close()
    return validated.metadata


def validate_and_decode(
    *,
    filename: str | None,
    content_type: str | None,
    data: bytes,
    settings: Settings,
) -> ValidatedUpload:
    """Validate an upload and return metadata plus the decoded RGB image.

    The caller owns the returned image and must close it.
    """
    return _validate(
        filename=filename, content_type=content_type, data=data, settings=settings
    )
