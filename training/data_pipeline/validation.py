"""Image validation: corruption, truncation and format detection.

Two independent decoders are used on purpose. Pillow's ``verify()`` catches
structural damage, a second full load catches truncated pixel data, and an
OpenCV decode confirms the payload is really readable by the pipeline that will
consume it. A file that passes one decoder and fails the other is reported.

Invalid files are reported and excluded from the manifests; they are never
deleted or modified.
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import IMAGE_EXTENSIONS
from .organisation import ImageRecord

EXTENSIONS_BY_FORMAT: Mapping[str, frozenset[str]] = {
    "JPEG": frozenset({".jpg", ".jpeg"}),
    "PNG": frozenset({".png"}),
    "BMP": frozenset({".bmp"}),
    "TIFF": frozenset({".tif", ".tiff"}),
    "WEBP": frozenset({".webp"}),
}

# Stable reason codes, mirroring the API's error-code style.
EMPTY_FILE = "empty_file"
UNREADABLE = "unreadable"
CORRUPTED = "corrupted_image"
TRUNCATED = "truncated_image"
UNSUPPORTED_FORMAT = "unsupported_format"
EXTENSION_MISMATCH = "extension_mismatch"
INVALID_DIMENSIONS = "invalid_dimensions"
MISSING_FILE = "missing_file"


@dataclass(frozen=True)
class ImageProbe:
    """The result of decoding one file with both decoders."""

    path: Path
    ok: bool
    code: str | None = None
    detail: str | None = None
    width: int | None = None
    height: int | None = None
    channels: int | None = None
    mode: str | None = None
    detected_format: str | None = None
    exif_orientation: int = 1
    size_bytes: int = 0
    bit_depth: int | None = None


@dataclass(frozen=True)
class RecordProblem:
    """A rejected record, with the side that failed."""

    record_id: str
    path: Path
    is_reference: bool
    code: str
    detail: str


@dataclass(frozen=True)
class ValidationOutcome:
    valid: tuple[ImageRecord, ...]
    problems: tuple[RecordProblem, ...]
    probes: tuple[ImageProbe, ...]

    @property
    def invalid_ids(self) -> frozenset[str]:
        return frozenset(problem.record_id for problem in self.problems)

    @property
    def valid_ids(self) -> frozenset[str]:
        return frozenset(record.record_id for record in self.valid)


def _failure(
    path: Path,
    code: str,
    detail: str,
    *,
    size_bytes: int = 0,
    width: int | None = None,
    height: int | None = None,
    mode: str | None = None,
    detected_format: str | None = None,
    exif_orientation: int = 1,
) -> ImageProbe:
    return ImageProbe(
        path=path,
        ok=False,
        code=code,
        detail=detail,
        width=width,
        height=height,
        mode=mode,
        detected_format=detected_format,
        size_bytes=size_bytes,
        exif_orientation=exif_orientation,
    )


def _orientation(image: Image.Image) -> int:
    try:
        value = image.getexif().get(0x0112, 1)
    except Exception:  # pragma: no cover - malformed EXIF is not fatal
        return 1
    return int(value) if isinstance(value, int) else 1


def probe_image(path: Path) -> ImageProbe:
    """Decode ``path`` with Pillow and OpenCV and report the first failure."""
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        return _failure(path, UNREADABLE, f"cannot stat file: {exc}")

    if size_bytes == 0:
        return _failure(path, EMPTY_FILE, "file is empty", size_bytes=0)

    try:
        data = path.read_bytes()
    except OSError as exc:
        return _failure(path, UNREADABLE, f"cannot read file: {exc}", size_bytes=size_bytes)

    width = height = None
    mode = None
    detected_format = None
    orientation = 1

    try:
        with Image.open(io.BytesIO(data)) as image:
            # Pillow 13 requires verify() directly after open, before metadata.
            image.verify()
    except Image.DecompressionBombError:
        return _failure(
            path, INVALID_DIMENSIONS, "image exceeds the decompression-bomb limit",
            size_bytes=size_bytes,
        )
    except UnidentifiedImageError:
        return _failure(
            path, UNSUPPORTED_FORMAT, "Pillow cannot identify the image format",
            size_bytes=size_bytes,
        )
    except OSError as exc:
        code = TRUNCATED if "truncated" in str(exc).lower() else CORRUPTED
        return _failure(path, code, str(exc), size_bytes=size_bytes)
    except Exception as exc:  # pragma: no cover - defensive
        return _failure(path, CORRUPTED, str(exc), size_bytes=size_bytes)

    # ``verify()`` does not decode pixel data, so read the metadata and load it.
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            mode = image.mode
            detected_format = (image.format or "").upper() or None
            orientation = _orientation(image)
            image.load()
    except OSError as exc:
        code = TRUNCATED if "truncated" in str(exc).lower() else CORRUPTED
        return _failure(
            path, code, str(exc),
            size_bytes=size_bytes, width=width, height=height, mode=mode,
            detected_format=detected_format, exif_orientation=orientation,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return _failure(
            path, CORRUPTED, str(exc),
            size_bytes=size_bytes, width=width, height=height, mode=mode,
            detected_format=detected_format, exif_orientation=orientation,
        )

    try:
        array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception as exc:  # pragma: no cover - defensive
        return _failure(
            path, CORRUPTED, f"OpenCV decode failed: {exc}",
            size_bytes=size_bytes, width=width, height=height, mode=mode,
            detected_format=detected_format, exif_orientation=orientation,
        )

    if array is None or array.size == 0:
        return _failure(
            path, CORRUPTED, "OpenCV could not decode the pixel data",
            size_bytes=size_bytes, width=width, height=height, mode=mode,
            detected_format=detected_format, exif_orientation=orientation,
        )

    if width is None or height is None or width <= 0 or height <= 0:
        return _failure(
            path, INVALID_DIMENSIONS, "image has no pixels",
            size_bytes=size_bytes, mode=mode, detected_format=detected_format,
        )

    suffix = path.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        return _failure(
            path, UNSUPPORTED_FORMAT, f"unsupported extension '{suffix}'",
            size_bytes=size_bytes, width=width, height=height, mode=mode,
            detected_format=detected_format, exif_orientation=orientation,
        )

    allowed = EXTENSIONS_BY_FORMAT.get(detected_format or "")
    if allowed is None:
        return _failure(
            path, UNSUPPORTED_FORMAT, f"detected format '{detected_format}' is not supported",
            size_bytes=size_bytes, width=width, height=height, mode=mode,
            detected_format=detected_format, exif_orientation=orientation,
        )
    if suffix not in allowed:
        return _failure(
            path, EXTENSION_MISMATCH,
            f"extension '{suffix}' does not match detected format '{detected_format}'",
            size_bytes=size_bytes, width=width, height=height, mode=mode,
            detected_format=detected_format, exif_orientation=orientation,
        )

    channels = 1 if array.ndim == 2 else int(array.shape[2])
    return ImageProbe(
        path=path,
        ok=True,
        width=width,
        height=height,
        channels=channels,
        mode=mode,
        detected_format=detected_format,
        exif_orientation=orientation,
        size_bytes=size_bytes,
        bit_depth=array.dtype.itemsize * 8,
    )


def validate_records(
    records: Iterable[ImageRecord],
    *,
    limit: int | None = None,
) -> ValidationOutcome:
    """Validate every record; a record is valid when both its images decode."""
    cache: dict[Path, ImageProbe] = {}
    valid: list[ImageRecord] = []
    problems: list[RecordProblem] = []

    for index, record in enumerate(records):
        if limit is not None and index >= limit:
            break

        record_ok = True
        for path, is_reference in (
            (record.low_path, False),
            (record.high_path, True),
        ):
            if path is None:
                continue
            if not path.is_file():
                problems.append(
                    RecordProblem(
                        record.record_id, path, is_reference, MISSING_FILE, "file does not exist"
                    )
                )
                record_ok = False
                continue

            probe = cache.get(path)
            if probe is None:
                probe = probe_image(path)
                cache[path] = probe

            if not probe.ok:
                problems.append(
                    RecordProblem(
                        record.record_id,
                        path,
                        is_reference,
                        probe.code or CORRUPTED,
                        probe.detail or "image could not be decoded",
                    )
                )
                record_ok = False

        if record_ok:
            valid.append(record)

    return ValidationOutcome(
        valid=tuple(valid),
        problems=tuple(problems),
        probes=tuple(cache.values()),
    )
