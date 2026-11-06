"""Decoding, resizing, cropping and augmentation.

The conventions follow ``final_decision.md`` §1.5 exactly:

* work in float32 sRGB in [0, 1];
* decode with OpenCV and convert BGR to RGB;
* apply EXIF orientation exactly once;
* resize the long edge of SICE images to 1200 with area interpolation, then
  take random 384x384 crops;
* LOL images take random 384x384 crops; synthetic LOL images are already
  384x384 and are not cropped further;
* augmentation is flips and 90-degree rotations only;
* no colour jitter anywhere.

Cropping raises instead of silently upscaling an image that is too small; the
caller records the skip.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import (
    CROP_SIZE,
    DARKEN_PROBABILITY,
    SICE_LONG_EDGE,
)
from .synthetic import save_png, synth_dark

__all__ = [
    "PreprocessingError",
    "apply_darkening",
    "apply_orientation",
    "augment_views",
    "decode_rgb",
    "load_rgb_float32",
    "random_augment",
    "random_crop",
    "random_crop_pair",
    "resize_long_edge",
    "save_png",
    "to_uint8_rgb",
]


class PreprocessingError(ValueError):
    """Raised when an image cannot be prepared as the pipeline expects."""


def _read_exif_orientation(data: bytes) -> int:
    """Read tag 0x0112 without decoding pixels."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            value = image.getexif().get(0x0112, 1)
    except Exception:
        return 1
    return int(value) if isinstance(value, int) else 1


def apply_orientation(image: np.ndarray, orientation: int) -> np.ndarray:
    """Apply an EXIF orientation value to an RGB array."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise PreprocessingError(f"Expected an HxWx3 image, got {image.shape}.")

    if orientation == 1:
        return image
    if orientation == 2:
        return image[:, ::-1]
    if orientation == 3:
        return image[::-1, ::-1]
    if orientation == 4:
        return image[::-1, :]
    if orientation == 5:
        return np.transpose(image, (1, 0, 2))
    if orientation == 6:
        return np.rot90(image, k=-1)
    if orientation == 7:
        return np.transpose(image, (1, 0, 2))[::-1, ::-1]
    if orientation == 8:
        return np.rot90(image, k=1)
    return image


def decode_rgb(data: bytes) -> np.ndarray:
    """Decode image bytes into an RGB uint8 array, orientation applied once."""
    if not data:
        raise PreprocessingError("Cannot decode an empty buffer.")

    array = cv2.imdecode(
        np.frombuffer(data, dtype=np.uint8),
        cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
    )
    if array is None or array.size == 0:
        raise PreprocessingError("OpenCV could not decode the image data.")

    rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(apply_orientation(rgb, _read_exif_orientation(data)))


def load_rgb_float32(path: Path) -> np.ndarray:
    """Load an image as float32 sRGB in [0, 1]."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise PreprocessingError(f"Could not read '{path}': {exc}") from exc

    rgb = decode_rgb(data)
    return np.ascontiguousarray(rgb.astype(np.float32) / 255.0)


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise PreprocessingError(f"Expected an HxWx3 image, got {array.shape}.")
    if array.dtype == np.uint8:
        return array
    return np.clip(np.round(array * 255.0), 0, 255).astype(np.uint8)


def resize_long_edge(
    image: np.ndarray,
    long_edge: int = SICE_LONG_EDGE,
    *,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Scale so the longest edge equals ``long_edge``; never upscales."""
    if long_edge <= 0:
        raise PreprocessingError(f"long_edge must be positive, got {long_edge}.")

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= long_edge:
        return image

    scale = long_edge / float(longest)
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, target, interpolation=interpolation)


def _crop_offsets(height: int, width: int, size: int, rng: np.random.Generator) -> tuple[int, int]:
    if size <= 0:
        raise PreprocessingError(f"Crop size must be positive, got {size}.")
    if height < size or width < size:
        raise PreprocessingError(
            f"Image is {width}x{height}, smaller than the {size}x{size} crop."
        )

    top = int(rng.integers(0, height - size + 1))
    left = int(rng.integers(0, width - size + 1))
    return top, left


def random_crop(
    image: np.ndarray,
    size: int = CROP_SIZE,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Take a seeded random crop. Raises when the image is too small."""
    generator = rng if rng is not None else np.random.default_rng()
    top, left = _crop_offsets(image.shape[0], image.shape[1], size, generator)
    return image[top : top + size, left : left + size]


def random_crop_pair(
    low: np.ndarray,
    high: np.ndarray,
    size: int = CROP_SIZE,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a low/reference pair with the same coordinates."""
    if low.shape[:2] != high.shape[:2]:
        raise PreprocessingError(
            f"Pair shapes differ: {low.shape[:2]} vs {high.shape[:2]}."
        )

    generator = rng if rng is not None else np.random.default_rng()
    top, left = _crop_offsets(low.shape[0], low.shape[1], size, generator)
    return (
        low[top : top + size, left : left + size],
        high[top : top + size, left : left + size],
    )


def augment_views(image: np.ndarray) -> list[np.ndarray]:
    """Return the eight flip/rotation views, in a fixed order."""
    views: list[np.ndarray] = []
    for flip in (False, True):
        base = image[:, ::-1] if flip else image
        for quarter_turns in range(4):
            views.append(np.ascontiguousarray(np.rot90(base, k=quarter_turns)))
    return views


def random_augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return one of the eight flip/rotation views, chosen by ``rng``."""
    views = augment_views(image)
    return views[int(rng.integers(0, len(views)))]


def apply_darkening(
    image: np.ndarray,
    rng: np.random.Generator,
    probability: float = DARKEN_PROBABILITY,
) -> np.ndarray:
    """Apply the specified darkening augmentation with probability ``p``."""
    if not 0.0 <= probability <= 1.0:
        raise PreprocessingError(f"probability must be in [0, 1], got {probability}.")
    if rng.random() < probability:
        return synth_dark(image, rng)
    return image
