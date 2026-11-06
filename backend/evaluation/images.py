"""Image array helpers for evaluation.

Everything is float32 RGB in [0, 1]. Inputs are validated once here so metric
implementations can assume a consistent representation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .types import MetricInputError

_EPSILON = 1e-6


def to_float32_rgb(image: Image.Image | np.ndarray, *, name: str = "image") -> np.ndarray:
    """Convert a PIL image or array to contiguous float32 RGB in [0, 1]."""
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    else:
        array = np.asarray(image)

    if array.size == 0:
        raise MetricInputError(f"{name} is empty.")

    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    elif array.dtype in (np.float32, np.float64, np.float16):
        array = array.astype(np.float32)
    else:
        raise MetricInputError(
            f"{name} must be uint8 or float, got dtype '{array.dtype}'."
        )

    if array.ndim != 3 or array.shape[2] != 3:
        raise MetricInputError(
            f"{name} must be an HxWx3 RGB image, got shape {array.shape}."
        )
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise MetricInputError(f"{name} is too small for evaluation: {array.shape[:2]}.")

    if not np.isfinite(array).all():
        raise MetricInputError(f"{name} contains NaN or infinite values.")
    if array.min() < -_EPSILON or array.max() > 1.0 + _EPSILON:
        raise MetricInputError(
            f"{name} must lie in [0, 1]; got [{array.min():.4f}, {array.max():.4f}]."
        )

    return np.ascontiguousarray(np.clip(array, 0.0, 1.0))


def load_float32_rgb(path: str | Path) -> np.ndarray:
    """Load an image file as float32 RGB in [0, 1], EXIF orientation applied."""
    with Image.open(path) as opened:
        upright = ImageOps.exif_transpose(opened)
        return to_float32_rgb(upright, name=str(path))


def to_gray(image: np.ndarray) -> np.ndarray:
    """Rec. 601 luma, as used by the pipeline (0.299/0.587/0.114)."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise MetricInputError(f"Expected an HxWx3 image, got {array.shape}.")
    return (
        0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]
    ).astype(np.float32)


def align_to_reference_mean(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Scale ``image`` so its mean matches ``reference``.

    ``final_decision.md`` §6.2 asks for a "GT-mean-aligned" PSNR/SSIM variant
    without giving a formula. This is the simplest definition that matches the
    name: one multiplicative scale on all channels, clipped to [0, 1]. A
    zero-mean image is returned unchanged.
    """
    mean_image = float(np.mean(image))
    mean_reference = float(np.mean(reference))
    if mean_image <= _EPSILON:
        return image
    return np.clip(image * (mean_reference / mean_image), 0.0, 1.0).astype(np.float32)
