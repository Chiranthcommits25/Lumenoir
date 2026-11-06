"""Array conventions shared by the classical CV primitives.

Every function works on float32 RGB in [0, 1] or on one-channel luminance
arrays, matching ``final_decision.md`` §3: sRGB, no linearisation,
``Y = 0.299R + 0.587G + 0.114B``.
"""

from __future__ import annotations

import numpy as np

_EPSILON = 1e-6


def as_float_rgb(image: np.ndarray, *, name: str = "image") -> np.ndarray:
    """Validate and return a contiguous float32 RGB image in [0, 1]."""
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError(f"{name} is empty.")

    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    elif array.dtype in (np.float16, np.float32, np.float64):
        array = array.astype(np.float32)
    else:
        raise ValueError(
            f"{name} must be uint8 or float, got dtype '{array.dtype}'."
        )

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} must be HxWx3 RGB, got shape {array.shape}.")
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError(f"{name} is too small: {array.shape[:2]}.")

    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    if array.min() < -_EPSILON or array.max() > 1.0 + _EPSILON:
        raise ValueError(
            f"{name} must lie in [0, 1]; got [{array.min():.4f}, {array.max():.4f}]."
        )

    return np.ascontiguousarray(np.clip(array, 0.0, 1.0))


def as_gray(image: np.ndarray, *, name: str = "image") -> np.ndarray:
    """Validate and return a contiguous float32 2-D array."""
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError(f"{name} is empty.")
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {array.shape}.")
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError(f"{name} is too small: {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return np.ascontiguousarray(array)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec. 601 luma, as specified in ``final_decision.md`` §3."""
    array = as_float_rgb(rgb)
    return np.ascontiguousarray(
        (
            0.299 * array[..., 0]
            + 0.587 * array[..., 1]
            + 0.114 * array[..., 2]
        ).astype(np.float32)
    )


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    """Hermite smoothstep clamped to [0, 1]."""
    if edge1 <= edge0:
        raise ValueError(f"edge1 must be greater than edge0, got {edge0} and {edge1}.")
    t = (float(value) - edge0) / (edge1 - edge0)
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def clip(value: float, low: float, high: float) -> float:
    if high < low:
        raise ValueError(f"high must be >= low, got {low} and {high}.")
    return float(min(high, max(low, value)))
