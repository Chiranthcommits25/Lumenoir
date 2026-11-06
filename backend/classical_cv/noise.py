"""Immerkaer noise estimation (``final_decision.md`` §3.6).

``R = |Y * N|`` with ``N = [[1,-2,1],[-2,4,-2],[1,-2,1]]``.
For Gaussian noise ``E|R| = 6 sigma sqrt(2/pi)``, so
``sigma = sqrt(pi/2)/6 * mean|R|`` over the flattest pixels.

The same implementation backs the evaluation framework, which imports these
functions instead of keeping a second copy.
"""

from __future__ import annotations

import cv2
import numpy as np

from .arrays import as_gray
from .filters import box_filter

NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
IMMERKAER_CONSTANT = float(np.sqrt(np.pi / 2.0) / 6.0)


def noise_response(gray: np.ndarray) -> np.ndarray:
    """Cropped ``|Y * N|`` response used by the estimator."""
    array = as_gray(gray, name="gray")
    response = cv2.filter2D(
        array, -1, NOISE_KERNEL, borderType=cv2.BORDER_REFLECT
    )
    return np.abs(response)[1:-1, 1:-1]


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Sobel magnitude, cropped to match :func:`noise_response`."""
    array = as_gray(gray, name="gray")
    dx = cv2.Sobel(array, cv2.CV_32F, 1, 0, borderType=cv2.BORDER_REFLECT)
    dy = cv2.Sobel(array, cv2.CV_32F, 0, 1, borderType=cv2.BORDER_REFLECT)
    return np.hypot(dx, dy)[1:-1, 1:-1]


def flat_mask(gray: np.ndarray, percentile: float = 50.0) -> np.ndarray:
    """Mask of the flattest ``percentile``% of pixels by Sobel gradient."""
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile!r}.")

    gradient = gradient_magnitude(gray)
    if gradient.size == 0:
        return np.zeros_like(gradient, dtype=bool)

    threshold = np.percentile(gradient, percentile)
    mask = gradient <= threshold
    if not mask.any():
        mask = np.ones_like(gradient, dtype=bool)
    return mask


def noise_sigma(gray: np.ndarray, percentile: float = 50.0) -> float:
    """Estimate the Gaussian noise standard deviation of a grayscale image."""
    array = as_gray(gray, name="gray")
    if array.shape[0] < 3 or array.shape[1] < 3:
        return 0.0

    response = noise_response(array)
    mask = flat_mask(array, percentile)
    if mask.size != response.size:
        mask = mask[: response.shape[0], : response.shape[1]]
        if mask.size != response.size:
            return 0.0
    if not mask.any():
        return 0.0
    return float(IMMERKAER_CONSTANT * float(response[mask].mean()))


def noise_map(
    gray: np.ndarray,
    radius: int = 7,
    *,
    sigma: float | None = None,
) -> np.ndarray:
    """Local noise sigma, optionally clipped to ``[0.5*sigma, 2*sigma]``."""
    array = as_gray(gray, name="gray")
    response = np.abs(
        cv2.filter2D(array, -1, NOISE_KERNEL, borderType=cv2.BORDER_REFLECT)
    )
    local = IMMERKAER_CONSTANT * box_filter(response, radius)

    if sigma is not None and sigma > 0.0:
        local = np.clip(local, 0.5 * float(sigma), 2.0 * float(sigma))
    return np.ascontiguousarray(local.astype(np.float32))


def relative_noise(gray: np.ndarray, percentile: float = 50.0) -> float:
    """Noise sigma divided by the mean luminance of the flat region."""
    array = as_gray(gray, name="gray")
    if array.shape[0] < 3 or array.shape[1] < 3:
        return 0.0

    response = noise_response(array)
    mask = flat_mask(array, percentile)
    if mask.size != response.size or not mask.any():
        return 0.0

    sigma = IMMERKAER_CONSTANT * float(response[mask].mean())
    level = float(array[1:-1, 1:-1][mask].mean())
    if level <= 1e-6:
        return 0.0 if sigma <= 1e-9 else float("inf")
    return sigma / level
