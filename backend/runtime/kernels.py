"""Kernel definitions shared by the torch backends.

These are derived from OpenCV impulse responses so the accelerator path keeps
the same taps as the CPU reference. The module imports NumPy and OpenCV only;
PyTorch is never imported here.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

_K5 = (np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0).astype(
    np.float32
)
PYR_DOWN_KERNEL = np.outer(_K5, _K5).astype(np.float32)
PYR_UP_KERNEL = (4.0 * PYR_DOWN_KERNEL).astype(np.float32)


@lru_cache(maxsize=8)
def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """Derive OpenCV's Gaussian taps for this build and sigma."""
    sigma = float(sigma)
    approximate = max(3, round(sigma * 8.0 + 1.0) | 1)
    size = approximate + 10
    if size % 2 == 0:
        size += 1
    impulse = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    impulse[center, center] = 1.0
    blurred = cv2.GaussianBlur(impulse, (0, 0), sigmaX=sigma)
    row = blurred[center].astype(np.float64)
    peak = float(np.max(np.abs(row)))
    if peak <= 0.0:
        raise RuntimeError("OpenCV returned an empty Gaussian kernel.")
    support = np.nonzero(np.abs(row) > peak * 1e-12)[0]
    taps = row[support.min() : support.max() + 1]
    taps = taps / taps.sum()
    return taps.astype(np.float32)


@lru_cache(maxsize=8)
def laplacian_kernel_2d(ksize: int) -> np.ndarray:
    """Derive OpenCV's ``ksize`` Laplacian kernel from an impulse response."""
    ksize = int(ksize)
    radius = max(1, ksize // 2)
    size = 2 * radius + 3
    impulse = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    impulse[center, center] = 1.0
    response = cv2.Laplacian(impulse, cv2.CV_32F, ksize=ksize)
    kernel = response[
        center - radius : center + radius + 1,
        center - radius : center + radius + 1,
    ]
    return np.ascontiguousarray(kernel, dtype=np.float32)


__all__ = [
    "PYR_DOWN_KERNEL",
    "PYR_UP_KERNEL",
    "gaussian_kernel_1d",
    "laplacian_kernel_2d",
]
