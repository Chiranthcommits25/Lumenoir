"""Box and guided filters (``final_decision.md`` §3, core primitives).

Both are O(N) and use only OpenCV core functions; no opencv-contrib is needed.
"""

from __future__ import annotations

import cv2
import numpy as np

from .arrays import as_gray


def box_filter(image: np.ndarray, radius: int) -> np.ndarray:
    """Mean filter over a ``(2r+1, 2r+1)`` window with reflected borders."""
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError(f"box_filter expects a 2-D or HxWxC array, got {array.shape}.")
    if not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError(f"radius must be a non-negative integer, got {radius!r}.")

    data = array.astype(np.float32, copy=False)
    if radius == 0:
        return np.ascontiguousarray(data.copy())

    size = 2 * int(radius) + 1
    return cv2.boxFilter(
        data,
        -1,
        (size, size),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )


def guided_filter(
    guide: np.ndarray,
    source: np.ndarray,
    radius: int,
    eps: float,
) -> np.ndarray:
    """Edge-preserving guided filter (He et al.), exactly as specified.

    ``a = (box(I*p) - box(I)*box(p)) / (box(I*I) - box(I)^2 + eps)``
    ``b = box(p) - a*box(I)``
    ``q = box(a)*I + box(b)``
    """
    guide_array = as_gray(guide, name="guide")
    source_array = as_gray(source, name="source")
    if guide_array.shape != source_array.shape:
        raise ValueError(
            f"guide {guide_array.shape} and source {source_array.shape} must match."
        )
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps!r}.")

    mean_guide = box_filter(guide_array, radius)
    mean_source = box_filter(source_array, radius)

    covariance = box_filter(guide_array * source_array, radius) - mean_guide * mean_source
    variance = box_filter(guide_array * guide_array, radius) - mean_guide * mean_guide
    a = covariance / (variance + np.float32(eps))
    b = mean_source - a * mean_guide

    return np.ascontiguousarray(box_filter(a, radius) * guide_array + box_filter(b, radius))
