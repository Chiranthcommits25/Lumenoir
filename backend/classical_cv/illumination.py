"""LIME-style illumination estimation (``final_decision.md`` §3.1).

``T0 = max_c I_c``, then a guided filter with ``guide = Y`` and
``input = T0``, computed at reduced resolution and upsampled because the map is
smooth. ``T`` is clipped to ``[1e-3, 1]``; ``T >= T0`` is deliberately not
enforced (that would copy ``T0`` noise into the map).
"""

from __future__ import annotations

import cv2
import numpy as np

from .arrays import as_float_rgb, luminance
from .filters import guided_filter

MIN_ILLUMINATION = 1e-3
MAX_ILLUMINATION = 1.0
RADIUS_FRACTION = 0.03


def estimate_illumination(
    rgb: np.ndarray,
    *,
    radius: int | None = None,
    eps: float = 1e-3,
    downsample: int = 2,
) -> np.ndarray:
    """Return the illumination map ``T`` at the input resolution.

    Args:
        rgb: HxWx3 float32 RGB in [0, 1].
        radius: guided-filter radius in working-resolution pixels. When omitted,
            it is ``round(0.03 * min(H, W))`` scaled by ``downsample``.
        eps: guided-filter regularisation.
        downsample: working-resolution divisor (the map is upsampled back).

    Returns:
        HxW float32 illumination map in [1e-3, 1].
    """
    image = as_float_rgb(rgb)
    if not isinstance(downsample, (int, np.integer)) or downsample < 1:
        raise ValueError(f"downsample must be a positive integer, got {downsample!r}.")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps!r}.")

    height, width = image.shape[:2]
    guide = luminance(image)
    t0 = image.max(axis=2)

    if downsample > 1:
        work_size = (max(1, width // downsample), max(1, height // downsample))
        guide_work = cv2.resize(guide, work_size, interpolation=cv2.INTER_AREA)
        t0_work = cv2.resize(t0, work_size, interpolation=cv2.INTER_AREA)
        if radius is None:
            base_radius = max(1, round(RADIUS_FRACTION * min(height, width)))
            work_radius = max(1, round(base_radius / downsample))
        else:
            work_radius = int(radius)
    else:
        guide_work = guide
        t0_work = t0
        work_radius = int(radius) if radius is not None else max(
            1, round(RADIUS_FRACTION * min(height, width))
        )

    smoothed = guided_filter(guide_work, t0_work, work_radius, eps)

    if downsample > 1:
        smoothed = cv2.resize(
            smoothed, (width, height), interpolation=cv2.INTER_LINEAR
        )

    return np.ascontiguousarray(
        np.clip(smoothed, MIN_ILLUMINATION, MAX_ILLUMINATION).astype(np.float32)
    )
