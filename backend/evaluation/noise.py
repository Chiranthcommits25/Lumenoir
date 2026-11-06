"""Compatibility wrapper for the evaluation noise metrics.

The canonical Immerkaer implementation now lives in
``backend.classical_cv.noise`` so the pipeline and the evaluation framework
share one definition.
"""

from __future__ import annotations

from ..classical_cv.noise import (
    IMMERKAER_CONSTANT,
    NOISE_KERNEL,
    flat_mask,
    gradient_magnitude,
    noise_map,
    noise_response,
    noise_sigma,
    relative_noise,
)

__all__ = [
    "IMMERKAER_CONSTANT",
    "NOISE_KERNEL",
    "flat_mask",
    "gradient_magnitude",
    "noise_map",
    "noise_response",
    "noise_sigma",
    "relative_noise",
]
