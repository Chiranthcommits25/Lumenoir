"""GPU colour-cast statistics for LumenScan and HueLock.

The CPU reference lives in :mod:`backend.classical_cv.colour`. This module uses
the same masks and thresholds; LAB conversion uses the OpenCV 5 trilinear LUT
in :mod:`backend.runtime.torch_ops`, so cast measurements match the CPU path.
"""

from __future__ import annotations

from .colour import (
    HIGH_LIGHT,
    LOW_CHROMA,
    LOW_LIGHT,
    MIN_LEVEL,
    TARGET_BRIGHTNESS,
    ColourCast,
    MidToneLabStats,
)


def mid_tone_lab_stats_tensor(
    rgb,
    *,
    p50: float,
    ops,
    target_brightness: float = TARGET_BRIGHTNESS,
    min_level: float = MIN_LEVEL,
) -> MidToneLabStats:
    """Tensor port of ``colour.mid_tone_lab_stats``."""
    torch = ops.torch
    gain = float(target_brightness) / max(float(p50), float(min_level))
    brightened = torch.clamp(rgb * gain, min=0.0, max=1.0)
    lab = ops.rgb_to_lab(brightened)
    lightness = lab[..., 0]
    a_channel = lab[..., 1]
    b_channel = lab[..., 2]
    chroma = torch.hypot(a_channel, b_channel)
    mask = (
        (lightness >= LOW_LIGHT)
        & (lightness <= HIGH_LIGHT)
        & (chroma < LOW_CHROMA)
    )
    fraction = float(mask.to(torch.float32).mean())
    if not bool(mask.any()):
        return MidToneLabStats(
            mean_a=0.0, mean_b=0.0, std_a=0.0, std_b=0.0, pixel_fraction=0.0
        )

    values_a = a_channel[mask].to(torch.float32)
    values_b = b_channel[mask].to(torch.float32)
    return MidToneLabStats(
        mean_a=float(values_a.mean()),
        mean_b=float(values_b.mean()),
        std_a=float(values_a.std(unbiased=False)),
        std_b=float(values_b.std(unbiased=False)),
        pixel_fraction=fraction,
    )


def detect_colour_cast_tensor(rgb, *, p50: float, ops) -> ColourCast:
    """Tensor port of ``colour.detect_colour_cast``."""
    stats = mid_tone_lab_stats_tensor(rgb, p50=p50, ops=ops)
    magnitude = float((stats.mean_a**2 + stats.mean_b**2) ** 0.5)
    return ColourCast(
        a=stats.mean_a,
        b=stats.mean_b,
        magnitude=magnitude,
        pixel_fraction=stats.pixel_fraction,
    )


__all__ = ["detect_colour_cast_tensor", "mid_tone_lab_stats_tensor"]
