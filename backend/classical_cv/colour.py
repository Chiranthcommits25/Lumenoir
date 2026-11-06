"""LAB colour-cast detection (``final_decision.md`` §3.7, detection only).

This module measures the cast that LumenScan reports. It deliberately does not
correct it: HueLock is a later milestone.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .arrays import as_float_rgb, clip

TARGET_BRIGHTNESS = 0.45
MIN_LEVEL = 0.02
LOW_LIGHT = 25.0
HIGH_LIGHT = 75.0
LOW_CHROMA = 30.0


@dataclass(frozen=True)
class ColourCast:
    """Mean LAB cast over mid-tone, low-chroma pixels."""

    a: float
    b: float
    magnitude: float
    pixel_fraction: float


@dataclass(frozen=True)
class MidToneLabStats:
    """Mean and spread of LAB ``a*``/``b*`` over the mid-tone mask."""

    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    pixel_fraction: float


def mid_tone_lab_stats(
    rgb: np.ndarray,
    *,
    p50: float,
    target_brightness: float = TARGET_BRIGHTNESS,
    min_level: float = MIN_LEVEL,
) -> MidToneLabStats:
    """LAB statistics over mid-tone, low-chroma pixels of a brightened copy."""
    image = as_float_rgb(rgb)
    gain = float(target_brightness) / max(float(p50), float(min_level))
    brightened = np.clip(image * gain, 0.0, 1.0)

    lab = cv2.cvtColor(brightened, cv2.COLOR_RGB2LAB)
    lightness = lab[..., 0]
    a_channel = lab[..., 1]
    b_channel = lab[..., 2]
    chroma = np.hypot(a_channel, b_channel)

    mask = (
        (lightness >= LOW_LIGHT)
        & (lightness <= HIGH_LIGHT)
        & (chroma < LOW_CHROMA)
    )
    fraction = float(mask.mean())
    if not mask.any():
        return MidToneLabStats(
            mean_a=0.0, mean_b=0.0, std_a=0.0, std_b=0.0, pixel_fraction=0.0
        )

    values_a = a_channel[mask]
    values_b = b_channel[mask]
    return MidToneLabStats(
        mean_a=float(values_a.mean()),
        mean_b=float(values_b.mean()),
        std_a=float(values_a.std()),
        std_b=float(values_b.std()),
        pixel_fraction=fraction,
    )


def detect_colour_cast(
    rgb: np.ndarray,
    *,
    p50: float,
    target_brightness: float = TARGET_BRIGHTNESS,
    min_level: float = MIN_LEVEL,
) -> ColourCast:
    """Measure the LAB cast on a temporarily brightened copy.

    A dark input is too noisy to judge colour, so the image is scaled by
    ``target_brightness / max(p50, min_level)`` first, as in §3.7.
    """
    stats = mid_tone_lab_stats(
        rgb,
        p50=p50,
        target_brightness=target_brightness,
        min_level=min_level,
    )
    return ColourCast(
        a=stats.mean_a,
        b=stats.mean_b,
        magnitude=float(np.hypot(stats.mean_a, stats.mean_b)),
        pixel_fraction=stats.pixel_fraction,
    )


def hue_lock_beta(
    cast: ColourCast,
    *,
    threshold: float = 6.0,
    scale: float = 20.0,
    maximum: float = 0.5,
) -> float:
    """Raw ``beta`` from §5.3. The colour-spread guards belong to HueLock."""
    return clip((cast.magnitude - threshold) / scale, 0.0, maximum)
