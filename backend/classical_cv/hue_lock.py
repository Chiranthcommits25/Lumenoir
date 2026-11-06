"""HueLock: LAB colour-cast correction.

Implements ``final_decision.md`` §3.7 using LumenScan's detected cast and raw
``beta``:

1. LumenScan supplies ``(mean a*, mean b*)``, the cast magnitude and the raw
   ``beta = clip((c-6)/20, 0, 0.5)``;
2. if the mid-tone ``a*``/``b*`` spread exceeds 15, ``beta`` is halved
   (a genuinely colourful scene);
3. the applied shift is capped at 8 LAB units;
4. only ``a*`` and ``b*`` are shifted; ``L*`` is untouched, so highlights,
   shadows and overall lightness are preserved, and chroma is not boosted.

The module expects float32 RGB in [0, 1] and returns the same dimensions.
LumenFuse, EdgeGuard and the API are not involved.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .arrays import as_float_rgb
from .colour import mid_tone_lab_stats
from .scan import LumenScanResult, scan


@dataclass(frozen=True)
class HueLockConfig:
    """Guards from §3.7 step 5."""

    spread_threshold: float = 15.0
    max_shift: float = 8.0
    halve_on_spread: bool = True


@dataclass(frozen=True)
class HueLockResult:
    image: np.ndarray
    cast_a: float
    cast_b: float
    cast_magnitude: float
    beta_raw: float
    beta_applied: float
    shift_magnitude: float
    spread: float
    spread_guarded: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "shape": list(self.image.shape),
            "cast": [self.cast_a, self.cast_b],
            "cast_magnitude": self.cast_magnitude,
            "beta_raw": self.beta_raw,
            "beta_applied": self.beta_applied,
            "shift_magnitude": self.shift_magnitude,
            "spread": self.spread,
            "spread_guarded": self.spread_guarded,
        }


def correct(
    image: np.ndarray,
    *,
    scan_result: LumenScanResult | None = None,
    config: HueLockConfig | None = None,
) -> HueLockResult:
    """Apply the guarded LAB cast shift.

    Args:
        image: HxWx3 float32 RGB in [0, 1].
        scan_result: LumenScan output for this image. Computed if omitted.
        config: spread threshold and maximum LAB shift.

    Returns:
        ``HueLockResult``; when the raw beta is zero the input is returned
        unchanged (no LAB round trip).
    """
    config = config or HueLockConfig()
    rgb = as_float_rgb(image)
    result = scan_result or scan(rgb, None)
    features = result.features
    parameters = result.parameters

    cast_a = float(features.colour_cast_a)
    cast_b = float(features.colour_cast_b)
    cast_magnitude = float(features.colour_cast_magnitude)
    beta_raw = float(parameters.hue_lock_beta)

    stats = mid_tone_lab_stats(rgb, p50=float(features.p50))
    spread = max(stats.std_a, stats.std_b)
    spread_guarded = spread > config.spread_threshold

    beta = beta_raw / 2.0 if (spread_guarded and config.halve_on_spread) else beta_raw
    if cast_magnitude > 0.0:
        beta = min(beta, float(config.max_shift) / cast_magnitude)
    beta = max(0.0, beta)

    if beta <= 0.0 or cast_magnitude <= 0.0:
        return HueLockResult(
            image=rgb.copy(),
            cast_a=cast_a,
            cast_b=cast_b,
            cast_magnitude=cast_magnitude,
            beta_raw=beta_raw,
            beta_applied=0.0,
            shift_magnitude=0.0,
            spread=spread,
            spread_guarded=spread_guarded,
        )

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 1] = lab[..., 1] - np.float32(beta * cast_a)
    lab[..., 2] = lab[..., 2] - np.float32(beta * cast_b)
    corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    corrected = np.ascontiguousarray(
        np.clip(corrected, 0.0, 1.0).astype(np.float32)
    )

    if corrected.shape != rgb.shape:
        raise RuntimeError(
            f"HueLock changed the image shape: {rgb.shape} -> {corrected.shape}."
        )

    return HueLockResult(
        image=corrected,
        cast_a=cast_a,
        cast_b=cast_b,
        cast_magnitude=cast_magnitude,
        beta_raw=beta_raw,
        beta_applied=beta,
        shift_magnitude=beta * cast_magnitude,
        spread=spread,
        spread_guarded=spread_guarded,
    )


__all__ = ["HueLockConfig", "HueLockResult", "correct"]
