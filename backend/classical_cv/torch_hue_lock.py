"""GPU HueLock.

Same guarded LAB cast shift as :mod:`backend.classical_cv.hue_lock`; the shift
is computed on device tensors and only ``a*``/``b*`` change.
"""

from __future__ import annotations

from dataclasses import dataclass

from .hue_lock import HueLockConfig
from .scan import LumenScanResult
from .torch_colour import mid_tone_lab_stats_tensor


@dataclass(frozen=True)
class TorchHueLockResult:
    image: object
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


def correct_tensor(
    image,
    *,
    scan_result: LumenScanResult,
    config: HueLockConfig | None = None,
    ops,
) -> TorchHueLockResult:
    config = config or HueLockConfig()
    torch = ops.torch
    rgb = image.to(ops.dtype)
    features = scan_result.features
    parameters = scan_result.parameters

    cast_a = float(features.colour_cast_a)
    cast_b = float(features.colour_cast_b)
    cast_magnitude = float(features.colour_cast_magnitude)
    beta_raw = float(parameters.hue_lock_beta)

    stats = mid_tone_lab_stats_tensor(rgb, p50=float(features.p50), ops=ops)
    spread = max(stats.std_a, stats.std_b)
    spread_guarded = spread > config.spread_threshold

    beta = beta_raw / 2.0 if (spread_guarded and config.halve_on_spread) else beta_raw
    if cast_magnitude > 0.0:
        beta = min(beta, float(config.max_shift) / cast_magnitude)
    beta = max(0.0, beta)

    if beta <= 0.0 or cast_magnitude <= 0.0:
        return TorchHueLockResult(
            image=rgb,
            cast_a=cast_a,
            cast_b=cast_b,
            cast_magnitude=cast_magnitude,
            beta_raw=beta_raw,
            beta_applied=0.0,
            shift_magnitude=0.0,
            spread=spread,
            spread_guarded=spread_guarded,
        )

    lab = ops.rgb_to_lab(rgb)
    lab[..., 1] = lab[..., 1] - float(beta * cast_a)
    lab[..., 2] = lab[..., 2] - float(beta * cast_b)
    corrected = torch.clamp(ops.lab_to_rgb(lab), min=0.0, max=1.0).to(rgb.dtype)
    if corrected.shape != rgb.shape:
        raise RuntimeError(
            f"HueLock changed the image shape: {tuple(rgb.shape)} -> "
            f"{tuple(corrected.shape)}."
        )
    return TorchHueLockResult(
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


__all__ = ["TorchHueLockResult", "correct_tensor"]
