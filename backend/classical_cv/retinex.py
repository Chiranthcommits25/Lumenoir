"""LumenRetinex: illumination-adaptive classical enhancement.

Implements ``final_decision.md`` §3.1-§3.4 and the highlight-protection layers
from §3.8:

1. illumination ``T`` from the existing LIME-style estimator (§3.1);
2. gain ``G = clip(T^(gamma-1), 1, G_cap)`` with the highlight shoulder and the
   ``1 / max_c`` limit (§3.3);
3. one gain shared by R, G and B;
4. CLAHE on LAB ``L*`` with the LumenScan clip limit and tile grid, blended
   ``L = (1-b) L + b CLAHE(L)`` (§3.4);
5. the soft-knee highlight shoulder (§3.8).

The module is independent of the API. It does not implement LumenFuse,
EdgeGuard or HueLock.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .arrays import as_float_rgb
from .illumination import estimate_illumination
from .scan import LumenParameters, LumenScanResult, scan

LAB_MAX = 100.0
UINT16_MAX = 65535.0


@dataclass(frozen=True)
class LumenRetinexConfig:
    """Geometric and protection constants from §3.3 and §3.8."""

    illumination_downsample: int = 2
    highlight_low: float = 0.65
    highlight_high: float = 0.95
    gain_epsilon: float = 1e-3
    shoulder_start: float = 0.85
    shoulder_width: float = 0.15
    apply_clahe: bool = True
    apply_shoulder: bool = True
    respect_bypass_gate: bool = True


@dataclass(frozen=True)
class LumenRetinexResult:
    """Enhanced RGB image plus the maps used to produce it."""

    image: np.ndarray
    gain: np.ndarray | None
    illumination: np.ndarray | None
    bypassed: bool
    parameters: LumenParameters

    def as_dict(self) -> dict[str, object]:
        return {
            "bypassed": self.bypassed,
            "shape": list(self.image.shape),
            "gain_range": None
            if self.gain is None
            else [float(self.gain.min()), float(self.gain.max())],
            "illumination_shape": None
            if self.illumination is None
            else list(self.illumination.shape),
        }


def compute_gain(
    image: np.ndarray,
    illumination: np.ndarray,
    parameters: LumenParameters,
    config: LumenRetinexConfig | None = None,
) -> np.ndarray:
    """Return the final gain map ``G''(x)`` from §3.3.

    ``G  = clip(T^(gamma-1), 1, G_cap)``
    ``h  = 1 - smoothstep(0.65, 0.95, max_c I)``
    ``G' = 1 + (G-1) h``
    ``G''= min(G', 1 / (max_c I + 1e-3))``
    """
    config = config or LumenRetinexConfig()
    rgb = as_float_rgb(image)
    illum = np.asarray(illumination, dtype=np.float32)

    if illum.shape != rgb.shape[:2]:
        raise ValueError(
            f"illumination shape {illum.shape} must match the image {rgb.shape[:2]}."
        )
    if not np.isfinite(illum).all():
        raise ValueError("illumination contains NaN or infinite values.")
    if illum.min() <= 0.0:
        raise ValueError("illumination must be strictly positive.")
    if not 0.0 < parameters.gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {parameters.gamma!r}.")
    if parameters.gain_cap < 1.0:
        raise ValueError(f"gain_cap must be >= 1, got {parameters.gain_cap!r}.")
    if config.highlight_high <= config.highlight_low:
        raise ValueError("highlight_high must be greater than highlight_low.")
    if config.gain_epsilon <= 0.0:
        raise ValueError("gain_epsilon must be positive.")

    raw_gain = np.power(illum, np.float32(parameters.gamma - 1.0))
    gain = np.clip(raw_gain, 1.0, np.float32(parameters.gain_cap))

    max_channel = rgb.max(axis=2)
    shoulder = highlight_shoulder_weight(max_channel, config)
    attenuated = 1.0 + (gain - 1.0) * shoulder
    limited = np.minimum(attenuated, 1.0 / (max_channel + config.gain_epsilon))
    return np.ascontiguousarray(limited.astype(np.float32))


def _shoulder_values(
    values: np.ndarray,
    config: LumenRetinexConfig,
) -> np.ndarray:
    if config.shoulder_width <= 0.0:
        raise ValueError("shoulder_width must be positive.")

    start = np.float32(config.shoulder_start)
    width = np.float32(config.shoulder_width)
    above = values > start
    compressed = start + width * np.tanh((values - start) / width)
    return np.where(above, compressed, values).astype(np.float32)


def apply_highlight_shoulder(
    image: np.ndarray,
    config: LumenRetinexConfig | None = None,
) -> np.ndarray:
    """Soft-knee shoulder: ``y = x`` up to 0.85, tanh above it."""
    config = config or LumenRetinexConfig()
    rgb = as_float_rgb(image)
    result = _shoulder_values(rgb, config)
    return np.ascontiguousarray(np.clip(result, 0.0, 1.0).astype(np.float32))


def _clahe_rgb(
    image: np.ndarray,
    parameters: LumenParameters,
) -> np.ndarray:
    """CLAHE on LAB L* only; returns float32 RGB without final clipping."""
    rgb = as_float_rgb(image)
    if parameters.clahe_clip <= 0.0:
        raise ValueError(f"clahe_clip must be positive, got {parameters.clahe_clip!r}.")

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness = lab[..., 0]

    lightness_u16 = np.clip(
        np.round(lightness * (UINT16_MAX / LAB_MAX)), 0, UINT16_MAX
    ).astype(np.uint16)
    clahe = cv2.createCLAHE(
        clipLimit=float(parameters.clahe_clip),
        tileGridSize=tuple(int(v) for v in parameters.clahe_tiles),
    )
    equalised = clahe.apply(lightness_u16).astype(np.float32) * (LAB_MAX / UINT16_MAX)

    blend = float(parameters.clahe_blend)
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"clahe_blend must be in [0, 1], got {blend!r}.")
    lab[..., 0] = (1.0 - blend) * lightness + blend * equalised

    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)


def apply_clahe(
    image: np.ndarray,
    parameters: LumenParameters,
) -> np.ndarray:
    """CLAHE on LAB L* only, clipped to a valid RGB image."""
    return np.ascontiguousarray(np.clip(_clahe_rgb(image, parameters), 0.0, 1.0))


def enhance(
    image: np.ndarray,
    *,
    scan_result: LumenScanResult | None = None,
    config: LumenRetinexConfig | None = None,
    illumination: np.ndarray | None = None,
) -> LumenRetinexResult:
    """Run LumenRetinex on one RGB image.

    Args:
        image: HxWx3 float32 RGB in [0, 1].
        scan_result: LumenScan output for this image. Computed if omitted.
        config: geometric/protection constants.
        illumination: optional full-resolution ``T`` override (tests or reuse).

    Returns:
        ``LumenRetinexResult`` with the enhanced image and the gain/illumination
        maps (``None`` when the bypass gate returns the input unchanged).
    """
    config = config or LumenRetinexConfig()
    rgb = as_float_rgb(image)
    result = scan_result or scan(rgb, None)

    if config.respect_bypass_gate and result.parameters.bypass_gate:
        return LumenRetinexResult(
            image=rgb.copy(),
            gain=None,
            illumination=None,
            bypassed=True,
            parameters=result.parameters,
        )

    parameters = result.parameters
    if illumination is None:
        illum = estimate_illumination(
            rgb,
            eps=parameters.guided_eps,
            downsample=config.illumination_downsample,
        )
    else:
        illum = np.asarray(illumination, dtype=np.float32)
        if illum.shape != rgb.shape[:2]:
            raise ValueError(
                f"illumination shape {illum.shape} must match the image {rgb.shape[:2]}."
            )

    gain = compute_gain(rgb, illum, parameters, config)
    corrected = np.clip(rgb * gain[..., None], 0.0, 1.0)

    if config.apply_clahe:
        corrected = _clahe_rgb(corrected, parameters)
    if config.apply_shoulder:
        corrected = _shoulder_values(corrected, config)
    corrected = np.clip(corrected, 0.0, 1.0)

    corrected = np.ascontiguousarray(corrected.astype(np.float32))
    if corrected.shape != rgb.shape:
        raise RuntimeError(
            f"LumenRetinex changed the image shape: {rgb.shape} -> {corrected.shape}."
        )

    return LumenRetinexResult(
        image=corrected,
        gain=gain,
        illumination=illum,
        bypassed=False,
        parameters=parameters,
    )


def gamma_for_illumination(
    illumination: np.ndarray,
    parameters: LumenParameters,
    config: LumenRetinexConfig | None = None,
) -> np.ndarray:
    """Expose the raw ``T^(gamma-1)`` map before clipping (diagnostics)."""
    config = config or LumenRetinexConfig()
    illum = np.asarray(illumination, dtype=np.float32)
    if illum.min() <= 0.0:
        raise ValueError("illumination must be strictly positive.")
    return np.ascontiguousarray(
        np.clip(
            np.power(illum, np.float32(parameters.gamma - 1.0)),
            1.0,
            np.float32(parameters.gain_cap),
        )
    )


def highlight_shoulder_weight(
    max_channel: np.ndarray,
    config: LumenRetinexConfig | None = None,
) -> np.ndarray:
    """``h(x)`` from §3.3, exposed for tests and diagnostics."""
    config = config or LumenRetinexConfig()
    values = np.asarray(max_channel, dtype=np.float32)
    low = np.float32(config.highlight_low)
    high = np.float32(config.highlight_high)
    t = np.clip((values - low) / (high - low), 0.0, 1.0)
    smoothed = t * t * (3.0 - 2.0 * t)
    return np.ascontiguousarray((1.0 - smoothed).astype(np.float32))


__all__ = [
    "LumenRetinexConfig",
    "LumenRetinexResult",
    "apply_clahe",
    "apply_highlight_shoulder",
    "compute_gain",
    "enhance",
    "gamma_for_illumination",
    "highlight_shoulder_weight",
]
