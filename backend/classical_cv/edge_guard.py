"""EdgeGuard: post-fusion, gain-aware denoising.

Implements ``final_decision.md`` §3.5:

- luma: effective gain ``G = clip(blur(Y_F,5)/(blur(Y_0,5)+0.02), 1, 30)``,
  ``G_ref = P75(G)``, guided filter with ``eps = (1.2 sigma G_ref)^2`` and a
  gain-dependent blend ``w = clip((G-1.5)/(G_ref-1.5),0,1)^0.8``;
- chroma: bilateral filter on Cr/Cb at half resolution with
  ``sigmaColor = clip(3*255*sigma*sqrt(G_med), 8, 40)`` and ``sigmaSpace=5``.

The module expects an already fused image plus the original low-light image. It
does not perform fusion, and it does not select LumenScan parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .arrays import as_float_rgb, clip, luminance
from .filters import guided_filter
from .scan import LumenScanResult


@dataclass(frozen=True)
class EdgeGuardConfig:
    """Constants from §3.5 and the §3.2 guided-filter table."""

    effective_gain_kernel: int = 5
    effective_gain_epsilon: float = 0.02
    gain_min: float = 1.0
    gain_max: float = 30.0
    weight_threshold: float = 1.5
    weight_exponent: float = 0.8
    luma_radius_small: int = 4
    luma_radius_large: int = 6
    luma_pixel_threshold: int = 1_000_000
    bilateral_diameter: int = 7
    bilateral_sigma_space: float = 5.0
    chroma_downsample: int = 2
    sigma_color_min: float = 8.0
    sigma_color_max: float = 40.0
    sigma_color_gain: float = 3.0


@dataclass(frozen=True)
class EdgeGuardResult:
    image: np.ndarray
    effective_gain: np.ndarray
    luma_weight: np.ndarray
    gain_reference: float
    sigma: float
    sigma_color: float
    luma_radius: int

    def as_dict(self) -> dict[str, object]:
        return {
            "shape": list(self.image.shape),
            "gain_reference": self.gain_reference,
            "sigma": self.sigma,
            "sigma_color": self.sigma_color,
            "luma_radius": self.luma_radius,
            "gain_range": [float(self.effective_gain.min()), float(self.effective_gain.max())],
            "weight_range": [float(self.luma_weight.min()), float(self.luma_weight.max())],
        }


def _validate_kernel(kernel: int) -> int:
    if not isinstance(kernel, (int, np.integer)) or kernel <= 0 or kernel % 2 == 0:
        raise ValueError(f"effective_gain_kernel must be a positive odd integer, got {kernel!r}.")
    return int(kernel)


def effective_gain_map(
    fused: np.ndarray,
    original: np.ndarray,
    config: EdgeGuardConfig | None = None,
) -> np.ndarray:
    """``G = clip(blur(Y_F,5)/(blur(Y_0,5)+0.02), 1, 30)``."""
    config = config or EdgeGuardConfig()
    fused_rgb = as_float_rgb(fused, name="fused")
    original_rgb = as_float_rgb(original, name="original")
    if fused_rgb.shape != original_rgb.shape:
        raise ValueError(
            f"fused {fused_rgb.shape} and original {original_rgb.shape} must match."
        )

    kernel = _validate_kernel(config.effective_gain_kernel)
    fused_luma = luminance(fused_rgb)
    original_luma = luminance(original_rgb)
    blurred_fused = cv2.blur(fused_luma, (kernel, kernel))
    blurred_original = cv2.blur(original_luma, (kernel, kernel))

    gain = blurred_fused / (blurred_original + np.float32(config.effective_gain_epsilon))
    return np.ascontiguousarray(
        np.clip(gain, config.gain_min, config.gain_max).astype(np.float32)
    )


def _luma_denoise(
    fused_luma: np.ndarray,
    gain: np.ndarray,
    sigma: float,
    pixels: int,
    config: EdgeGuardConfig,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    gain_reference = float(np.percentile(gain, 75.0))
    radius = (
        config.luma_radius_large
        if pixels > config.luma_pixel_threshold
        else config.luma_radius_small
    )

    if gain_reference <= config.weight_threshold:
        return fused_luma, np.zeros_like(fused_luma), gain_reference, radius

    weight = np.clip(
        (gain - np.float32(config.weight_threshold))
        / np.float32(gain_reference - config.weight_threshold),
        0.0,
        1.0,
    ) ** np.float32(config.weight_exponent)

    eps = (1.2 * float(sigma) * gain_reference) ** 2
    eps = max(eps, 1e-12)
    smoothed = guided_filter(fused_luma, fused_luma, radius, eps)
    blended = weight * smoothed + (1.0 - weight) * fused_luma
    return np.ascontiguousarray(blended.astype(np.float32)), weight, gain_reference, radius


def _chroma_denoise(
    fused_rgb: np.ndarray,
    gain: np.ndarray,
    sigma: float,
    config: EdgeGuardConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    height, width = fused_rgb.shape[:2]
    ycrcb = cv2.cvtColor(fused_rgb, cv2.COLOR_RGB2YCrCb)

    median_gain = float(np.median(gain))
    sigma_color = float(
        clip(
            config.sigma_color_gain * 255.0 * float(sigma) * np.sqrt(max(median_gain, 1.0)),
            config.sigma_color_min,
            config.sigma_color_max,
        )
    )

    downsample = int(config.chroma_downsample)
    if downsample < 1:
        raise ValueError(f"chroma_downsample must be >= 1, got {downsample!r}.")
    half_size = (max(1, width // downsample), max(1, height // downsample))

    channels: list[np.ndarray] = []
    for index in (1, 2):
        chroma = ycrcb[..., index] * 255.0
        reduced = cv2.resize(chroma, half_size, interpolation=cv2.INTER_AREA)
        filtered = cv2.bilateralFilter(
            reduced,
            d=int(config.bilateral_diameter),
            sigmaColor=sigma_color,
            sigmaSpace=float(config.bilateral_sigma_space),
        )
        restored = cv2.resize(filtered, (width, height), interpolation=cv2.INTER_LINEAR)
        channels.append(restored / 255.0)

    return channels[0], channels[1], sigma_color


def denoise(
    fused: np.ndarray,
    original: np.ndarray,
    *,
    sigma: float | None = None,
    scan_result: LumenScanResult | None = None,
    config: EdgeGuardConfig | None = None,
) -> EdgeGuardResult:
    """Denoise a fused image using the original low-light image for gain.

    Args:
        fused: HxWx3 float32 RGB in [0, 1], the fusion output.
        original: HxWx3 float32 RGB in [0, 1], the input low-light image.
        sigma: LumenScan noise estimate. Taken from ``scan_result`` when omitted.
        scan_result: used only for ``features.noise_sigma``.
        config: constants from §3.5.

    Returns:
        ``EdgeGuardResult`` with the denoised image, effective gain, luma weight
        and the selected constants.
    """
    config = config or EdgeGuardConfig()
    fused_rgb = as_float_rgb(fused, name="fused")
    original_rgb = as_float_rgb(original, name="original")
    if fused_rgb.shape != original_rgb.shape:
        raise ValueError(
            f"fused {fused_rgb.shape} and original {original_rgb.shape} must match."
        )

    if sigma is None:
        if scan_result is None:
            raise ValueError("Either sigma or scan_result must be provided.")
        sigma = float(scan_result.features.noise_sigma)
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError(f"sigma must be finite and non-negative, got {sigma!r}.")

    gain = effective_gain_map(fused_rgb, original_rgb, config)
    fused_luma = luminance(fused_rgb)
    denoised_luma, weight, gain_reference, radius = _luma_denoise(
        fused_luma, gain, sigma, fused_rgb.shape[0] * fused_rgb.shape[1], config
    )
    cr, cb, sigma_color = _chroma_denoise(fused_rgb, gain, sigma, config)

    ycrcb = np.empty_like(fused_rgb)
    ycrcb[..., 0] = denoised_luma
    ycrcb[..., 1] = cr
    ycrcb[..., 2] = cb
    rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    rgb = np.ascontiguousarray(np.clip(rgb, 0.0, 1.0).astype(np.float32))

    if rgb.shape != fused_rgb.shape:
        raise RuntimeError(
            f"EdgeGuard changed the image shape: {fused_rgb.shape} -> {rgb.shape}."
        )

    return EdgeGuardResult(
        image=rgb,
        effective_gain=gain,
        luma_weight=np.ascontiguousarray(weight.astype(np.float32)),
        gain_reference=gain_reference,
        sigma=sigma,
        sigma_color=sigma_color,
        luma_radius=radius,
    )


__all__ = [
    "EdgeGuardConfig",
    "EdgeGuardResult",
    "denoise",
    "effective_gain_map",
]
