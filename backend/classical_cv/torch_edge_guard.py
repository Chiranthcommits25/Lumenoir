"""GPU EdgeGuard.

The gain-aware luma guided filter and chroma bilateral filter from
:mod:`backend.classical_cv.edge_guard` on device tensors. The bilateral filter
uses the same colour/spatial kernel and differs from OpenCV only by float
round-off.
"""

from __future__ import annotations

from dataclasses import dataclass

from .edge_guard import EdgeGuardConfig
from .scan import LumenScanResult


@dataclass(frozen=True)
class TorchEdgeGuardResult:
    image: object
    effective_gain: object
    luma_weight: object
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
            "gain_range": [
                float(self.effective_gain.min()),
                float(self.effective_gain.max()),
            ],
            "weight_range": [
                float(self.luma_weight.min()),
                float(self.luma_weight.max()),
            ],
        }


def effective_gain_tensor(fused, original, config: EdgeGuardConfig, *, ops):
    torch = ops.torch
    kernel = int(config.effective_gain_kernel)
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError(
            "effective_gain_kernel must be a positive odd integer, "
            f"got {kernel!r}."
        )
    fused_luma = ops.luma(fused)
    original_luma = ops.luma(original)
    blurred_fused = ops.box_blur(fused_luma, kernel)
    blurred_original = ops.box_blur(original_luma, kernel)
    gain = blurred_fused / (blurred_original + float(config.effective_gain_epsilon))
    return torch.clamp(
        gain, min=float(config.gain_min), max=float(config.gain_max)
    ).to(fused.dtype)


def denoise_tensor(
    fused,
    original,
    *,
    sigma: float | None = None,
    scan_result: LumenScanResult | None = None,
    config: EdgeGuardConfig | None = None,
    ops,
) -> TorchEdgeGuardResult:
    config = config or EdgeGuardConfig()
    torch = ops.torch
    fused = fused.to(ops.dtype)
    original = original.to(ops.dtype)
    if fused.shape != original.shape:
        raise ValueError(
            f"fused {tuple(fused.shape)} and original {tuple(original.shape)} "
            "must match."
        )
    if sigma is None:
        if scan_result is None:
            raise ValueError("Either sigma or scan_result must be provided.")
        sigma = float(scan_result.features.noise_sigma)
    sigma = float(sigma)
    if not torch.isfinite(torch.tensor(sigma)) or sigma < 0.0:
        raise ValueError(f"sigma must be finite and non-negative, got {sigma!r}.")

    gain = effective_gain_tensor(fused, original, config, ops=ops)
    fused_luma = ops.luma(fused)
    gain_reference = float(ops.quantile(gain, 0.75))
    pixels = fused.shape[0] * fused.shape[1]
    radius = (
        int(config.luma_radius_large)
        if pixels > config.luma_pixel_threshold
        else int(config.luma_radius_small)
    )

    if gain_reference <= config.weight_threshold:
        denoised_luma = fused_luma
        weight = torch.zeros_like(fused_luma)
    else:
        weight = torch.clamp(
            (gain - float(config.weight_threshold))
            / float(gain_reference - config.weight_threshold),
            min=0.0,
            max=1.0,
        ).pow(float(config.weight_exponent))
        eps = max((1.2 * float(sigma) * gain_reference) ** 2, 1e-12)
        smoothed = ops.guided_filter(fused_luma, fused_luma, radius, eps)
        denoised_luma = weight * smoothed + (1.0 - weight) * fused_luma

    median_gain = float(ops.quantile(gain, 0.5))
    sigma_color = float(
        min(
            max(
                config.sigma_color_gain
                * 255.0
                * float(sigma)
                * max(median_gain, 1.0) ** 0.5,
                config.sigma_color_min,
            ),
            config.sigma_color_max,
        )
    )

    ycrcb = ops.rgb_to_ycrcb(fused)
    downsample = int(config.chroma_downsample)
    if downsample < 1:
        raise ValueError(f"chroma_downsample must be >= 1, got {downsample!r}.")
    half_size = (
        max(1, fused.shape[1] // downsample),
        max(1, fused.shape[0] // downsample),
    )
    channels = []
    for index in (1, 2):
        chroma = ycrcb[..., index] * 255.0
        reduced = ops.resize(chroma, half_size, mode="area")
        filtered = ops.bilateral(
            reduced,
            d=int(config.bilateral_diameter),
            sigma_color=sigma_color,
            sigma_space=float(config.bilateral_sigma_space),
        )
        restored = ops.resize(
            filtered, (fused.shape[1], fused.shape[0]), mode="bilinear"
        )
        channels.append(restored / 255.0)

    combined = torch.stack(
        [denoised_luma, channels[0], channels[1]], dim=2
    ).to(torch.float32)
    rgb = ops.ycrcb_to_rgb(combined)
    rgb = torch.clamp(rgb, min=0.0, max=1.0).to(fused.dtype)
    if rgb.shape != fused.shape:
        raise RuntimeError(
            f"EdgeGuard changed the image shape: {tuple(fused.shape)} -> "
            f"{tuple(rgb.shape)}."
        )
    return TorchEdgeGuardResult(
        image=rgb,
        effective_gain=gain,
        luma_weight=weight,
        gain_reference=gain_reference,
        sigma=sigma,
        sigma_color=sigma_color,
        luma_radius=radius,
    )


__all__ = ["TorchEdgeGuardResult", "denoise_tensor", "effective_gain_tensor"]
