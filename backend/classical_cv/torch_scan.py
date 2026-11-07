"""GPU LumenScan: measurements on one device.

This mirrors :mod:`backend.classical_cv.scan` exactly: the rule functions and
dataclasses are reused, and only the image measurements are tensor operations.
``illumination`` and ``noise_map`` in the returned result are device tensors.
"""

from __future__ import annotations

from .colour import hue_lock_beta
from .scan import (
    BLACK_LEVEL,
    CLIP_LEVEL,
    LumenParameters,
    LumenScanConfig,
    LumenScanFeatures,
    LumenScanResult,
    _classify_darkness,
    _select_clahe,
    _select_dl_strength,
    _select_fusion,
    _select_gain_cap,
    _select_gamma,
)
from .torch_colour import detect_colour_cast_tensor


def _analysis_size(height: int, width: int, max_edge: int) -> tuple[int, int]:
    longest = max(height, width)
    if longest <= max_edge:
        return width, height
    scale = max_edge / float(longest)
    return max(1, round(width * scale)), max(1, round(height * scale))


def scan_tensor(image, config: LumenScanConfig | None = None, *, ops) -> LumenScanResult:
    """Measure a float RGB tensor and return the rule-based parameters."""
    config = config or LumenScanConfig()
    torch = ops.torch
    rgb = image.to(ops.dtype)
    height, width = rgb.shape[:2]

    full_luminance = ops.luma(rgb)
    sigma = ops.noise_sigma(full_luminance, config.noise_flat_percentile)
    local_noise = ops.noise_map(
        full_luminance, config.noise_map_radius, sigma=sigma
    )

    analysis_width, analysis_height = _analysis_size(
        height, width, config.analysis_max_edge
    )
    analysis = ops.resize(
        rgb, (analysis_width, analysis_height), mode="area"
    )
    analysis_luminance = ops.luma(analysis)
    percentiles = ops.quantile(
        analysis_luminance, [0.05, 0.50, 0.95]
    ).to(torch.float32)
    p5, p50, p95 = (float(value) for value in percentiles)
    mean = float(analysis_luminance.mean())
    dynamic_range = float(p95 - p5)

    guided_eps = (
        config.noisy_eps
        if sigma > config.noisy_sigma_threshold
        else config.illumination_eps
    )
    illumination = ops.estimate_illumination(
        analysis,
        eps=guided_eps,
        downsample=config.illumination_downsample,
    )
    nonuniformity = float(
        torch.std(
            torch.log2(illumination + 1.0 / 64.0), unbiased=False
        )
    )

    cast = detect_colour_cast_tensor(analysis, p50=p50, ops=ops)
    max_channels = torch.amax(analysis, dim=2)
    highlight_fraction = float((max_channels >= CLIP_LEVEL).to(torch.float32).mean())
    black_fraction = float((max_channels <= BLACK_LEVEL).to(torch.float32).mean())

    features = LumenScanFeatures(
        mean=mean,
        p5=p5,
        p50=p50,
        p95=p95,
        dynamic_range=dynamic_range,
        nonuniformity=nonuniformity,
        noise_sigma=sigma,
        colour_cast_a=cast.a,
        colour_cast_b=cast.b,
        colour_cast_magnitude=cast.magnitude,
        highlight_fraction=highlight_fraction,
        black_fraction=black_fraction,
        darkness=_classify_darkness(p50, config),
    )

    clahe_clip, clahe_tiles = _select_clahe(
        dynamic_range, sigma, height * width, config
    )
    fusion_contrast, fusion_saturation, fusion_exposure, fusion_noise, fusion_tau = (
        _select_fusion(sigma)
    )

    parameters = LumenParameters(
        gamma=_select_gamma(p50, config),
        gain_cap=_select_gain_cap(sigma, config),
        guided_eps=guided_eps,
        clahe_clip=clahe_clip,
        clahe_tiles=clahe_tiles,
        clahe_blend=config.clahe_blend,
        dl_strength=_select_dl_strength(p50, config),
        fusion_contrast=fusion_contrast,
        fusion_saturation=fusion_saturation,
        fusion_exposure=fusion_exposure,
        fusion_noise=fusion_noise,
        fusion_tau=fusion_tau,
        hue_lock_beta=hue_lock_beta(
            cast,
            threshold=config.hue_lock_threshold,
            scale=config.hue_lock_scale,
            maximum=config.hue_lock_max,
        ),
        output_blend_alpha=float(
            1.0 - ops.smoothstep(0.25, 0.50, torch.tensor(p50, device=ops.device))
        ),
        bypass_gate=(
            p50 >= config.bypass_p50
            and dynamic_range >= config.bypass_dynamic_range
            and highlight_fraction < config.bypass_highlight_fraction
        ),
    )

    return LumenScanResult(
        features=features,
        parameters=parameters,
        illumination=illumination,
        noise_map=local_noise,
        analysis_size=(analysis_width, analysis_height),
    )


__all__ = ["scan_tensor"]
