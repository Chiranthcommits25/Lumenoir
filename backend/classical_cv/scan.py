"""LumenScan: image statistics and rule-based parameter selection.

Implements ``final_decision.md`` §5. Features are measured on a copy downscaled
to about 256 px (except noise, which needs the full-resolution image), and the
rule table turns them into the parameters consumed by later stages. No
enhancement is applied here: LumenRetinex, LumenFuse, EdgeGuard and HueLock are
separate milestones.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from .arrays import as_float_rgb, clip, luminance, smoothstep
from .colour import ColourCast, detect_colour_cast, hue_lock_beta
from .illumination import estimate_illumination
from .noise import noise_map, noise_sigma

ANALYSIS_MAX_EDGE = 256
CLIP_LEVEL = 0.98
BLACK_LEVEL = 0.02

FUSION_LOW_NOISE = (1.0, 0.5, 1.0, 1.0, 0.08)
FUSION_MID_NOISE = (0.7, 0.5, 1.0, 1.5, 0.08)
FUSION_HIGH_NOISE = (0.5, 0.5, 1.0, 2.0, 0.06)


class DarknessClass(StrEnum):
    EXTREME = "extreme"
    DARK = "dark"
    DIM = "dim"
    NORMAL = "normal"


@dataclass(frozen=True)
class LumenScanConfig:
    """Starting coefficients from §5. Rule-based for the MVP; tunable later."""

    analysis_max_edge: int = ANALYSIS_MAX_EDGE
    noise_flat_percentile: float = 50.0
    noise_map_radius: int = 7

    illumination_downsample: int = 2
    illumination_eps: float = 1e-3
    noisy_eps: float = 3e-3
    noisy_sigma_threshold: float = 0.02

    target_brightness: float = 0.45
    gamma_min: float = 0.30
    noise_cap_constant: float = 0.06
    gain_cap_min: float = 3.0
    gain_cap_max: float = 10.0

    clahe_min: float = 1.0
    clahe_max: float = 4.0
    clahe_base: float = 1.0
    clahe_contrast_gain: float = 3.0
    clahe_noise_gain: float = 40.0
    clahe_tiles_small: tuple[int, int] = (8, 8)
    clahe_tiles_large: tuple[int, int] = (12, 12)
    clahe_pixel_threshold: int = 1_000_000
    clahe_blend: float = 0.6

    dl_strength_min: float = 0.4
    dl_strength_slope: float = 0.6
    dl_strength_midpoint: float = 0.15
    dl_strength_span: float = 0.35

    hue_lock_threshold: float = 6.0
    hue_lock_scale: float = 20.0
    hue_lock_max: float = 0.5

    bypass_p50: float = 0.45
    bypass_dynamic_range: float = 0.45
    bypass_highlight_fraction: float = 0.01

    extreme_threshold: float = 0.06
    dark_threshold: float = 0.20
    dim_threshold: float = 0.45


@dataclass(frozen=True)
class LumenScanFeatures:
    """All §5.2 measurements, on the analysis copy except noise."""

    mean: float
    p5: float
    p50: float
    p95: float
    dynamic_range: float
    nonuniformity: float
    noise_sigma: float
    colour_cast_a: float
    colour_cast_b: float
    colour_cast_magnitude: float
    highlight_fraction: float
    black_fraction: float
    darkness: DarknessClass

    @property
    def illumination_label(self) -> str:
        if self.nonuniformity < 1.0:
            return "uniform"
        if self.nonuniformity <= 2.0:
            return "moderate"
        return "strong"

    def as_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean,
            "p5": self.p5,
            "p50": self.p50,
            "p95": self.p95,
            "dynamic_range": self.dynamic_range,
            "nonuniformity": self.nonuniformity,
            "illumination": self.illumination_label,
            "noise_sigma": self.noise_sigma,
            "colour_cast_a": self.colour_cast_a,
            "colour_cast_b": self.colour_cast_b,
            "colour_cast_magnitude": self.colour_cast_magnitude,
            "highlight_fraction": self.highlight_fraction,
            "black_fraction": self.black_fraction,
            "darkness": self.darkness.value,
        }


@dataclass(frozen=True)
class LumenParameters:
    """Parameters selected by the rule table; consumed by later stages."""

    gamma: float
    gain_cap: float
    guided_eps: float
    clahe_clip: float
    clahe_tiles: tuple[int, int]
    clahe_blend: float
    dl_strength: float
    fusion_contrast: float
    fusion_saturation: float
    fusion_exposure: float
    fusion_noise: float
    fusion_tau: float
    hue_lock_beta: float
    output_blend_alpha: float
    bypass_gate: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "gamma": self.gamma,
            "gain_cap": self.gain_cap,
            "guided_eps": self.guided_eps,
            "clahe_clip": self.clahe_clip,
            "clahe_tiles": list(self.clahe_tiles),
            "clahe_blend": self.clahe_blend,
            "dl_strength": self.dl_strength,
            "fusion_contrast": self.fusion_contrast,
            "fusion_saturation": self.fusion_saturation,
            "fusion_exposure": self.fusion_exposure,
            "fusion_noise": self.fusion_noise,
            "fusion_tau": self.fusion_tau,
            "hue_lock_beta": self.hue_lock_beta,
            "output_blend_alpha": self.output_blend_alpha,
            "bypass_gate": self.bypass_gate,
        }


@dataclass(frozen=True)
class LumenScanResult:
    features: LumenScanFeatures
    parameters: LumenParameters
    illumination: np.ndarray
    noise_map: np.ndarray
    analysis_size: tuple[int, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "features": self.features.as_dict(),
            "parameters": self.parameters.as_dict(),
            "analysis_size": list(self.analysis_size),
            "illumination_shape": list(self.illumination.shape),
            "noise_map_shape": list(self.noise_map.shape),
        }


def _analysis_copy(image: np.ndarray, max_edge: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_edge:
        return image

    scale = max_edge / float(longest)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _classify_darkness(p50: float, config: LumenScanConfig) -> DarknessClass:
    if p50 < config.extreme_threshold:
        return DarknessClass.EXTREME
    if p50 < config.dark_threshold:
        return DarknessClass.DARK
    if p50 < config.dim_threshold:
        return DarknessClass.DIM
    return DarknessClass.NORMAL


def _select_gamma(p50: float, config: LumenScanConfig) -> float:
    if p50 >= config.target_brightness:
        return 1.0
    if p50 <= 1e-6:
        return config.gamma_min

    gamma = np.log(config.target_brightness) / np.log(p50)
    return clip(float(gamma), config.gamma_min, 1.0)


def _select_gain_cap(sigma: float, config: LumenScanConfig) -> float:
    if sigma <= 1e-8:
        return config.gain_cap_max
    return clip(
        config.noise_cap_constant / sigma,
        config.gain_cap_min,
        config.gain_cap_max,
    )


def _select_clahe(
    dynamic_range: float,
    sigma: float,
    pixels: int,
    config: LumenScanConfig,
) -> tuple[float, tuple[int, int]]:
    low_contrast = clip(
        (config.target_brightness - dynamic_range) / config.target_brightness,
        0.0,
        1.0,
    )
    value = clip(
        config.clahe_base
        + config.clahe_contrast_gain * low_contrast
        - config.clahe_noise_gain * sigma,
        config.clahe_min,
        config.clahe_max,
    )
    tiles = (
        config.clahe_tiles_large
        if pixels > config.clahe_pixel_threshold
        else config.clahe_tiles_small
    )
    return value, tiles


def _select_dl_strength(p50: float, config: LumenScanConfig) -> float:
    value = 1.0 - config.dl_strength_slope * (
        (p50 - config.dl_strength_midpoint) / config.dl_strength_span
    )
    return clip(value, config.dl_strength_min, 1.0)


def _select_fusion(sigma: float) -> tuple[float, float, float, float, float]:
    if sigma < 0.008:
        return FUSION_LOW_NOISE
    if sigma < 0.02:
        return FUSION_MID_NOISE
    return FUSION_HIGH_NOISE


def scan(
    image: np.ndarray,
    config: LumenScanConfig | None = None,
) -> LumenScanResult:
    """Measure the input and return the rule-based parameters."""
    config = config or LumenScanConfig()
    rgb = as_float_rgb(image)
    height, width = rgb.shape[:2]

    full_luminance = luminance(rgb)
    sigma = noise_sigma(full_luminance, config.noise_flat_percentile)
    local_noise = noise_map(full_luminance, config.noise_map_radius, sigma=sigma)

    analysis = _analysis_copy(rgb, config.analysis_max_edge)
    analysis_luminance = luminance(analysis)
    p5, p50, p95 = (
        float(value)
        for value in np.percentile(analysis_luminance, [5.0, 50.0, 95.0])
    )
    mean = float(analysis_luminance.mean())
    dynamic_range = float(p95 - p5)

    guided_eps = (
        config.noisy_eps if sigma > config.noisy_sigma_threshold else config.illumination_eps
    )
    illumination = estimate_illumination(
        analysis,
        eps=guided_eps,
        downsample=config.illumination_downsample,
    )
    nonuniformity = float(np.std(np.log2(illumination + 1.0 / 64.0)))

    cast: ColourCast = detect_colour_cast(analysis, p50=p50)
    max_channels = analysis.max(axis=2)
    highlight_fraction = float((max_channels >= CLIP_LEVEL).mean())
    black_fraction = float((max_channels <= BLACK_LEVEL).mean())

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
        output_blend_alpha=1.0 - smoothstep(0.25, 0.50, p50),
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
        analysis_size=(analysis.shape[1], analysis.shape[0]),
    )
