"""LumenFuse: noise-aware Laplacian-pyramid fusion.

Implements ``final_decision.md`` §4. Candidates are the original image, the
LumenRetinex output and the Zero-DCE++ output. Weights are computed from
contrast, saturation, well-exposedness and a gain-aware noise penalty, refined
with a guided filter, and blended in a Laplacian pyramid. OpenCV's
``createMergeMertens`` is deliberately not used because it has no noise term and
no guided weight refinement.

Only fusion is implemented here. EdgeGuard, HueLock and the API pipeline are
separate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..classical_cv.arrays import as_float_rgb, luminance
from ..classical_cv.edge_guard import EdgeGuardConfig, effective_gain_map
from ..classical_cv.filters import guided_filter
from ..classical_cv.scan import LumenParameters, LumenScanResult, scan

CANDIDATE_COUNT = 3
EPSILON = 1e-6


@dataclass(frozen=True)
class LumenFuseConfig:
    """Constants from §4.3 and the guided-refinement table in §3.2."""

    epsilon: float = EPSILON
    contrast_blur_sigma: float = 1.0
    laplacian_kernel: int = 3
    exposure_target: float = 0.5
    exposure_sigma: float = 0.2

    effective_gain_kernel: int = 5
    effective_gain_epsilon: float = 0.02
    gain_min: float = 1.0
    gain_max: float = 30.0

    weight_guided_radius: int = 8
    weight_guided_eps: float = 1e-2

    max_levels: int = 5
    highlight_threshold: float = 0.85
    highlight_multiplier: float = 3.0


@dataclass(frozen=True)
class FusionWeights:
    """Initial normalized weights and the guided-refined, renormalized ones."""

    initial: np.ndarray
    refined: np.ndarray
    parameters: LumenParameters
    sigma: float

    def as_dict(self) -> dict[str, object]:
        return {
            "sigma": self.sigma,
            "initial_sum_range": [
                float(self.initial.sum(axis=2).min()),
                float(self.initial.sum(axis=2).max()),
            ],
            "refined_sum_range": [
                float(self.refined.sum(axis=2).min()),
                float(self.refined.sum(axis=2).max()),
            ],
        }


@dataclass(frozen=True)
class LumenFuseResult:
    image: np.ndarray
    weights: np.ndarray
    levels: int
    parameters: LumenParameters
    sigma: float
    initial_weights: np.ndarray | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "shape": list(self.image.shape),
            "levels": self.levels,
            "sigma": self.sigma,
            "weight_sum_range": [
                float(self.weights.sum(axis=2).min()),
                float(self.weights.sum(axis=2).max()),
            ],
            "weight_range": [
                [float(self.weights[..., k].min()), float(self.weights[..., k].max())]
                for k in range(CANDIDATE_COUNT)
            ],
        }


def _validate_config(config: LumenFuseConfig) -> None:
    if config.epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {config.epsilon!r}.")
    if config.contrast_blur_sigma <= 0.0:
        raise ValueError("contrast_blur_sigma must be positive.")
    if config.exposure_sigma <= 0.0:
        raise ValueError("exposure_sigma must be positive.")
    if config.laplacian_kernel <= 0 or config.laplacian_kernel % 2 == 0:
        raise ValueError("laplacian_kernel must be positive and odd.")
    if config.effective_gain_kernel <= 0 or config.effective_gain_kernel % 2 == 0:
        raise ValueError("effective_gain_kernel must be positive and odd.")
    if config.effective_gain_epsilon <= 0.0:
        raise ValueError("effective_gain_epsilon must be positive.")
    if config.gain_max < config.gain_min or config.gain_min < 1.0:
        raise ValueError("gain bounds must satisfy 1 <= gain_min <= gain_max.")
    if config.weight_guided_radius < 0:
        raise ValueError("weight_guided_radius must be non-negative.")
    if config.weight_guided_eps <= 0.0:
        raise ValueError("weight_guided_eps must be positive.")
    if config.max_levels < 0:
        raise ValueError("max_levels must be non-negative.")
    if not 0.0 < config.highlight_threshold <= 1.0:
        raise ValueError("highlight_threshold must be in (0, 1].")
    if config.highlight_multiplier < 1.0:
        raise ValueError("highlight_multiplier must be >= 1.")


def _validate_parameters(parameters: LumenParameters) -> None:
    exponents = (
        parameters.fusion_contrast,
        parameters.fusion_saturation,
        parameters.fusion_exposure,
        parameters.fusion_noise,
    )
    if any(not np.isfinite(value) or value < 0.0 for value in exponents):
        raise ValueError("Fusion exponents must be finite and non-negative.")
    if not np.isfinite(parameters.fusion_tau) or parameters.fusion_tau <= 0.0:
        raise ValueError("fusion_tau must be finite and positive.")


def normalize_weights(weights: np.ndarray, *, epsilon: float = 0.0) -> np.ndarray:
    """Normalize per-pixel candidate weights with safe zero-sum handling."""
    array = np.asarray(weights, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != CANDIDATE_COUNT:
        raise ValueError(f"weights must be HxWx{CANDIDATE_COUNT}, got {array.shape}.")
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative.")
    if not np.isfinite(array).all():
        raise ValueError("weights contain NaN or infinite values.")

    numerator = np.clip(array, 0.0, None) + epsilon
    denominator = numerator.sum(axis=2, keepdims=True)
    zero = denominator[..., 0] <= 0.0

    result = np.empty_like(numerator)
    result[~zero] = numerator[~zero] / denominator[~zero]
    result[zero] = 1.0 / CANDIDATE_COUNT
    return result


def pyramid_levels(width: int, height: int, max_levels: int = 5) -> int:
    """Top pyramid index ``L = min(max_levels, floor(log2(min(H,W))) - 3)``."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")
    if max_levels < 0:
        raise ValueError("max_levels must be non-negative.")
    computed = math.floor(math.log2(min(width, height))) - 3
    return max(0, min(int(max_levels), computed))


def gaussian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    """Gaussian pyramid with ``levels + 1`` entries."""
    if levels < 0:
        raise ValueError("levels must be non-negative.")
    array = np.asarray(image, dtype=np.float32)
    pyramid = [array]
    for _ in range(levels):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def laplacian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    """Laplacian pyramid; the last entry is the Gaussian residual."""
    if levels < 0:
        raise ValueError("levels must be non-negative.")
    gaussian = gaussian_pyramid(image, levels)
    pyramid: list[np.ndarray] = []
    for index in range(levels):
        height, width = gaussian[index].shape[:2]
        expanded = cv2.pyrUp(gaussian[index + 1], dstsize=(width, height))
        pyramid.append(gaussian[index] - expanded)
    pyramid.append(gaussian[-1])
    return pyramid


def collapse_pyramid(pyramid: list[np.ndarray]) -> np.ndarray:
    """Reconstruct an image from a Laplacian pyramid."""
    if not pyramid:
        raise ValueError("pyramid must not be empty.")
    result = pyramid[-1]
    for index in range(len(pyramid) - 2, -1, -1):
        height, width = pyramid[index].shape[:2]
        result = cv2.pyrUp(result, dstsize=(width, height)) + pyramid[index]
    return result


def _resolve_scan(
    original: np.ndarray,
    scan_result: LumenScanResult | None,
) -> LumenScanResult:
    return scan_result or scan(original)


def _candidate_weight(
    candidate: np.ndarray,
    original: np.ndarray,
    local_sigma: np.ndarray,
    parameters: LumenParameters,
    config: LumenFuseConfig,
    edge_config: EdgeGuardConfig,
) -> np.ndarray:
    luma = luminance(candidate)
    blurred = cv2.GaussianBlur(
        luma, (0, 0), sigmaX=float(config.contrast_blur_sigma)
    )
    contrast = np.abs(
        cv2.Laplacian(blurred, cv2.CV_32F, ksize=int(config.laplacian_kernel))
    ).astype(np.float64)

    saturation = np.std(candidate.astype(np.float64), axis=2)
    exposure = np.exp(
        -((luma.astype(np.float64) - config.exposure_target) ** 2)
        / (2.0 * config.exposure_sigma**2)
    )

    gain = effective_gain_map(candidate, original, edge_config).astype(np.float64)
    noise = 1.0 / (
        1.0
        + (
            gain
            * local_sigma.astype(np.float64)
            / float(parameters.fusion_tau)
        )
        ** 2
    )

    return (
        np.power(contrast + config.epsilon, parameters.fusion_contrast)
        * np.power(saturation + config.epsilon, parameters.fusion_saturation)
        * np.power(exposure + config.epsilon, parameters.fusion_exposure)
        * np.power(noise, parameters.fusion_noise)
    )


def compute_weights(
    original: np.ndarray,
    classical: np.ndarray,
    deep_learning: np.ndarray,
    *,
    scan_result: LumenScanResult | None = None,
    config: LumenFuseConfig | None = None,
) -> FusionWeights:
    """Compute raw candidate weights, normalize, refine and renormalize."""
    config = config or LumenFuseConfig()
    _validate_config(config)

    candidates = (
        as_float_rgb(original, name="original"),
        as_float_rgb(classical, name="classical"),
        as_float_rgb(deep_learning, name="deep_learning"),
    )
    shape = candidates[0].shape
    if any(candidate.shape != shape for candidate in candidates[1:]):
        raise ValueError("All fusion candidates must have the same shape.")

    resolved = _resolve_scan(candidates[0], scan_result)
    if resolved.noise_map.shape != shape[:2]:
        raise ValueError(
            f"noise_map shape {resolved.noise_map.shape} does not match "
            f"the candidates {shape[:2]}."
        )
    _validate_parameters(resolved.parameters)

    edge_config = EdgeGuardConfig(
        effective_gain_kernel=config.effective_gain_kernel,
        effective_gain_epsilon=config.effective_gain_epsilon,
        gain_min=config.gain_min,
        gain_max=config.gain_max,
    )

    raw = np.stack(
        [
            _candidate_weight(
                candidate,
                candidates[0],
                resolved.noise_map,
                resolved.parameters,
                config,
                edge_config,
            )
            for candidate in candidates
        ],
        axis=2,
    )

    highlight = candidates[0].max(axis=2) > config.highlight_threshold
    raw[..., 0] *= np.where(highlight, config.highlight_multiplier, 1.0)

    initial = normalize_weights(raw, epsilon=config.epsilon)

    guide = luminance(candidates[0])
    refined = np.stack(
        [
            guided_filter(
                guide,
                initial[..., index].astype(np.float32),
                config.weight_guided_radius,
                config.weight_guided_eps,
            )
            for index in range(CANDIDATE_COUNT)
        ],
        axis=2,
    ).astype(np.float64)
    refined = normalize_weights(refined)

    return FusionWeights(
        initial=initial.astype(np.float64),
        refined=refined.astype(np.float32),
        parameters=resolved.parameters,
        sigma=float(resolved.features.noise_sigma),
    )


def fuse(
    original: np.ndarray,
    classical: np.ndarray,
    deep_learning: np.ndarray,
    *,
    scan_result: LumenScanResult | None = None,
    config: LumenFuseConfig | None = None,
) -> LumenFuseResult:
    """Fuse the three candidates into one float32 RGB image.

    Args:
        original: HxWx3 float32 RGB in [0, 1].
        classical: LumenRetinex output, same shape.
        deep_learning: Zero-DCE++ output, same shape.
        scan_result: LumenScan output for the original. Computed when omitted;
            its full-resolution noise map drives the noise penalty.
        config: fusion constants from §4.

    Returns:
        ``LumenFuseResult`` with the fused image, refined weights, pyramid level
        count, selected parameters and noise sigma.
    """
    config = config or LumenFuseConfig()
    _validate_config(config)

    candidates = (
        as_float_rgb(original, name="original"),
        as_float_rgb(classical, name="classical"),
        as_float_rgb(deep_learning, name="deep_learning"),
    )
    shape = candidates[0].shape
    if any(candidate.shape != shape for candidate in candidates[1:]):
        raise ValueError("All fusion candidates must have the same shape.")

    resolved = _resolve_scan(candidates[0], scan_result)
    weights = compute_weights(
        *candidates, scan_result=resolved, config=config
    )

    height, width = shape[:2]
    levels = pyramid_levels(width, height, config.max_levels)

    weight_pyramids = [
        gaussian_pyramid(weights.refined[..., index], levels)
        for index in range(CANDIDATE_COUNT)
    ]
    candidate_pyramids = [
        laplacian_pyramid(candidate, levels) for candidate in candidates
    ]

    fused_laplacian: list[np.ndarray] = []
    for level in range(levels + 1):
        blended = np.zeros_like(candidate_pyramids[0][level], dtype=np.float32)
        for index in range(CANDIDATE_COUNT):
            blended += (
                weight_pyramids[index][level][..., None]
                * candidate_pyramids[index][level]
            )
        fused_laplacian.append(blended)

    fused = collapse_pyramid(fused_laplacian)
    fused = np.ascontiguousarray(np.clip(fused, 0.0, 1.0).astype(np.float32))

    if fused.shape != shape:
        raise RuntimeError(
            f"LumenFuse changed the image shape: {shape} -> {fused.shape}."
        )

    return LumenFuseResult(
        image=fused,
        weights=weights.refined,
        levels=levels,
        parameters=weights.parameters,
        sigma=weights.sigma,
        initial_weights=weights.initial.astype(np.float32),
    )


__all__ = [
    "CANDIDATE_COUNT",
    "FusionWeights",
    "LumenFuseConfig",
    "LumenFuseResult",
    "collapse_pyramid",
    "compute_weights",
    "fuse",
    "gaussian_pyramid",
    "laplacian_pyramid",
    "normalize_weights",
    "pyramid_levels",
]
