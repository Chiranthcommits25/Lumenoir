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

import importlib.util
import logging
import math
from contextlib import nullcontext
from dataclasses import dataclass

import cv2
import numpy as np

from ..classical_cv.arrays import as_float_rgb
from ..classical_cv.filters import box_filter, guided_filter
from ..classical_cv.scan import LumenParameters, LumenScanResult, scan

CANDIDATE_COUNT = 3
EPSILON = 1e-6
_DEFAULT_GUIDED_FILTER = guided_filter
logger = logging.getLogger(__name__)


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
    array = np.asarray(weights)
    if array.ndim != 3 or array.shape[2] != CANDIDATE_COUNT:
        raise ValueError(f"weights must be HxWx{CANDIDATE_COUNT}, got {array.shape}.")
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative.")
    if not np.isfinite(array).all():
        raise ValueError("weights contain NaN or infinite values.")

    # Keep the weight math in float32: the inputs are float32 and the final
    # pyramids are float32. The old float64 temporaries dominated memory traffic.
    array = array.astype(np.float32, copy=False)
    numerator = np.maximum(array, 0.0) + np.float32(epsilon)
    denominator = numerator.sum(axis=2, keepdims=True)
    result = np.empty_like(numerator)
    np.divide(numerator, denominator, out=result, where=denominator > 0.0)
    zero = denominator[..., 0] <= 0.0
    if zero.any():
        result[zero] = np.float32(1.0 / CANDIDATE_COUNT)
    return result


def pyramid_levels(width: int, height: int, max_levels: int = 5) -> int:
    """Top pyramid index ``L = min(max_levels, floor(log2(min(H,W))) - 3)``."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")
    if max_levels < 0:
        raise ValueError("max_levels must be non-negative.")
    computed = math.floor(math.log2(min(width, height))) - 3
    return max(0, min(int(max_levels), computed))


def _gaussian_pyramid_core(image, levels: int, ops):
    """Backend-agnostic Gaussian pyramid construction."""
    if levels < 0:
        raise ValueError("levels must be non-negative.")
    pyramid = [image]
    for _ in range(levels):
        pyramid.append(ops.pyr_down(pyramid[-1]))
    return pyramid


def _laplacian_pyramid_core(image, levels: int, ops):
    """Backend-agnostic Laplacian pyramid construction."""
    if levels < 0:
        raise ValueError("levels must be non-negative.")
    gaussian = _gaussian_pyramid_core(image, levels, ops)
    pyramid = []
    for index in range(levels):
        height, width = gaussian[index].shape[:2]
        expanded = ops.pyr_up(gaussian[index + 1], (width, height))
        pyramid.append(gaussian[index] - expanded)
    pyramid.append(gaussian[-1])
    return pyramid


def _collapse_pyramid_core(pyramid, ops):
    """Backend-agnostic Laplacian pyramid reconstruction."""
    if not pyramid:
        raise ValueError("pyramid must not be empty.")
    result = pyramid[-1]
    for index in range(len(pyramid) - 2, -1, -1):
        height, width = pyramid[index].shape[:2]
        result = ops.pyr_up(result, (width, height)) + pyramid[index]
    return result


def gaussian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    """Gaussian pyramid with ``levels + 1`` entries."""
    array = np.asarray(image, dtype=np.float32)
    return _gaussian_pyramid_core(array, levels, _NUMPY_OPS)


def laplacian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    """Laplacian pyramid; the last entry is the Gaussian residual."""
    array = np.asarray(image, dtype=np.float32)
    return _laplacian_pyramid_core(array, levels, _NUMPY_OPS)


def collapse_pyramid(pyramid: list[np.ndarray]) -> np.ndarray:
    """Reconstruct an image from a Laplacian pyramid."""
    return _collapse_pyramid_core(pyramid, _NUMPY_OPS)


def _resolve_scan(
    original: np.ndarray,
    scan_result: LumenScanResult | None,
) -> LumenScanResult:
    return scan_result or scan(original)


def _luma(rgb: np.ndarray) -> np.ndarray:
    """Rec. 601 luma for already-validated float32 RGB (no revalidation)."""
    return np.ascontiguousarray(
        (
            0.299 * rgb[..., 0]
            + 0.587 * rgb[..., 1]
            + 0.114 * rgb[..., 2]
        ),
        dtype=np.float32,
    )


def _prepare_candidates(
    original: np.ndarray,
    classical: np.ndarray,
    deep_learning: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates = (
        as_float_rgb(original, name="original"),
        as_float_rgb(classical, name="classical"),
        as_float_rgb(deep_learning, name="deep_learning"),
    )
    shape = candidates[0].shape
    if any(candidate.shape != shape for candidate in candidates[1:]):
        raise ValueError("All fusion candidates must have the same shape.")
    return candidates


def _refine_weight_channels(
    guide: np.ndarray,
    weights: np.ndarray,
    radius: int,
    eps: float,
) -> np.ndarray:
    """Guided-filter all three weight channels with shared guide statistics."""
    if guided_filter is not _DEFAULT_GUIDED_FILTER:
        # Preserve the existing seam for callers/tests that replace the
        # single-channel primitive.
        return np.stack(
            [
                guided_filter(guide, weights[..., index], radius, eps)
                for index in range(CANDIDATE_COUNT)
            ],
            axis=2,
        )

    mean_guide = box_filter(guide, radius)
    mean_source = box_filter(weights, radius)
    variance = box_filter(guide * guide, radius) - mean_guide * mean_guide
    covariance = (
        box_filter(guide[..., None] * weights, radius)
        - mean_guide[..., None] * mean_source
    )
    a = covariance / (variance[..., None] + np.float32(eps))
    b = mean_source - a * mean_guide[..., None]
    return np.ascontiguousarray(
        box_filter(a, radius) * guide[..., None] + box_filter(b, radius)
    )


class _NumpyOps:
    """CPU backend: the existing NumPy/OpenCV primitives, unchanged."""

    name = "numpy"

    def context(self):
        return nullcontext()

    def scalar(self, value: float):
        return np.float32(value)

    def luma(self, rgb):
        return _luma(rgb)

    def gaussian_blur(self, image, sigma):
        return cv2.GaussianBlur(image, (0, 0), sigmaX=float(sigma))

    def laplacian(self, image, ksize):
        return cv2.Laplacian(image, cv2.CV_32F, ksize=int(ksize))

    def box_blur(self, image, size):
        return cv2.blur(image, (int(size), int(size)))

    def box_filter(self, image, radius):
        return box_filter(image, int(radius))

    def clip(self, image, low, high):
        return np.clip(image, low, high)

    def power(self, image, exponent):
        return np.power(image, exponent)

    def exp(self, image):
        return np.exp(image)

    def sqrt(self, image):
        return np.sqrt(image)

    def abs(self, image):
        return np.abs(image)

    def maximum(self, image, value):
        return np.maximum(image, value)

    def where(self, condition, first, second):
        return np.where(condition, first, second)

    def sum(self, image, axis, keepdims=False):
        return np.sum(image, axis=axis, keepdims=keepdims)

    def max(self, image, axis):
        return np.max(image, axis=axis)

    def stack(self, values, axis):
        return np.stack(values, axis=axis)

    def iadd(self, left, right):
        left += right
        return left

    def normalize(self, weights, epsilon):
        return normalize_weights(weights, epsilon=float(epsilon))

    def refine(self, guide, weights, radius, eps):
        return _refine_weight_channels(
            guide, weights, int(radius), float(eps)
        )

    def gaussian_pyramid(self, image, levels):
        return _gaussian_pyramid_core(image, levels, self)

    def laplacian_pyramid(self, image, levels):
        return _laplacian_pyramid_core(image, levels, self)

    def collapse_pyramid(self, pyramid):
        return _collapse_pyramid_core(pyramid, self)

    def pyr_down(self, image):
        return cv2.pyrDown(image)

    def pyr_up(self, image, dst_size):
        return cv2.pyrUp(image, dstsize=dst_size)

    def prepare(self, candidates, noise_map):
        return candidates, noise_map

    def finalize_image(self, image):
        return np.ascontiguousarray(image.astype(np.float32))

    def to_numpy(self, value):
        return value


_NUMPY_OPS = _NumpyOps()


def _candidate_weight_core(
    candidate, luma, original_luma_blur, local_sigma, parameters, config, ops
):
    blurred = ops.gaussian_blur(luma, float(config.contrast_blur_sigma))
    contrast = ops.abs(ops.laplacian(blurred, int(config.laplacian_kernel)))

    red = candidate[..., 0]
    green = candidate[..., 1]
    blue = candidate[..., 2]
    channel_mean = (red + green + blue) / ops.scalar(3.0)
    saturation = ops.sqrt(
        (
            (red - channel_mean) ** 2
            + (green - channel_mean) ** 2
            + (blue - channel_mean) ** 2
        )
        / ops.scalar(3.0)
    )
    exposure = ops.exp(
        -((luma - ops.scalar(config.exposure_target)) ** 2)
        / ops.scalar(2.0 * config.exposure_sigma**2)
    )

    kernel = int(config.effective_gain_kernel)
    gain = ops.box_blur(luma, kernel) / (
        original_luma_blur + ops.scalar(config.effective_gain_epsilon)
    )
    gain = ops.clip(gain, config.gain_min, config.gain_max)
    noise = ops.scalar(1.0) / (
        ops.scalar(1.0)
        + (gain * local_sigma / ops.scalar(parameters.fusion_tau)) ** 2
    )

    return (
        ops.power(contrast + ops.scalar(config.epsilon), parameters.fusion_contrast)
        * ops.power(saturation + ops.scalar(config.epsilon), parameters.fusion_saturation)
        * ops.power(exposure + ops.scalar(config.epsilon), parameters.fusion_exposure)
        * ops.power(noise, parameters.fusion_noise)
    )


def _compute_weights_core(candidates, noise_map, parameters, config, ops):
    """Backend-agnostic weight generation shared by CPU and torch."""
    shape = candidates[0].shape
    if noise_map.shape != shape[:2]:
        raise ValueError(
            f"noise_map shape {noise_map.shape} does not match "
            f"the candidates {shape[:2]}."
        )
    _validate_parameters(parameters)

    lumas = [ops.luma(candidate) for candidate in candidates]
    kernel = int(config.effective_gain_kernel)
    original_luma_blur = ops.box_blur(lumas[0], kernel)

    raw_channels = [
        _candidate_weight_core(
            candidate,
            luma,
            original_luma_blur,
            noise_map,
            parameters,
            config,
            ops,
        )
        for candidate, luma in zip(candidates, lumas, strict=True)
    ]

    highlight = ops.max(candidates[0], axis=2) > ops.scalar(config.highlight_threshold)
    raw_channels[0] = raw_channels[0] * ops.where(
        highlight,
        ops.scalar(config.highlight_multiplier),
        ops.scalar(1.0),
    )

    initial = ops.normalize(ops.stack(raw_channels, axis=2), config.epsilon)
    refined = ops.refine(
        lumas[0],
        initial,
        int(config.weight_guided_radius),
        float(config.weight_guided_eps),
    )
    refined = ops.normalize(refined, 0.0)
    return initial, refined


def _fuse_core(candidates, refined, levels: int, ops):
    """Backend-agnostic multiscale fusion shared by CPU and torch."""
    weight_pyramids = [
        ops.gaussian_pyramid(refined[..., index], levels)
        for index in range(CANDIDATE_COUNT)
    ]
    candidate_pyramids = [
        ops.laplacian_pyramid(candidate, levels) for candidate in candidates
    ]

    fused_laplacian = []
    for level in range(levels + 1):
        blended = (
            weight_pyramids[0][level][..., None] * candidate_pyramids[0][level]
        )
        for index in range(1, CANDIDATE_COUNT):
            blended = ops.iadd(
                blended,
                weight_pyramids[index][level][..., None]
                * candidate_pyramids[index][level],
            )
        fused_laplacian.append(blended)

    return ops.clip(ops.collapse_pyramid(fused_laplacian), 0.0, 1.0)


def _select_ops(backend: str, device: str | None, precision: str):
    """Choose the CPU or torch backend, falling back safely for ``auto``."""
    choice = (backend or "cpu").strip().lower()
    if choice == "cpu":
        return _NUMPY_OPS
    if choice not in {"auto", "torch"}:
        raise ValueError(
            f"Unsupported fusion backend {backend!r}; use 'cpu', 'torch' or 'auto'."
        )

    try:
        if choice == "auto" and importlib.util.find_spec("torch") is None:
            return _NUMPY_OPS
        from .torch_backend import TorchFusionOps

        if choice == "auto":
            requested = (device or "").strip().lower()
            if requested == "cpu":
                return _NUMPY_OPS
            if not requested and not TorchFusionOps.cuda_available():
                return _NUMPY_OPS
        return TorchFusionOps(device=device, precision=precision)
    except Exception as exc:
        if choice == "auto":
            logger.warning(
                "LumenFuse GPU backend unavailable; using the CPU backend: %s", exc
            )
            return _NUMPY_OPS
        raise


def _weights_from_ops(
    candidates, noise_map, parameters, config, ops, sigma: float
) -> FusionWeights:
    initial, refined = _compute_weights_core(
        candidates, noise_map, parameters, config, ops
    )
    return FusionWeights(
        initial=ops.to_numpy(initial),
        refined=ops.to_numpy(refined),
        parameters=parameters,
        sigma=sigma,
    )


def _compute_weights_from_candidates(
    candidates: tuple[np.ndarray, np.ndarray, np.ndarray],
    resolved: LumenScanResult,
    config: LumenFuseConfig,
) -> FusionWeights:
    return _weights_from_ops(
        candidates,
        resolved.noise_map,
        resolved.parameters,
        config,
        _NUMPY_OPS,
        float(resolved.features.noise_sigma),
    )


def compute_weights(
    original: np.ndarray,
    classical: np.ndarray,
    deep_learning: np.ndarray,
    *,
    scan_result: LumenScanResult | None = None,
    config: LumenFuseConfig | None = None,
    backend: str = "cpu",
    device: str | None = None,
    precision: str = "float32",
) -> FusionWeights:
    """Compute raw candidate weights, normalize, refine and renormalize.

    ``backend`` is ``"cpu"`` (default, existing NumPy/OpenCV path), ``"torch"``
    (explicit torch backend) or ``"auto"`` (CUDA when available, CPU otherwise).
    ``device`` and ``precision`` only apply to the torch backend.
    """
    config = config or LumenFuseConfig()
    _validate_config(config)
    candidates = _prepare_candidates(original, classical, deep_learning)
    resolved = _resolve_scan(candidates[0], scan_result)

    ops = _select_ops(backend, device, precision)
    backend_candidates, backend_noise = ops.prepare(candidates, resolved.noise_map)
    with ops.context():
        return _weights_from_ops(
            backend_candidates,
            backend_noise,
            resolved.parameters,
            config,
            ops,
            float(resolved.features.noise_sigma),
        )


def fuse(
    original: np.ndarray,
    classical: np.ndarray,
    deep_learning: np.ndarray,
    *,
    scan_result: LumenScanResult | None = None,
    config: LumenFuseConfig | None = None,
    backend: str = "cpu",
    device: str | None = None,
    precision: str = "float32",
) -> LumenFuseResult:
    """Fuse the three candidates into one float32 RGB image.

    Args:
        original: HxWx3 float32 RGB in [0, 1].
        classical: LumenRetinex output, same shape.
        deep_learning: Zero-DCE++ output, same shape.
        scan_result: LumenScan output for the original. Computed when omitted;
            its full-resolution noise map drives the noise penalty.
        config: fusion constants from §4.
        backend: ``"cpu"`` (default), ``"torch"`` or ``"auto"``. ``auto`` uses
            the CUDA torch backend when available and otherwise the CPU path.
        device: torch device override (``"cuda"``, ``"cuda:0"``, ``"cpu"``,
            ``"mps"``). Ignored by the CPU backend.
        precision: ``"float32"`` (default) or ``"float16"`` for the torch
            backend.

    Returns:
        ``LumenFuseResult`` with the fused image, refined weights, pyramid level
        count, selected parameters and noise sigma.
    """
    config = config or LumenFuseConfig()
    _validate_config(config)

    candidates = _prepare_candidates(original, classical, deep_learning)
    resolved = _resolve_scan(candidates[0], scan_result)
    shape = candidates[0].shape

    height, width = shape[:2]
    levels = pyramid_levels(width, height, config.max_levels)

    ops = _select_ops(backend, device, precision)
    backend_candidates, backend_noise = ops.prepare(candidates, resolved.noise_map)
    sigma = float(resolved.features.noise_sigma)

    with ops.context():
        initial, refined = _compute_weights_core(
            backend_candidates,
            backend_noise,
            resolved.parameters,
            config,
            ops,
        )
        fused = _fuse_core(backend_candidates, refined, levels, ops)
        fused = ops.finalize_image(fused)
        initial_np = ops.to_numpy(initial)
        refined_np = ops.to_numpy(refined)
        fused_np = ops.to_numpy(fused)

    if fused_np.shape != shape:
        raise RuntimeError(
            f"LumenFuse changed the image shape: {shape} -> {fused_np.shape}."
        )

    return LumenFuseResult(
        image=fused_np,
        weights=refined_np,
        levels=levels,
        parameters=resolved.parameters,
        sigma=sigma,
        initial_weights=initial_np,
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
