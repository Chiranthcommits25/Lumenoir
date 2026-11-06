"""No-reference and artifact metrics.

These follow ``final_decision.md`` §6.3-§6.4. Metrics that compare against the
low-light input use ``MetricInput.original``; when a reference exists they use
the reference instead (edge correlation, hue shift). NIQE needs ``pyiqa``.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from skimage.color import rgb2lab

from .images import to_gray
from .noise import flat_mask, relative_noise
from .registry import register_metric
from .types import MetricDefinition, MetricDependencyError, MetricInput, MetricInputError

_CLIP_LEVEL = 0.98
_HIGHLIGHT_LOW = 0.85
_HIGHLIGHT_HIGH = 0.98
_BLACK_LEVEL = 0.02
_LOE_SIZE = 50
_STRONG_EDGE_PERCENTILE = 90.0
_HALO_EDGE_PERCENTILE = 95.0
_HALO_BLUR_SIGMA = 2.0
_COLOUR_CHROMA_THRESHOLD = 10.0
_NIQE_CACHE: dict[str, object] = {}


def _comparison(metric_input: MetricInput) -> np.ndarray:
    """Reference when available, otherwise the low-light input."""
    target = metric_input.reference if metric_input.reference is not None else metric_input.original
    if target is None:
        raise MetricInputError(
            "This metric needs either a reference image or the low-light input."
        )
    if target.shape != metric_input.image.shape:
        raise MetricInputError(
            f"Comparison shape {target.shape} differs from output {metric_input.image.shape}."
        )
    return target


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(dx, dy)


def _local_std(gray: np.ndarray, window: int = 7) -> np.ndarray:
    mean = cv2.blur(gray, (window, window))
    mean_square = cv2.blur(gray * gray, (window, window))
    return np.sqrt(np.maximum(mean_square - mean * mean, 0.0))


def compute_niqe(metric_input: MetricInput) -> float:
    """NIQE via IQA-PyTorch (relative comparisons only)."""
    device = os.getenv("LUMENOIR_EVAL_DEVICE", "cpu")
    model = _NIQE_CACHE.get(device)
    if model is None:
        try:
            import pyiqa
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise MetricDependencyError("pyiqa") from exc
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise MetricDependencyError("torch") from exc
        model = pyiqa.create_metric("niqe", device=device)
        _NIQE_CACHE[device] = model
    else:
        import torch

    tensor = torch.from_numpy(
        np.ascontiguousarray(metric_input.image.transpose(2, 0, 1))
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        value = model(tensor)
    return float(value.item())


def compute_noise_relative(metric_input: MetricInput) -> float:
    """sigma_flat(out) / mean(Y_out[flat]) on the flattest 50% of pixels."""
    return relative_noise(to_gray(metric_input.image))


def compute_noise_residual(metric_input: MetricInput) -> float:
    """Mean absolute residual against the reference in flat regions."""
    reference = metric_input.require_reference()
    if reference.shape != metric_input.image.shape:
        raise MetricInputError(
            f"Reference shape {reference.shape} differs from output {metric_input.image.shape}."
        )
    gray_reference = to_gray(reference)
    mask = flat_mask(gray_reference)
    if not mask.any():
        return 0.0
    difference = np.abs(metric_input.image - reference).mean(axis=2)[1:-1, 1:-1]
    if difference.shape != mask.shape:
        return 0.0
    return float(difference[mask].mean())


def compute_edge_preservation(metric_input: MetricInput) -> float | None:
    """Pearson correlation of gradient magnitudes on the strongest 10% edges."""
    target = _comparison(metric_input)
    reference_gradient = _gradient_magnitude(to_gray(target))
    output_gradient = _gradient_magnitude(to_gray(metric_input.image))
    if reference_gradient.size == 0:
        return None

    threshold = np.percentile(reference_gradient, _STRONG_EDGE_PERCENTILE)
    mask = reference_gradient >= threshold
    if int(mask.sum()) < 2:
        return None

    left = output_gradient[mask]
    right = reference_gradient[mask]
    if float(np.std(left)) < 1e-9 or float(np.std(right)) < 1e-9:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def compute_texture_retention(metric_input: MetricInput, percentile: float = 90.0) -> float | None:
    """Ratio of local standard deviation in the most textured patches.

    ``final_decision.md`` asks for "local-std retention in textured patches"
    without fixing the patch threshold, so it is a parameter here (default: the
    90th percentile of local standard deviation).
    """
    target = _comparison(metric_input)
    reference_std = _local_std(to_gray(target))
    output_std = _local_std(to_gray(metric_input.image))
    if reference_std.size == 0:
        return None

    threshold = np.percentile(reference_std, percentile)
    mask = reference_std >= threshold
    if not mask.any():
        return None

    denominator = float(reference_std[mask].mean())
    if denominator <= 1e-9:
        return None
    return float(output_std[mask].mean() / denominator)


def compute_hue_shift(metric_input: MetricInput) -> float | None:
    """Mean absolute wrapped hue difference in degrees on chroma > 10 pixels."""
    target = _comparison(metric_input)
    lab_target = rgb2lab(target)
    lab_output = rgb2lab(metric_input.image)

    chroma_target = np.hypot(lab_target[..., 1], lab_target[..., 2])
    chroma_output = np.hypot(lab_output[..., 1], lab_output[..., 2])
    mask = (chroma_target > _COLOUR_CHROMA_THRESHOLD) & (
        chroma_output > _COLOUR_CHROMA_THRESHOLD
    )
    if not mask.any():
        return None

    hue_target = np.arctan2(lab_target[..., 2], lab_target[..., 1])
    hue_output = np.arctan2(lab_output[..., 2], lab_output[..., 1])
    wrapped = np.angle(np.exp(1j * (hue_output - hue_target)))
    return float(np.degrees(np.abs(wrapped[mask])).mean())


def compute_new_clipping(metric_input: MetricInput) -> float:
    """Fraction of newly clipped pixels: out >= 0.98 minus input >= 0.98."""
    original = metric_input.require_original()
    if original.shape != metric_input.image.shape:
        raise MetricInputError(
            f"Original shape {original.shape} differs from output {metric_input.image.shape}."
        )
    input_max = original.max(axis=2)
    output_max = metric_input.image.max(axis=2)
    return float((output_max >= _CLIP_LEVEL).mean() - (input_max >= _CLIP_LEVEL).mean())


def compute_highlight_push(metric_input: MetricInput) -> float:
    """Share of input pixels in [0.85, 0.98] pushed to clipping."""
    original = metric_input.require_original()
    if original.shape != metric_input.image.shape:
        raise MetricInputError(
            f"Original shape {original.shape} differs from output {metric_input.image.shape}."
        )
    input_max = original.max(axis=2)
    output_max = metric_input.image.max(axis=2)
    near_highlight = (input_max >= _HIGHLIGHT_LOW) & (input_max < _HIGHLIGHT_HIGH)
    if not near_highlight.any():
        return 0.0
    return float((output_max[near_highlight] >= _CLIP_LEVEL).mean())


def compute_halo_rate(metric_input: MetricInput) -> float:
    """Gradient-reversal rate at the strongest 5% of input edges."""
    original = metric_input.require_original()
    if original.shape != metric_input.image.shape:
        raise MetricInputError(
            f"Original shape {original.shape} differs from output {metric_input.image.shape}."
        )

    input_gray = cv2.GaussianBlur(
        to_gray(original), (0, 0), sigmaX=_HALO_BLUR_SIGMA
    )
    output_gray = cv2.GaussianBlur(
        to_gray(metric_input.image), (0, 0), sigmaX=_HALO_BLUR_SIGMA
    )

    input_dx = cv2.Sobel(input_gray, cv2.CV_32F, 1, 0, ksize=3)
    input_dy = cv2.Sobel(input_gray, cv2.CV_32F, 0, 1, ksize=3)
    output_dx = cv2.Sobel(output_gray, cv2.CV_32F, 1, 0, ksize=3)
    output_dy = cv2.Sobel(output_gray, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = np.hypot(input_dx, input_dy)
    threshold = np.percentile(magnitude, _HALO_EDGE_PERCENTILE)
    mask = magnitude >= threshold
    if not mask.any():
        return 0.0

    flipped_x = np.sign(output_dx[mask]) != np.sign(input_dx[mask])
    flipped_y = np.sign(output_dy[mask]) != np.sign(input_dy[mask])
    return float(np.logical_or(flipped_x, flipped_y).mean())


def _loe_lightness(rgb: np.ndarray) -> np.ndarray:
    resized = cv2.resize(
        rgb, (_LOE_SIZE, _LOE_SIZE), interpolation=cv2.INTER_AREA
    )
    return resized.max(axis=2).astype(np.float64)


def compute_loe(metric_input: MetricInput) -> float:
    """Lightness Order Error (NPE-paper definition) against the input.

    Both images are downscaled to 50x50 and reduced to their max-RGB
    lightness. LOE is the fraction of valid pixel pairs whose relative
    lightness order changes.
    """
    original = metric_input.require_original()
    if original.shape != metric_input.image.shape:
        raise MetricInputError(
            f"Original shape {original.shape} differs from output {metric_input.image.shape}."
        )

    input_lightness = _loe_lightness(original)
    output_lightness = _loe_lightness(metric_input.image)

    input_diff = input_lightness[:, None] - input_lightness[None, :]
    output_diff = output_lightness[:, None] - output_lightness[None, :]
    valid = (input_diff != 0.0) & (output_diff != 0.0)
    if not valid.any():
        return 0.0

    mismatches = (np.sign(input_diff) != np.sign(output_diff)) & valid
    return float(mismatches.sum() / valid.sum())


def compute_black_crushing(metric_input: MetricInput) -> float:
    """New crushed-black fraction at max-channel <= 0.02."""
    original = metric_input.require_original()
    if original.shape != metric_input.image.shape:
        raise MetricInputError(
            f"Original shape {original.shape} differs from output {metric_input.image.shape}."
        )
    input_max = original.max(axis=2)
    output_max = metric_input.image.max(axis=2)
    return float(
        (output_max <= _BLACK_LEVEL).mean() - (input_max <= _BLACK_LEVEL).mean()
    )


register_metric(
    MetricDefinition(
        key="niqe",
        label="NIQE",
        function=compute_niqe,
        requires_reference=False,
        unit="score",
        higher_is_better=False,
        description="No-reference perceptual quality (pyiqa); relative comparisons only.",
    )
)
register_metric(
    MetricDefinition(
        key="noise_relative",
        label="Relative noise",
        function=compute_noise_relative,
        requires_reference=False,
        unit="ratio",
        higher_is_better=False,
        description="Immerkaer sigma on flat regions divided by their mean luminance.",
    )
)
register_metric(
    MetricDefinition(
        key="noise_residual",
        label="Noise residual (paired)",
        function=compute_noise_residual,
        requires_reference=True,
        unit="intensity",
        higher_is_better=False,
        description="Mean absolute residual against the reference in flat regions.",
    )
)
register_metric(
    MetricDefinition(
        key="edge_preservation",
        label="Edge preservation",
        function=compute_edge_preservation,
        requires_reference=False,
        unit="correlation",
        higher_is_better=True,
        description="Gradient-magnitude correlation on the strongest 10% of edges.",
    )
)
register_metric(
    MetricDefinition(
        key="texture_retention",
        label="Texture retention",
        function=compute_texture_retention,
        requires_reference=False,
        unit="ratio",
        higher_is_better=True,
        description="Local-standard-deviation ratio in the most textured patches.",
    )
)
register_metric(
    MetricDefinition(
        key="hue_shift_deg",
        label="Hue shift",
        function=compute_hue_shift,
        requires_reference=False,
        unit="degrees",
        higher_is_better=False,
        description="Mean absolute hue difference on pixels with chroma > 10.",
    )
)
register_metric(
    MetricDefinition(
        key="new_clipping",
        label="New clipped pixels",
        function=compute_new_clipping,
        requires_reference=False,
        unit="fraction",
        higher_is_better=False,
        description="Output clipping fraction minus input clipping fraction at 0.98.",
    )
)
register_metric(
    MetricDefinition(
        key="highlight_push",
        label="Highlight push",
        function=compute_highlight_push,
        requires_reference=False,
        unit="fraction",
        higher_is_better=False,
        description="Share of input pixels in [0.85, 0.98] pushed to clipping.",
    )
)
register_metric(
    MetricDefinition(
        key="halo_rate",
        label="Halo rate",
        function=compute_halo_rate,
        requires_reference=False,
        unit="rate",
        higher_is_better=False,
        description="Gradient-reversal rate on the strongest 5% of input edges.",
    )
)
register_metric(
    MetricDefinition(
        key="loe",
        label="LOE",
        function=compute_loe,
        requires_reference=False,
        unit="rate",
        higher_is_better=False,
        description=(
            "Lightness Order Error on 50x50 max-RGB lightness maps, rate of "
            "pixel pairs whose relative order changes."
        ),
    )
)
register_metric(
    MetricDefinition(
        key="black_crushing",
        label="Black-level crushing",
        function=compute_black_crushing,
        requires_reference=False,
        unit="fraction",
        higher_is_better=False,
        description=(
            "Output crushed-black fraction minus input crushed-black fraction "
            "at max-channel 0.02."
        ),
    )
)
