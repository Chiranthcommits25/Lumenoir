"""Reference-based metrics: PSNR, SSIM, LPIPS and CIEDE2000 colour difference.

Definitions follow ``final_decision.md`` §6.2 and §6.4. LPIPS and the aligned
PSNR/SSIM variants are optional additions; every failure is explicit rather
than a fabricated number.
"""

from __future__ import annotations

import os

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .images import align_to_reference_mean
from .registry import register_metric
from .types import MetricDefinition, MetricDependencyError, MetricInput, MetricInputError

_LPIPS_CACHE: dict[str, object] = {}
_DEFAULT_DEVICE = os.getenv("LUMENOIR_EVAL_DEVICE", "cpu")
_MIN_SSIM_SIDE = 7


def _check_ssim_size(array: np.ndarray) -> None:
    if min(array.shape[0], array.shape[1]) < _MIN_SSIM_SIDE:
        raise MetricInputError(
            f"SSIM needs at least {_MIN_SSIM_SIDE} pixels per side, got {array.shape[:2]}."
        )


def _check_pair_shapes(reference: np.ndarray, output: np.ndarray) -> None:
    if reference.shape != output.shape:
        raise MetricInputError(
            f"Reference shape {reference.shape} differs from output {output.shape}."
        )


def compute_psnr(metric_input: MetricInput) -> float:
    """PSNR in dB against the reference, data_range = 1."""
    reference = metric_input.require_reference()
    _check_pair_shapes(reference, metric_input.image)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Identical images give an infinite PSNR; that is the correct value.
        return float(
            peak_signal_noise_ratio(reference, metric_input.image, data_range=1.0)
        )


def compute_ssim(metric_input: MetricInput) -> float:
    """SSIM against the reference, computed per channel and averaged."""
    reference = metric_input.require_reference()
    _check_ssim_size(reference)
    _check_ssim_size(metric_input.image)
    _check_pair_shapes(reference, metric_input.image)
    return float(
        structural_similarity(
            reference, metric_input.image, channel_axis=2, data_range=1.0
        )
    )


def compute_psnr_aligned(metric_input: MetricInput) -> float:
    """PSNR after matching the output mean to the reference mean."""
    reference = metric_input.require_reference()
    aligned = align_to_reference_mean(metric_input.image, reference)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(peak_signal_noise_ratio(reference, aligned, data_range=1.0))


def compute_ssim_aligned(metric_input: MetricInput) -> float:
    """SSIM after matching the output mean to the reference mean."""
    reference = metric_input.require_reference()
    aligned = align_to_reference_mean(metric_input.image, reference)
    _check_ssim_size(reference)
    return float(
        structural_similarity(reference, aligned, channel_axis=2, data_range=1.0)
    )


def _torch_module():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise MetricDependencyError("torch") from exc
    return torch


def _lpips_model(device: str):
    model = _LPIPS_CACHE.get(device)
    if model is not None:
        return model

    try:
        import lpips
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise MetricDependencyError("lpips") from exc

    torch = _torch_module()
    with torch.no_grad():
        created = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    _LPIPS_CACHE[device] = created
    return created


def compute_lpips(metric_input: MetricInput) -> float:
    """LPIPS (AlexNet) against the reference; inputs scaled to [-1, 1]."""
    reference = metric_input.require_reference()
    if reference.shape != metric_input.image.shape:
        raise MetricInputError(
            f"Reference shape {reference.shape} differs from output {metric_input.image.shape}."
        )

    device = os.getenv("LUMENOIR_EVAL_DEVICE", _DEFAULT_DEVICE)
    torch = _torch_module()
    model = _lpips_model(device)

    def to_tensor(array: np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
        return (tensor.unsqueeze(0).to(device) * 2.0) - 1.0

    with torch.no_grad():
        value = model(to_tensor(reference), to_tensor(metric_input.image))
    return float(value.mean().item())


def compute_delta_e00(metric_input: MetricInput) -> float:
    """Mean CIEDE2000 colour difference against the reference."""
    reference = metric_input.require_reference()
    if reference.shape != metric_input.image.shape:
        raise MetricInputError(
            f"Reference shape {reference.shape} differs from output {metric_input.image.shape}."
        )
    lab_reference = rgb2lab(reference)
    lab_output = rgb2lab(metric_input.image)
    return float(np.mean(deltaE_ciede2000(lab_reference, lab_output)))


register_metric(
    MetricDefinition(
        key="psnr",
        label="PSNR",
        function=compute_psnr,
        requires_reference=True,
        unit="dB",
        higher_is_better=True,
        description="Full-reference fidelity (skimage, data_range=1.0).",
    )
)
register_metric(
    MetricDefinition(
        key="ssim",
        label="SSIM",
        function=compute_ssim,
        requires_reference=True,
        unit="ratio",
        higher_is_better=True,
        description="Structural similarity against the reference (channel-averaged).",
    )
)
register_metric(
    MetricDefinition(
        key="psnr_aligned",
        label="PSNR (mean-aligned)",
        function=compute_psnr_aligned,
        requires_reference=True,
        unit="dB",
        higher_is_better=True,
        description="PSNR after matching the output mean to the reference mean.",
    )
)
register_metric(
    MetricDefinition(
        key="ssim_aligned",
        label="SSIM (mean-aligned)",
        function=compute_ssim_aligned,
        requires_reference=True,
        unit="ratio",
        higher_is_better=True,
        description="SSIM after matching the output mean to the reference mean.",
    )
)
register_metric(
    MetricDefinition(
        key="lpips",
        label="LPIPS",
        function=compute_lpips,
        requires_reference=True,
        unit="distance",
        higher_is_better=False,
        description="Learned perceptual distance (lpips, AlexNet).",
    )
)
register_metric(
    MetricDefinition(
        key="delta_e00",
        label="Colour distortion (CIEDE2000)",
        function=compute_delta_e00,
        requires_reference=True,
        unit="dE",
        higher_is_better=False,
        description="Mean CIEDE2000 colour difference against the reference.",
    )
)
