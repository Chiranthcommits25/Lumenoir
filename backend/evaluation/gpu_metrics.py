"""GPU implementations of the reference metrics used by the harness.

Only PSNR and SSIM are accelerated here; LPIPS and NIQE already use their own
torch backends, and CIEDE2000 stays on the CPU. Selection is explicit through
``LUMENOIR_EVAL_DEVICE`` (``cuda``, ``cuda:N``, ``mps`` or ``auto``); the default
is the skimage/CPU path so metric numbers do not change silently.
"""

from __future__ import annotations

import os
import threading

import numpy as np

from .types import MetricInput, MetricInputError

_LOCK = threading.Lock()
_STATE: dict[str, object] = {}
_SELECTED_ENV = "LUMENOIR_EVAL_DEVICE"


def _selected_ops():
    requested = (os.getenv(_SELECTED_ENV, "") or "").strip().lower()
    if requested in {"", "cpu", "default"}:
        return None
    with _LOCK:
        state = _STATE.get("ops")
        if state is not None and _STATE.get("requested") == requested:
            return state
        try:
            from ..runtime.config import resolve_runtime
            from ..runtime.torch_ops import TorchImageOps

            runtime = resolve_runtime(requested)
            if not runtime.is_gpu:
                _STATE["requested"] = requested
                _STATE["ops"] = None
                return None
            ops = TorchImageOps(runtime)
        except Exception:
            _STATE["requested"] = requested
            _STATE["ops"] = None
            return None
        _STATE["requested"] = requested
        _STATE["ops"] = ops
        return ops


def gpu_metrics_available() -> bool:
    return _selected_ops() is not None


def _tensors(metric_input: MetricInput):
    ops = _selected_ops()
    if ops is None:
        return None
    cache = metric_input.cache
    image = cache.get("gpu_image")
    reference = cache.get("gpu_reference")
    if image is None:
        image = ops.ensure(metric_input.image)
    if reference is None:
        reference = ops.ensure(metric_input.require_reference())
    cache["gpu_ops"] = ops
    cache["gpu_image"] = image
    cache["gpu_reference"] = reference
    return ops, image, reference


def compute_psnr(metric_input: MetricInput) -> float:
    prepared = _tensors(metric_input)
    if prepared is None:
        raise MetricInputError("GPU metrics are not selected.")
    ops, image, reference = prepared
    if image.shape != reference.shape:
        raise MetricInputError(
            f"Reference shape {tuple(reference.shape)} differs from output "
            f"{tuple(image.shape)}."
        )
    torch = ops.torch
    with torch.inference_mode():
        mse = torch.mean((image - reference) ** 2)
        if float(mse) == 0.0:
            return float("inf")
        return float((-10.0 * torch.log10(mse)).item())


def _uniform(channel, radius: int, ops):
    size = 2 * radius + 1
    kernel = np.full((size, size), 1.0 / (size * size), dtype=np.float32)
    return ops.filter2d(channel, kernel, reflect101=True)


def _ssim_channel(reference, image, radius: int, ops):
    window = 2 * radius + 1
    samples = float(window * window)
    cov_norm = samples / (samples - 1.0)
    c1 = (0.01**2) * 1.0
    c2 = (0.03**2) * 1.0
    ux = _uniform(reference, radius, ops)
    uy = _uniform(image, radius, ops)
    uxx = _uniform(reference * reference, radius, ops)
    uyy = _uniform(image * image, radius, ops)
    uxy = _uniform(reference * image, radius, ops)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    numerator = (2.0 * ux * uy + c1) * (2.0 * vxy + c2)
    denominator = (ux * ux + uy * uy + c1) * (vx + vy + c2)
    result = numerator / denominator
    return result[radius:-radius, radius:-radius]


def compute_ssim(metric_input: MetricInput) -> float:
    prepared = _tensors(metric_input)
    if prepared is None:
        raise MetricInputError("GPU metrics are not selected.")
    ops, image, reference = prepared
    if image.shape != reference.shape:
        raise MetricInputError(
            f"Reference shape {tuple(reference.shape)} differs from output "
            f"{tuple(image.shape)}."
        )
    if min(image.shape[0], image.shape[1]) < 7:
        raise MetricInputError("SSIM needs at least 7 pixels per side.")
    torch = ops.torch
    radius = 3
    with torch.inference_mode():
        values = [
            _ssim_channel(reference[..., index], image[..., index], radius, ops)
            for index in range(3)
        ]
        return float(torch.stack(values).mean().item())


__all__ = ["compute_psnr", "compute_ssim", "gpu_metrics_available"]
