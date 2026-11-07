"""PyTorch backend for LumenFuse.

The fusion algorithm lives in :mod:`backend.fusion.lumen_fuse` and is shared
between backends. This module supplies the numerical primitives through
:class:`backend.runtime.torch_ops.TorchImageOps`; it keeps the original
constructor and helper names so existing configuration and tests keep working.

The module is imported lazily. Importing :mod:`backend.fusion.lumen_fuse` never
requires torch, and CPU-only deployments keep using the OpenCV backend.
"""

from __future__ import annotations

import torch

from ..runtime.kernels import (
    PYR_DOWN_KERNEL,
    PYR_UP_KERNEL,
    gaussian_kernel_1d,
    laplacian_kernel_2d,
)
from ..runtime.torch_ops import TorchImageOps

CANDIDATE_COUNT = 3

# Backwards-compatible names for callers that imported the private helpers.
_gaussian_kernel_1d = gaussian_kernel_1d
_laplacian_kernel_2d = laplacian_kernel_2d
_PYR_DOWN_KERNEL = PYR_DOWN_KERNEL
_PYR_UP_KERNEL = PYR_UP_KERNEL


class TorchFusionOps(TorchImageOps):
    """GPU/accelerator primitives for the shared LumenFuse core."""

    def __init__(self, *, device: str | None = None, precision: str = "float32"):
        requested = (device or "auto").strip().lower()
        if requested in {"", "auto", "default"}:
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested == "gpu":
            requested = "cuda"
        super().__init__(device=requested, precision=precision)
        torch.zeros(1, device=self.device, dtype=self.dtype)

    @classmethod
    def cuda_available(cls) -> bool:
        try:
            return bool(torch.cuda.is_available())
        except Exception:  # pragma: no cover - depends on the environment
            return False

    # -- pyramid protocol --------------------------------------------------

    def gaussian_pyramid(self, image, levels):
        from .lumen_fuse import _gaussian_pyramid_core

        return _gaussian_pyramid_core(image, levels, self)

    def laplacian_pyramid(self, image, levels):
        from .lumen_fuse import _laplacian_pyramid_core

        return _laplacian_pyramid_core(image, levels, self)

    def collapse_pyramid(self, pyramid):
        from .lumen_fuse import _collapse_pyramid_core

        return _collapse_pyramid_core(pyramid, self)


__all__ = ["TorchFusionOps"]
