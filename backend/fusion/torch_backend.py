"""PyTorch backend for LumenFuse.

This module implements only the numerical primitives LumenFuse needs; the
fusion algorithm itself lives in :mod:`backend.fusion.lumen_fuse` and is shared
between backends. The implementation mirrors OpenCV's kernels and border modes
(Gaussian blur, Laplacian, box filters, ``pyrDown``/``pyrUp``) so the CUDA path
produces practically equivalent results to the existing CPU path.

The module is imported lazily. Importing :mod:`backend.fusion.lumen_fuse` never
requires torch, and CPU-only deployments keep using the OpenCV backend.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
import torch
import torch.nn.functional as F

CANDIDATE_COUNT = 3

_K5 = (np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0).astype(
    np.float32
)
_PYR_DOWN_KERNEL = np.outer(_K5, _K5).astype(np.float32)
_PYR_UP_KERNEL = (4.0 * _PYR_DOWN_KERNEL).astype(np.float32)


@lru_cache(maxsize=8)
def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """Derive OpenCV's Gaussian taps for this build and sigma.

    Deriving the taps instead of hard-coding a kernel keeps the torch backend
    aligned with whichever OpenCV version supplies the CPU fallback.
    """
    sigma = float(sigma)
    approximate = max(3, round(sigma * 8.0 + 1.0) | 1)
    size = approximate + 10
    if size % 2 == 0:
        size += 1
    impulse = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    impulse[center, center] = 1.0
    blurred = cv2.GaussianBlur(impulse, (0, 0), sigmaX=sigma)
    row = blurred[center].astype(np.float64)
    peak = float(np.max(np.abs(row)))
    if peak <= 0.0:
        raise RuntimeError("OpenCV returned an empty Gaussian kernel.")
    support = np.nonzero(np.abs(row) > peak * 1e-12)[0]
    taps = row[support.min() : support.max() + 1]
    taps = taps / taps.sum()
    return taps.astype(np.float32)


@lru_cache(maxsize=8)
def _laplacian_kernel_2d(ksize: int) -> np.ndarray:
    """Derive OpenCV's ``ksize`` Laplacian kernel from an impulse response."""
    ksize = int(ksize)
    radius = max(1, ksize // 2)
    size = 2 * radius + 3
    impulse = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    impulse[center, center] = 1.0
    response = cv2.Laplacian(impulse, cv2.CV_32F, ksize=ksize)
    kernel = response[
        center - radius : center + radius + 1,
        center - radius : center + radius + 1,
    ]
    return np.ascontiguousarray(kernel, dtype=np.float32)


class TorchFusionOps:
    """GPU/accelerator primitives for the shared LumenFuse core."""

    name = "torch"

    def __init__(self, *, device: str | None = None, precision: str = "float32"):
        self._torch = torch
        choice = (precision or "float32").strip().lower()
        if choice in {"float32", "fp32"}:
            self.dtype = torch.float32
        elif choice in {"float16", "fp16", "half"}:
            self.dtype = torch.float16
        else:
            raise ValueError(
                f"Unsupported fusion precision {precision!r}; "
                "use 'float32' or 'float16'."
            )

        requested = (device or "auto").strip().lower()
        if requested in {"", "auto"}:
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested)

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if self.device.type == "mps":
            backends = getattr(torch, "backends", None)
            mps = getattr(backends, "mps", None)
            if mps is None or not mps.is_available():
                raise RuntimeError("MPS was requested but is not available.")
        if self.device.type not in {"cpu", "cuda", "mps"}:
            raise RuntimeError(f"Unsupported fusion device {self.device}.")

        # Fail during construction (not mid-image) if the device cannot allocate.
        torch.zeros(1, device=self.device, dtype=self.dtype)
        self._kernel_tensors: dict[tuple[object, ...], torch.Tensor] = {}

    @classmethod
    def cuda_available(cls) -> bool:
        try:
            return bool(torch.cuda.is_available())
        except Exception:  # pragma: no cover - depends on the environment
            return False

    # -- shared-core protocol ---------------------------------------------

    def context(self):
        return torch.inference_mode()

    def scalar(self, value: float) -> float:
        return float(value)

    def prepare(self, candidates, noise_map):
        arrays = [
            torch.from_numpy(np.ascontiguousarray(candidate)).to(
                self.device, self.dtype
            )
            for candidate in candidates
        ]
        noise = torch.from_numpy(np.ascontiguousarray(noise_map)).to(
            self.device, self.dtype
        )
        return arrays, noise

    def to_numpy(self, value: torch.Tensor) -> np.ndarray:
        return np.ascontiguousarray(
            value.detach().to(device="cpu", dtype=torch.float32).numpy()
        )

    def finalize_image(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def luma(self, rgb: torch.Tensor) -> torch.Tensor:
        return (
            0.299 * rgb[..., 0]
            + 0.587 * rgb[..., 1]
            + 0.114 * rgb[..., 2]
        )

    def clip(self, image, low, high):
        return torch.clamp(image, min=float(low), max=float(high))

    def power(self, image, exponent):
        return torch.pow(image, float(exponent))

    def exp(self, image):
        return torch.exp(image)

    def sqrt(self, image):
        return torch.sqrt(image)

    def abs(self, image):
        return torch.abs(image)

    def maximum(self, image, value):
        return torch.clamp_min(image, min=float(value))

    def where(self, condition, first, second):
        if not isinstance(first, torch.Tensor):
            first = torch.as_tensor(first, dtype=self.dtype, device=self.device)
        if not isinstance(second, torch.Tensor):
            second = torch.as_tensor(second, dtype=self.dtype, device=self.device)
        return torch.where(condition, first, second)

    def sum(self, image, axis, keepdims=False):
        return torch.sum(image, dim=int(axis), keepdim=bool(keepdims))

    def max(self, image, axis):
        return torch.amax(image, dim=int(axis))

    def stack(self, values, axis):
        return torch.stack(values, dim=int(axis))

    def iadd(self, left, right):
        left.add_(right)
        return left

    def normalize(self, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        if weights.dim() != 3 or weights.shape[2] != CANDIDATE_COUNT:
            raise ValueError(
                f"weights must be HxWx{CANDIDATE_COUNT}, got {tuple(weights.shape)}."
            )
        if not bool(torch.isfinite(weights).all()):
            raise ValueError("weights contain NaN or infinite values.")

        numerator = torch.clamp(weights, min=0.0) + float(epsilon)
        denominator = torch.sum(numerator, dim=2, keepdim=True)
        safe = torch.where(
            denominator > 0.0, denominator, torch.ones_like(denominator)
        )
        result = numerator / safe
        zero = denominator[..., 0] <= 0.0
        return torch.where(
            zero[..., None],
            torch.full_like(result, 1.0 / CANDIDATE_COUNT),
            result,
        )

    def refine(self, guide, weights, radius, eps):
        mean_guide = self.box_filter(guide, int(radius))
        mean_source = self.box_filter(weights, int(radius))
        variance = (
            self.box_filter(guide * guide, int(radius)) - mean_guide * mean_guide
        )
        covariance = (
            self.box_filter(guide[..., None] * weights, int(radius))
            - mean_guide[..., None] * mean_source
        )
        a = covariance / (variance[..., None] + float(eps))
        b = mean_source - a * mean_guide[..., None]
        return (
            self.box_filter(a, int(radius)) * guide[..., None]
            + self.box_filter(b, int(radius))
        )

    # -- pyramids ----------------------------------------------------------

    def gaussian_pyramid(self, image, levels):
        from .lumen_fuse import _gaussian_pyramid_core

        return _gaussian_pyramid_core(image, levels, self)

    def laplacian_pyramid(self, image, levels):
        from .lumen_fuse import _laplacian_pyramid_core

        return _laplacian_pyramid_core(image, levels, self)

    def collapse_pyramid(self, pyramid):
        from .lumen_fuse import _collapse_pyramid_core

        return _collapse_pyramid_core(pyramid, self)

    def pyr_down(self, image):
        channels, ndim = self._to_chw(image)
        padded = self._pad_chw(channels, 2, 2, reflect101=True)
        kernel = self._kernel(("pyr_down",), _PYR_DOWN_KERNEL)
        result = self._conv(padded, kernel, stride=2)
        return self._from_chw(result, ndim)

    def pyr_up(self, image, dst_size):
        channels, ndim = self._to_chw(image)
        count, height, width = channels.shape
        dst_width, dst_height = int(dst_size[0]), int(dst_size[1])
        if dst_height not in (2 * height - 1, 2 * height) or dst_width not in (
            2 * width - 1,
            2 * width,
        ):
            raise ValueError(
                f"pyr_up destination {dst_size} is not compatible with "
                f"source {(width, height)}."
            )

        canvas = torch.zeros(
            (count, dst_height, dst_width),
            dtype=self.dtype,
            device=self.device,
        )
        canvas[:, 0 : 2 * height : 2, 0 : 2 * width : 2] = channels
        padded = self._pad_chw(canvas, 2, 2, reflect101=True)
        kernel = self._kernel(("pyr_up",), _PYR_UP_KERNEL)
        result = self._conv(padded, kernel)
        return self._from_chw(result, ndim)

    # -- filters -----------------------------------------------------------

    def gaussian_blur(self, image, sigma):
        channels, ndim = self._to_chw(image)
        taps = _gaussian_kernel_1d(float(sigma))
        radius = len(taps) // 2
        padded = self._pad_chw(channels, radius, radius, reflect101=True)
        kernel = self._kernel(("gaussian", float(sigma)), taps)
        result = self._conv(padded, kernel.view(1, -1))
        result = self._conv(result, kernel.view(-1, 1))
        return self._from_chw(result, ndim)

    def laplacian(self, image, ksize):
        channels, ndim = self._to_chw(image)
        kernel_values = _laplacian_kernel_2d(int(ksize))
        radius = max(1, int(ksize) // 2)
        padded = self._pad_chw(channels, radius, radius, reflect101=True)
        kernel = self._kernel(("laplacian", int(ksize)), kernel_values)
        result = self._conv(padded, kernel)
        return self._from_chw(result, ndim)

    def box_blur(self, image, size):
        return self._box(image, int(size), reflect101=True)

    def box_filter(self, image, radius):
        return self._box(image, 2 * int(radius) + 1, reflect101=False)

    def _box(self, image, size: int, *, reflect101: bool):
        if size <= 1:
            return image
        channels, ndim = self._to_chw(image)
        radius = size // 2
        padded = self._pad_chw(channels, radius, radius, reflect101=reflect101)
        values = np.full((size, size), 1.0 / (size * size), dtype=np.float32)
        kernel = self._kernel(("box", size), values)
        result = self._conv(padded, kernel)
        return self._from_chw(result, ndim)

    # -- layout / padding / convolution helpers ---------------------------

    def _to_chw(self, image: torch.Tensor):
        if image.dim() == 2:
            return image.unsqueeze(0), 2
        if image.dim() == 3:
            return image.permute(2, 0, 1).contiguous(), 3
        raise ValueError(f"Expected a 2-D or HxWxC tensor, got {tuple(image.shape)}.")

    def _from_chw(self, image: torch.Tensor, ndim: int):
        if ndim == 2:
            return image[0]
        return image.permute(1, 2, 0)

    def _kernel(self, key, values: np.ndarray) -> torch.Tensor:
        cached = self._kernel_tensors.get(key)
        if cached is None:
            cached = torch.from_numpy(
                np.ascontiguousarray(values, dtype=np.float32)
            ).to(self.device, self.dtype)
            self._kernel_tensors[key] = cached
        return cached

    def _conv(self, image: torch.Tensor, weight: torch.Tensor, stride: int = 1):
        channels = image.shape[0]
        kernel = (
            weight.view(1, 1, *weight.shape)
            .expand(channels, 1, *weight.shape)
            .contiguous()
        )
        result = F.conv2d(
            image.unsqueeze(0), kernel, stride=stride, groups=channels
        )
        return result[0]

    def _pad_chw(
        self, image: torch.Tensor, pad_h: int, pad_w: int, *, reflect101: bool
    ) -> torch.Tensor:
        if pad_h:
            image = image.index_select(
                1, self._reflect_index(image.shape[1], pad_h, reflect101)
            )
        if pad_w:
            image = image.index_select(
                2, self._reflect_index(image.shape[2], pad_w, reflect101)
            )
        return image

    def _reflect_index(self, size: int, pad: int, reflect101: bool) -> torch.Tensor:
        if size == 1:
            return torch.zeros(size + 2 * pad, dtype=torch.long, device=self.device)
        if reflect101:
            period = 2 * (size - 1)
        else:
            period = 2 * size
        index = torch.arange(
            -pad, size + pad, dtype=torch.long, device=self.device
        )
        index = torch.remainder(index, period)
        if reflect101:
            return torch.where(index < size, index, period - index)
        return torch.where(index < size, index, period - 1 - index)
