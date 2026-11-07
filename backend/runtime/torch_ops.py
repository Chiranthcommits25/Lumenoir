"""PyTorch image operations used by the GPU pipeline.

The CPU reference remains NumPy/OpenCV. This module mirrors the OpenCV kernels
where practical and keeps every tensor on one device for a whole request.

Importing this module does not import PyTorch. :class:`TorchImageOps` imports it
in ``__init__``; CPU-only deployments never construct it.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .kernels import (
    PYR_DOWN_KERNEL,
    PYR_UP_KERNEL,
    gaussian_kernel_1d,
    laplacian_kernel_2d,
)

_CANDIDATE_COUNT = 3
_SRGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
_XYZ_TO_SRGB = (
    (3.2404542, -1.5371385, -0.4985314),
    (-0.9692660, 1.8760108, 0.0415560),
    (0.0556434, -0.2040259, 1.0572252),
)
_WHITE = (0.95047, 1.0, 1.08883)
_NOISE_KERNEL = np.array(
    [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]], dtype=np.float32
)
_SOBEL_X = np.array(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float32
)
_LAB_LUT_CACHE: np.ndarray | None = None


def _build_lab_lut() -> np.ndarray:
    """OpenCV 5's 33x33x33 RGB-to-LAB node table (int16 scale 16384)."""
    global _LAB_LUT_CACHE
    if _LAB_LUT_CACHE is not None:
        return _LAB_LUT_CACHE
    base = 1 << 14
    dimension = 33
    values = np.arange(dimension, dtype=np.float64) / (dimension - 1)
    r, g, b = np.meshgrid(values, values, values, indexing="ij")
    linear = lambda c: np.where(  # noqa: E731
        c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4
    )
    r, g, b = linear(r), linear(g), linear(b)
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    x = (r * matrix[0, 0] + g * matrix[0, 1] + b * matrix[0, 2]) / white[0]
    y = r * matrix[1, 0] + g * matrix[1, 1] + b * matrix[1, 2]
    z = (r * matrix[2, 0] + g * matrix[2, 1] + b * matrix[2, 2]) / white[2]
    threshold = 216.0 / 24389.0
    scale = 841.0 / 108.0
    bias = 16.0 / 116.0

    def f(value):
        return np.where(value > threshold, np.cbrt(value), value * scale + bias)

    fx, fy, fz = f(x), f(y), f(z)
    lightness = np.where(y > threshold, 116.0 * fy - 16.0, (29.0**3 / 27.0) * y)
    a_channel = 500.0 * (fx - fy)
    b_channel = 200.0 * (fy - fz)
    lut = np.stack(
        [
            np.rint(base * lightness / 100.0),
            np.rint(base * (a_channel + 128.0) / 256.0),
            np.rint(base * (b_channel + 128.0) / 256.0),
        ],
        axis=-1,
    )
    _LAB_LUT_CACHE = lut.astype(np.int32)
    return _LAB_LUT_CACHE


class TorchImageOps:
    """Device-resident image primitives and fusion protocol."""

    name = "torch"

    def __init__(self, runtime=None, *, device: str | None = None, precision: str | None = None):
        if runtime is not None:
            device = runtime.device
            precision = runtime.precision
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("PyTorch is required for the accelerator pipeline.") from exc

        self._torch = torch
        self.F = functional
        choice = (precision or "float32").strip().lower()
        if choice in {"float32", "fp32"}:
            self.dtype = torch.float32
        elif choice in {"float16", "fp16", "half"}:
            self.dtype = torch.float16
        else:
            raise ValueError(f"Unsupported precision {precision!r}.")
        self.device = torch.device((device or "cpu").strip().lower())
        if self.device.type not in {"cpu", "cuda", "mps"}:
            raise RuntimeError(f"Unsupported torch device {self.device}.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if self.device.type == "mps":
            backends = getattr(torch, "backends", None)
            mps = getattr(backends, "mps", None)
            if mps is None or not mps.is_available():
                raise RuntimeError("MPS was requested but is not available.")
        torch.zeros(1, device=self.device, dtype=self.dtype)
        self._kernel_tensors: dict[tuple[object, ...], Any] = {}

    # -- basic protocol ----------------------------------------------------

    @property
    def torch(self):
        return self._torch

    def context(self):
        return self._torch.inference_mode()

    def scalar(self, value: float) -> float:
        return float(value)

    def prepare(self, candidates, noise_map):
        converted = [self.ensure(candidate) for candidate in candidates]
        return converted, self.ensure(noise_map)

    def ensure(self, value):
        if self._torch.is_tensor(value):
            return value.to(device=self.device, dtype=self.dtype)
        return self._torch.from_numpy(np.ascontiguousarray(value)).to(
            device=self.device, dtype=self.dtype
        )

    def to_numpy(self, value):
        return (
            value.detach()
            .to(device="cpu", dtype=self._torch.float32)
            .contiguous()
            .numpy()
        )

    def finalize_image(self, image):
        return image

    # -- scalar/vector math protocol --------------------------------------

    def luma(self, rgb):
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    def clip(self, image, low, high):
        return self._torch.clamp(image, min=float(low), max=float(high))

    def power(self, image, exponent):
        return self._torch.pow(image, float(exponent))

    def exp(self, image):
        return self._torch.exp(image)

    def sqrt(self, image):
        return self._torch.sqrt(image)

    def abs(self, image):
        return self._torch.abs(image)

    def maximum(self, image, value):
        return self._torch.clamp_min(image, min=float(value))

    def where(self, condition, first, second):
        if not self._torch.is_tensor(first):
            first = self._torch.as_tensor(first, dtype=self.dtype, device=self.device)
        if not self._torch.is_tensor(second):
            second = self._torch.as_tensor(second, dtype=self.dtype, device=self.device)
        return self._torch.where(condition, first, second)

    def sum(self, image, axis, keepdims=False):
        return self._torch.sum(image, dim=int(axis), keepdim=bool(keepdims))

    def max(self, image, axis):
        return self._torch.amax(image, dim=int(axis))

    def stack(self, values, axis):
        return self._torch.stack(values, dim=int(axis))

    def iadd(self, left, right):
        left.add_(right)
        return left

    def normalize(self, weights, epsilon):
        if weights.dim() != 3 or weights.shape[2] != _CANDIDATE_COUNT:
            raise ValueError(
                f"weights must be HxWx{_CANDIDATE_COUNT}, got {tuple(weights.shape)}."
            )
        if not bool(self._torch.isfinite(weights).all()):
            raise ValueError("weights contain NaN or infinite values.")
        numerator = self._torch.clamp(weights, min=0.0) + float(epsilon)
        denominator = self._torch.sum(numerator, dim=2, keepdim=True)
        safe = self._torch.where(
            denominator > 0.0, denominator, self._torch.ones_like(denominator)
        )
        result = numerator / safe
        zero = denominator[..., 0] <= 0.0
        return self._torch.where(
            zero[..., None],
            self._torch.full_like(result, 1.0 / _CANDIDATE_COUNT),
            result,
        )

    def refine(self, guide, weights, radius, eps):
        mean_guide = self.box_filter(guide, int(radius))
        mean_source = self.box_filter(weights, int(radius))
        variance = self.box_filter(guide * guide, int(radius)) - mean_guide * mean_guide
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

    # -- layout / padding / convolution -----------------------------------

    def _to_chw(self, image):
        if image.dim() == 2:
            return image.unsqueeze(0), 2
        if image.dim() == 3:
            return image.permute(2, 0, 1).contiguous(), 3
        raise ValueError(f"Expected a 2-D or HxWxC tensor, got {tuple(image.shape)}.")

    def _from_chw(self, image, ndim):
        if ndim == 2:
            return image[0]
        return image.permute(1, 2, 0)

    def _kernel(self, key, values: np.ndarray, *, dtype=None):
        target_dtype = dtype if dtype is not None else self.dtype
        cache_key = (key, target_dtype)
        cached = self._kernel_tensors.get(cache_key)
        if cached is None:
            cached = self._torch.from_numpy(
                np.ascontiguousarray(values, dtype=np.float32)
            ).to(self.device, target_dtype)
            self._kernel_tensors[cache_key] = cached
        return cached

    def _conv(self, image, weight, stride: int = 1):
        # Convolution has no implicit dtype promotion: a float16 kernel against
        # a float32 operand raises "Input type and weight type should be the
        # same". Blind the kernel to the operand instead of the operand to the
        # kernel, so the deliberately float32 statistics paths keep their
        # float32 accumulation and the float16 path never widens the image.
        # The kernel is a few dozen taps and is cached per dtype, so this is
        # at most one conversion per filter and precision.
        if weight.dtype != image.dtype:
            weight = weight.to(image.dtype)
        channels = image.shape[0]
        kernel = (
            weight.view(1, 1, *weight.shape)
            .expand(channels, 1, *weight.shape)
            .contiguous()
        )
        result = self.F.conv2d(
            image.unsqueeze(0), kernel, stride=stride, groups=channels
        )
        return result[0]

    def filter2d(self, image, kernel: np.ndarray, *, reflect101: bool = False):
        # Statistics run in float32 even when the pipeline works in float16.
        channels, ndim = self._to_chw(image.to(self._torch.float32))
        radius_y = kernel.shape[0] // 2
        radius_x = kernel.shape[1] // 2
        padded = self._pad_chw(
            channels, radius_y, radius_x, reflect101=reflect101
        )
        weights = self._kernel(
            ("filter2d", kernel.tobytes()), kernel, dtype=self._torch.float32
        )
        result = self.F.conv2d(
            padded.unsqueeze(0), weights.view(1, 1, *weights.shape)
        )[0]
        return self._from_chw(result, ndim)

    def _pad_chw(self, image, pad_h: int, pad_w: int, *, reflect101: bool):
        if pad_h:
            image = image.index_select(
                1, self._reflect_index(image.shape[1], pad_h, reflect101)
            )
        if pad_w:
            image = image.index_select(
                2, self._reflect_index(image.shape[2], pad_w, reflect101)
            )
        return image

    def _reflect_index(self, size: int, pad: int, reflect101: bool):
        return self._reflect_range(size, -pad, size + pad, reflect101)

    def _reflect_range(self, size: int, start: int, stop: int, reflect101: bool):
        if size == 1:
            return self._torch.zeros(
                stop - start, dtype=self._torch.long, device=self.device
            )
        period = 2 * (size - 1) if reflect101 else 2 * size
        index = self._torch.arange(
            start, stop, dtype=self._torch.long, device=self.device
        )
        index = self._torch.remainder(index, period)
        if reflect101:
            return self._torch.where(index < size, index, period - index)
        return self._torch.where(index < size, index, period - 1 - index)

    def pad_reflect101(self, image, bottom: int, right: int):
        channels, ndim = self._to_chw(image)
        if bottom:
            channels = channels.index_select(
                1,
                self._reflect_range(
                    channels.shape[1], 0, channels.shape[1] + bottom, True
                ),
            )
        if right:
            channels = channels.index_select(
                2,
                self._reflect_range(
                    channels.shape[2], 0, channels.shape[2] + right, True
                ),
            )
        return self._from_chw(channels, ndim)

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
        # Match the operand: callers such as noise_map pass a float32 tensor
        # from filter2d while the pipeline works in float16.
        kernel = self._kernel(("box", size), values, dtype=channels.dtype)
        result = self._conv(padded, kernel)
        return self._from_chw(result, ndim)

    def gaussian_blur(self, image, sigma):
        channels, ndim = self._to_chw(image)
        taps = gaussian_kernel_1d(float(sigma))
        radius = len(taps) // 2
        padded = self._pad_chw(channels, radius, radius, reflect101=True)
        kernel = self._kernel(("gaussian", float(sigma)), taps, dtype=channels.dtype)
        result = self._conv(padded, kernel.view(1, -1))
        result = self._conv(result, kernel.view(-1, 1))
        return self._from_chw(result, ndim)

    def laplacian(self, image, ksize):
        channels, ndim = self._to_chw(image)
        values = laplacian_kernel_2d(int(ksize))
        radius = max(1, int(ksize) // 2)
        padded = self._pad_chw(channels, radius, radius, reflect101=True)
        kernel = self._kernel(("laplacian", int(ksize)), values, dtype=channels.dtype)
        result = self._conv(padded, kernel)
        return self._from_chw(result, ndim)

    def resize(self, image, size, *, mode: str = "area"):
        """Resize to ``(width, height)`` preserving the input rank."""
        width, height = int(size[0]), int(size[1])
        channels, ndim = self._to_chw(image)
        src_h, src_w = channels.shape[-2:]
        if (src_w, src_h) == (width, height):
            return image
        upscaling = width > src_w or height > src_h
        if mode == "area" and not upscaling:
            result = self._area_resize(channels, width, height)
            return self._from_chw(result, ndim)
        chosen = "bilinear" if upscaling or mode == "bilinear" else mode
        kwargs = {}
        if chosen == "bilinear":
            kwargs["align_corners"] = False
            kwargs["antialias"] = True
        result = self.F.interpolate(
            channels.unsqueeze(0), size=(height, width), mode=chosen, **kwargs
        )[0]
        return self._from_chw(result, ndim)

    def _axis_area_weights(self, source: int, destination: int):
        """Exact cv2 INTER_AREA overlap weights for one axis."""
        torch = self._torch
        scale = float(source) / float(destination)
        positions = torch.arange(
            destination, device=self.device, dtype=torch.float32
        )
        starts_exact = positions * scale
        stops_exact = (positions + 1.0) * scale
        starts = torch.floor(starts_exact)
        spans = (torch.ceil(stops_exact) - starts).to(torch.int64)
        max_span = int(spans.max().item())
        weights = torch.zeros(
            (destination, max_span), dtype=torch.float32, device=self.device
        )
        indices = torch.zeros(
            (destination, max_span), dtype=torch.int64, device=self.device
        )
        for offset in range(max_span):
            position = starts + float(offset)
            lower = torch.maximum(position, starts_exact)
            upper = torch.minimum(position + 1.0, stops_exact)
            weight = torch.clamp(upper - lower, min=0.0).to(torch.float32)
            index = position.to(torch.int64).clamp(0, source - 1)
            weights[:, offset] = weight
            indices[:, offset] = index
        weights = weights / torch.clamp(
            weights.sum(dim=1, keepdim=True), min=1e-12
        )
        return indices, weights

    def _area_resize(self, channels, width: int, height: int):
        torch = self._torch
        work = channels.to(torch.float32)
        x_index, x_weight = self._axis_area_weights(work.shape[2], width)
        y_index, y_weight = self._axis_area_weights(work.shape[1], height)
        intermediate = torch.zeros(
            (work.shape[0], work.shape[1], width),
            dtype=torch.float32,
            device=self.device,
        )
        for offset in range(x_index.shape[1]):
            intermediate += work.index_select(2, x_index[:, offset]) * x_weight[
                None, None, :, offset
            ]
        result = torch.zeros(
            (work.shape[0], height, width), dtype=torch.float32, device=self.device
        )
        for offset in range(y_index.shape[1]):
            result += intermediate.index_select(1, y_index[:, offset]) * y_weight[
                None, :, offset
            ][..., None]
        return result.to(channels.dtype)

    def pyr_down(self, image):
        channels, ndim = self._to_chw(image)
        padded = self._pad_chw(channels, 2, 2, reflect101=True)
        kernel = self._kernel(("pyr_down",), PYR_DOWN_KERNEL, dtype=channels.dtype)
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
                f"pyr_up destination {dst_size} is not compatible with source "
                f"{(width, height)}."
            )
        canvas = self._torch.zeros(
            (count, dst_height, dst_width),
            dtype=channels.dtype,
            device=channels.device,
        )
        canvas[:, 0 : 2 * height : 2, 0 : 2 * width : 2] = channels
        padded = self._pad_chw(canvas, 2, 2, reflect101=True)
        kernel = self._kernel(("pyr_up",), PYR_UP_KERNEL, dtype=channels.dtype)
        result = self._conv(padded, kernel)
        return self._from_chw(result, ndim)

    # -- colour spaces -----------------------------------------------------

    def rgb_to_lab(self, rgb):
        try:
            major = int(cv2.__version__.split(".", 1)[0])
        except (AttributeError, ValueError):  # pragma: no cover - defensive
            major = 4
        if major >= 5:
            return self._rgb_to_lab_lut(rgb)
        torch = self._torch
        work = rgb.to(torch.float32)
        r, g, b = work[..., 0], work[..., 1], work[..., 2]
        r = torch.where(r > 0.04045, ((r + 0.055) / 1.055) ** 2.4, r / 12.92)
        g = torch.where(g > 0.04045, ((g + 0.055) / 1.055) ** 2.4, g / 12.92)
        b = torch.where(b > 0.04045, ((b + 0.055) / 1.055) ** 2.4, b / 12.92)
        x = _SRGB_TO_XYZ[0][0] * r + _SRGB_TO_XYZ[0][1] * g + _SRGB_TO_XYZ[0][2] * b
        y = _SRGB_TO_XYZ[1][0] * r + _SRGB_TO_XYZ[1][1] * g + _SRGB_TO_XYZ[1][2] * b
        z = _SRGB_TO_XYZ[2][0] * r + _SRGB_TO_XYZ[2][1] * g + _SRGB_TO_XYZ[2][2] * b
        xn, yn, zn = _WHITE
        delta = 6.0 / 29.0

        def f(t):
            return torch.where(
                t > delta**3,
                torch.clamp_min(t, 0.0).pow(1.0 / 3.0),
                t / (3.0 * delta * delta) + 4.0 / 29.0,
            )

        fx, fy, fz = f(x / xn), f(y / yn), f(z / zn)
        lab = torch.stack(
            [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], dim=-1
        )
        return lab.to(rgb.dtype)

    def _rgb_to_lab_lut(self, rgb):
        torch = self._torch
        base = 1 << 14
        table = getattr(self, "_lab_lut", None)
        if table is None:
            table = torch.from_numpy(_build_lab_lut()).to(
                device=self.device, dtype=torch.int32
            )
            self._lab_lut = table
        work = torch.clamp(rgb.to(torch.float32), min=0.0, max=1.0)
        height, width = work.shape[:2]
        result = torch.empty(
            (height, width, 3), dtype=torch.float32, device=self.device
        )
        rows = max(1, 1_000_000 // max(width, 1))
        for y0 in range(0, height, rows):
            y1 = min(y0 + rows, height)
            block = torch.round(work[y0:y1] * float(base)).to(torch.int64)
            tx = block[..., 0] >> 9
            ty = block[..., 1] >> 9
            tz = block[..., 2] >> 9
            fx = (block[..., 0] >> 5) & 15
            fy = (block[..., 1] >> 5) & 15
            fz = (block[..., 2] >> 5) & 15
            total = torch.zeros(
                (y1 - y0, width, 3), dtype=torch.int64, device=self.device
            )
            for ix in (0, 1):
                wx = fx if ix else 16 - fx
                for iy in (0, 1):
                    wy = fy if iy else 16 - fy
                    for iz in (0, 1):
                        wz = fz if iz else 16 - fz
                        xi = torch.clamp(tx + ix, max=32)
                        yi = torch.clamp(ty + iy, max=32)
                        zi = torch.clamp(tz + iz, max=32)
                        total += table[xi, yi, zi] * (
                            wx * wy * wz
                        )[..., None]
            values = (total + 2048) >> 12
            result[y0:y1, :, 0] = values[..., 0] * (100.0 / base)
            result[y0:y1, :, 1] = values[..., 1] * (256.0 / base) - 128.0
            result[y0:y1, :, 2] = values[..., 2] * (256.0 / base) - 128.0
        return result

    def lab_to_rgb(self, lab):
        torch = self._torch
        work = lab.to(torch.float32)
        lightness, a, b = work[..., 0], work[..., 1], work[..., 2]
        fy = (lightness + 16.0) / 116.0
        fx = fy + a / 500.0
        fz = fy - b / 200.0
        xn, yn, zn = _WHITE
        delta = 6.0 / 29.0

        def finv(t):
            return torch.where(
                t > delta, t**3, 3.0 * delta * delta * (t - 4.0 / 29.0)
            )

        x, y, z = xn * finv(fx), yn * finv(fy), zn * finv(fz)
        r = _XYZ_TO_SRGB[0][0] * x + _XYZ_TO_SRGB[0][1] * y + _XYZ_TO_SRGB[0][2] * z
        g = _XYZ_TO_SRGB[1][0] * x + _XYZ_TO_SRGB[1][1] * y + _XYZ_TO_SRGB[1][2] * z
        bl = _XYZ_TO_SRGB[2][0] * x + _XYZ_TO_SRGB[2][1] * y + _XYZ_TO_SRGB[2][2] * z

        def gamma(c):
            return torch.where(
                c > 0.0031308,
                1.055 * torch.clamp_min(c, 0.0).pow(1.0 / 2.4) - 0.055,
                12.92 * c,
            )

        return torch.stack([gamma(r), gamma(g), gamma(bl)], dim=-1).to(lab.dtype)

    def rgb_to_ycrcb(self, rgb):
        y = self.luma(rgb)
        cr = (rgb[..., 0] - y) * 0.713 + 0.5
        cb = (rgb[..., 2] - y) * 0.564 + 0.5
        return self._torch.stack([y, cr, cb], dim=-1)

    def ycrcb_to_rgb(self, ycrcb):
        y = ycrcb[..., 0]
        cr = ycrcb[..., 1] - 0.5
        cb = ycrcb[..., 2] - 0.5
        r = y + 1.403 * cr
        g = y - 0.714 * cr - 0.344 * cb
        b = y + 1.773 * cb
        return self._torch.stack([r, g, b], dim=-1)

    # -- CLAHE -------------------------------------------------------------

    @staticmethod
    def _clahe_padding(height: int, width: int, tiles_y: int, tiles_x: int):
        """Mirror OpenCV's padding rule for the installed major version."""
        try:
            major = int(cv2.__version__.split(".", 1)[0])
        except (AttributeError, ValueError):  # pragma: no cover - defensive
            major = 4
        if major >= 5:
            # OpenCV 5 adds a full tile on a dimension that is already divisible
            # whenever the other dimension needs padding.
            if width % tiles_x or height % tiles_y:
                return tiles_y - (height % tiles_y), tiles_x - (width % tiles_x)
            return 0, 0
        return (tiles_y - (height % tiles_y)) % tiles_y, (
            tiles_x - (width % tiles_x)
        ) % tiles_x

    def clahe_uint16(self, source, clip_limit: float, tiles: tuple[int, int]):
        """CLAHE on a uint16 single-channel image, mirroring OpenCV.

        The histogram uses 65536 bins, the clip limit is
        ``int(clipLimit * tileArea / 65536)`` and the interpolation follows
        OpenCV's tile-centre geometry. Differences from OpenCV are at most one
        uint16 level on the tested sizes.
        """
        torch = self._torch
        tiles_x, tiles_y = int(tiles[0]), int(tiles[1])
        if tiles_x < 1 or tiles_y < 1:
            raise ValueError(f"tiles must be positive, got {tiles!r}.")
        if source.dim() != 2:
            raise ValueError(f"CLAHE expects a 2-D image, got {tuple(source.shape)}.")
        height, width = source.shape
        if height < tiles_y or width < tiles_x:
            raise ValueError(
                f"CLAHE tiles {tiles} are larger than the image {width}x{height}."
            )

        bottom, right = self._clahe_padding(height, width, tiles_y, tiles_x)
        padded = self.pad_reflect101(source.to(torch.int64), bottom, right)
        tile_h = padded.shape[0] // tiles_y
        tile_w = padded.shape[1] // tiles_x
        area = tile_h * tile_w
        hist_size = 65536
        clip = max(int(float(clip_limit) * area / hist_size), 1)
        lut_scale = float(hist_size - 1) / float(area)

        tiles_flat = (
            padded.reshape(tiles_y, tile_h, tiles_x, tile_w)
            .permute(0, 2, 1, 3)
            .reshape(tiles_y * tiles_x, area)
        )
        hist = torch.empty(
            (tiles_y * tiles_x, hist_size),
            dtype=torch.int64,
            device=self.device,
        )
        block = max(1, 4_000_000 // max(area, 1))
        for start in range(0, tiles_flat.shape[0], block):
            values = tiles_flat[start : start + block]
            ids = (
                torch.arange(values.shape[0], device=self.device).view(-1, 1)
                * hist_size
                + values
            )
            counts = torch.bincount(
                ids.reshape(-1), minlength=values.shape[0] * hist_size
            )
            hist[start : start + values.shape[0]] = counts.view(-1, hist_size)

        clipped = (hist - clip).clamp(min=0).sum(dim=1)
        hist = hist.clamp(max=clip)
        hist += (clipped // hist_size).unsqueeze(1)
        residual = clipped % hist_size
        step = torch.clamp(hist_size // residual.clamp(min=1), min=1)
        bins = torch.arange(hist_size, dtype=torch.int32, device=self.device)
        mask = ((bins.unsqueeze(0) % step.to(torch.int32).unsqueeze(1)) == 0) & (
            (bins.unsqueeze(0) // step.to(torch.int32).unsqueeze(1))
            < residual.to(torch.int32).unsqueeze(1)
        )
        hist += mask.to(torch.int64)
        lut = (
            torch.round(hist.cumsum(dim=1).to(torch.float32) * lut_scale)
            .clamp(0, hist_size - 1)
            .to(torch.int32)
            .view(tiles_y, tiles_x, hist_size)
        )

        xs = torch.arange(width, device=self.device, dtype=torch.float32)
        txf = xs / float(tile_w) - 0.5
        tx1 = torch.floor(txf).to(torch.int64)
        tx2 = tx1 + 1
        xa = txf - tx1.to(torch.float32)
        xa1 = 1.0 - xa
        tx1 = tx1.clamp(0, tiles_x - 1)
        tx2 = tx2.clamp(0, tiles_x - 1)

        ys = torch.arange(height, device=self.device, dtype=torch.float32)
        tyf = ys / float(tile_h) - 0.5
        ty1 = torch.floor(tyf).to(torch.int64)
        ty2 = ty1 + 1
        ya = tyf - ty1.to(torch.float32)
        ya1 = 1.0 - ya
        ty1 = ty1.clamp(0, tiles_y - 1)
        ty2 = ty2.clamp(0, tiles_y - 1)

        values = source.to(torch.int64)
        result = torch.empty(
            (height, width), dtype=torch.float32, device=self.device
        )
        rows_per_chunk = max(1, 2_000_000 // max(width, 1))
        for y0 in range(0, height, rows_per_chunk):
            y1 = min(y0 + rows_per_chunk, height)
            block_values = values[y0:y1]
            top = lut[ty1[y0:y1, None], tx1[None, :], block_values]
            top2 = lut[ty1[y0:y1, None], tx2[None, :], block_values]
            bottom = lut[ty2[y0:y1, None], tx1[None, :], block_values]
            bottom2 = lut[ty2[y0:y1, None], tx2[None, :], block_values]
            top_row = (
                top.to(torch.float32) * xa1[None, :]
                + top2.to(torch.float32) * xa[None, :]
            )
            bottom_row = (
                bottom.to(torch.float32) * xa1[None, :]
                + bottom2.to(torch.float32) * xa[None, :]
            )
            row = top_row * ya1[y0:y1, None] + bottom_row * ya[y0:y1, None]
            result[y0:y1] = row

        return (
            torch.round(result)
            .clamp(0, 65535)
            .to(torch.uint16)
        )

    # -- statistics / noise ------------------------------------------------

    def quantile(self, values, q):
        torch = self._torch
        flat = values.reshape(-1).to(torch.float32)
        query_list = [float(item) for item in q] if isinstance(q, (list, tuple)) else None
        queries = query_list if query_list is not None else float(q)
        if flat.numel() <= 16_000_000:
            q_tensor = torch.as_tensor(queries, dtype=torch.float32, device=self.device)
            return torch.quantile(flat, q_tensor)
        values_out = []
        n = flat.numel()
        for query in query_list if query_list is not None else [queries]:
            position = (n - 1) * query
            low = math.floor(position)
            high = math.ceil(position)
            v_low = torch.kthvalue(flat, low + 1).values
            if high == low:
                values_out.append(v_low)
            else:
                v_high = torch.kthvalue(flat, high + 1).values
                fraction = position - low
                values_out.append(v_low + (v_high - v_low) * fraction)
        return torch.stack(values_out) if len(values_out) > 1 else values_out[0]

    def noise_response(self, gray):
        response = self.filter2d(gray, _NOISE_KERNEL)
        return self._torch.abs(response)[1:-1, 1:-1]

    def gradient_magnitude(self, gray):
        dx = self.filter2d(gray, _SOBEL_X)
        dy = self.filter2d(gray, _SOBEL_X.T.copy())
        return self._torch.hypot(dx, dy)[1:-1, 1:-1]

    def flat_mask(self, gray, percentile: float = 50.0):
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"percentile must be in [0, 100], got {percentile!r}.")
        gradient = self.gradient_magnitude(gray)
        if gradient.numel() == 0:
            return self._torch.zeros_like(gradient, dtype=self._torch.bool)
        threshold = self.quantile(gradient, percentile / 100.0)
        mask = gradient <= threshold
        if not bool(mask.any()):
            mask = self._torch.ones_like(gradient, dtype=self._torch.bool)
        return mask

    def noise_sigma(self, gray, percentile: float = 50.0) -> float:
        if gray.shape[0] < 3 or gray.shape[1] < 3:
            return 0.0
        response = self.noise_response(gray)
        mask = self.flat_mask(gray, percentile)
        if mask.shape != response.shape:
            mask = mask[: response.shape[0], : response.shape[1]]
            if mask.shape != response.shape:
                return 0.0
        if not bool(mask.any()):
            return 0.0
        constant = float(np.sqrt(np.pi / 2.0) / 6.0)
        return float(constant * response[mask].mean())

    def noise_map(self, gray, radius: int = 7, *, sigma: float | None = None):
        response = self._torch.abs(
            self.filter2d(gray, _NOISE_KERNEL)
        )
        constant = float(np.sqrt(np.pi / 2.0) / 6.0)
        local = constant * self.box_filter(response, radius)
        if sigma is not None and sigma > 0.0:
            local = self._torch.clamp(local, min=0.5 * float(sigma), max=2.0 * float(sigma))
        return local.to(self.dtype)

    def guided_filter(self, guide, source, radius: int, eps: float):
        if guide.shape != source.shape:
            raise ValueError(
                f"guide {tuple(guide.shape)} and source {tuple(source.shape)} must match."
            )
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps!r}.")
        mean_guide = self.box_filter(guide, radius)
        mean_source = self.box_filter(source, radius)
        covariance = self.box_filter(guide * source, radius) - mean_guide * mean_source
        variance = self.box_filter(guide * guide, radius) - mean_guide * mean_guide
        a = covariance / (variance + float(eps))
        b = mean_source - a * mean_guide
        return self.box_filter(a, radius) * guide + self.box_filter(b, radius)

    def estimate_illumination(
        self,
        rgb,
        *,
        radius: int | None = None,
        eps: float = 1e-3,
        downsample: int = 2,
    ):
        if downsample < 1:
            raise ValueError(f"downsample must be positive, got {downsample!r}.")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps!r}.")
        height, width = rgb.shape[:2]
        guide = self.luma(rgb)
        t0 = self.max(rgb, 2)
        if downsample > 1:
            work_size = (max(1, width // downsample), max(1, height // downsample))
            guide_work = self.resize(guide, work_size, mode="area")
            t0_work = self.resize(t0, work_size, mode="area")
            if radius is None:
                base_radius = max(1, round(0.03 * min(height, width)))
                work_radius = max(1, round(base_radius / downsample))
            else:
                work_radius = int(radius)
        else:
            guide_work = guide
            t0_work = t0
            work_radius = (
                int(radius)
                if radius is not None
                else max(1, round(0.03 * min(height, width)))
            )
        smoothed = self.guided_filter(guide_work, t0_work, work_radius, eps)
        if downsample > 1:
            smoothed = self.resize(smoothed, (width, height), mode="bilinear")
        return self._torch.clamp(smoothed, min=1e-3, max=1.0).to(self.dtype)

    # -- filters -----------------------------------------------------------

    def bilateral(self, channel, d: int = 7, sigma_color: float = 25.0, sigma_space: float = 5.0):
        """Single-channel bilateral filter with OpenCV defaults."""
        torch = self._torch
        if channel.dim() != 2:
            raise ValueError(f"bilateral expects a 2-D image, got {tuple(channel.shape)}.")
        radius = int(d) // 2
        if radius < 1:
            return channel
        work = channel.to(torch.float32)
        padded = self._pad_chw(
            work.unsqueeze(0), radius, radius, reflect101=True
        )[0]
        height, width = work.shape
        accum = torch.zeros_like(work)
        norm = torch.zeros_like(work)
        color_scale = -0.5 / (float(sigma_color) * float(sigma_color))
        space_scale = -0.5 / (float(sigma_space) * float(sigma_space))
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                shifted = padded[
                    radius + dy : radius + dy + height,
                    radius + dx : radius + dx + width,
                ]
                weight = torch.exp(
                    (shifted - work) ** 2 * color_scale
                    + (dx * dx + dy * dy) * space_scale
                )
                accum += weight * shifted
                norm += weight
        return (accum / torch.clamp(norm, min=1e-12)).to(channel.dtype)

    def smoothstep(self, edge0: float, edge1: float, value):
        if edge1 <= edge0:
            raise ValueError(
                f"edge1 must be greater than edge0, got {edge0} and {edge1}."
            )
        t = (
            (value.to(self._torch.float32) - float(edge0))
            / (float(edge1) - float(edge0))
        ).clamp(0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def highlight_shoulder(self, values, start: float = 0.85, width: float = 0.15):
        if width <= 0.0:
            raise ValueError("shoulder width must be positive.")
        start_t = self._torch.tensor(
            start, dtype=self._torch.float32, device=values.device
        )
        width_t = self._torch.tensor(
            width, dtype=self._torch.float32, device=values.device
        )
        compressed = start_t + width_t * self._torch.tanh(
            (values.to(self._torch.float32) - start_t) / width_t
        )
        return self._torch.where(values > start, compressed, values)


__all__ = ["TorchImageOps"]
