"""Zero-DCE++ network, reproduced from the authors' implementation.

Source
------
``Li-Chongyi/Zero-DCE_extension`` → ``Zero-DCE++/model.py`` (branch ``main``).
Architecture verified against the released code and checkpoint:

- seven depthwise-separable blocks (3x3 depthwise + 1x1 pointwise), 32 channels
- skip concatenations ``cat(x3,x4)``, ``cat(x2,x5)``, ``cat(x1,x6)``
- curve map ``alpha = tanh(...)``, applied eight times as
  ``x + alpha * (x**2 - x)``
- downsampling: bilinear, ``align_corners=False`` (PyTorch default)
- upsampling: ``nn.UpsamplingBilinear2d``, which is exactly
  ``interpolate(..., mode="bilinear", align_corners=True)``
- 10,561 parameters, 28 state-dict tensors

Licence
-------
The original repository is released for academic research under
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). Cite:

    Li, Guo, Loy. Learning to Enhance Low-Light Image via Zero-Reference
    Deep Curve Estimation. IEEE TPAMI, 2021. arXiv:2103.00860.

This module is imported only when a Zero-DCE++ model is loaded, so the API
still starts without PyTorch installed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_CHANNELS = 32
CURVE_ITERATIONS = 8


class DepthwiseSeparableConv(nn.Module):
    """``CSDN_Tem`` from the official implementation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.point_conv(self.depth_conv(tensor))


class ZeroDCEPlusPlusNet(nn.Module):
    """Curve estimator. ``forward`` returns ``(enhanced, alpha)``."""

    def __init__(self, scale_factor: int = 12) -> None:
        super().__init__()
        if not 1 <= int(scale_factor) <= 12:
            raise ValueError(f"scale_factor must be in [1, 12], got {scale_factor!r}.")

        self.scale_factor = int(scale_factor)
        self.relu = nn.ReLU(inplace=True)

        channels = FEATURE_CHANNELS
        self.e_conv1 = DepthwiseSeparableConv(3, channels)
        self.e_conv2 = DepthwiseSeparableConv(channels, channels)
        self.e_conv3 = DepthwiseSeparableConv(channels, channels)
        self.e_conv4 = DepthwiseSeparableConv(channels, channels)
        self.e_conv5 = DepthwiseSeparableConv(channels * 2, channels)
        self.e_conv6 = DepthwiseSeparableConv(channels * 2, channels)
        self.e_conv7 = DepthwiseSeparableConv(channels * 2, 3)

    def forward(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.scale_factor == 1:
            downsampled = tensor
        else:
            downsampled = F.interpolate(
                tensor,
                scale_factor=1.0 / self.scale_factor,
                mode="bilinear",
                align_corners=False,
            )

        x1 = self.relu(self.e_conv1(downsampled))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        alpha = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        if self.scale_factor != 1:
            # Equivalent to nn.UpsamplingBilinear2d(scale_factor=...).
            alpha = F.interpolate(
                alpha,
                scale_factor=float(self.scale_factor),
                mode="bilinear",
                align_corners=True,
            )

        enhanced = tensor
        for _ in range(CURVE_ITERATIONS):
            enhanced = enhanced + alpha * (torch.pow(enhanced, 2) - enhanced)

        return enhanced, alpha
