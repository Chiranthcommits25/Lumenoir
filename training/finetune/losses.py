"""Documented Zero-DCE++ training losses.

The first four terms follow the Zero-DCE++ definitions in
``final_decision.md`` §2.6 and the official ``Myloss.py`` scaling recorded in
``docs/zero_dce.md``. The highlight and chromaticity terms are the two
LUMENOIR additions from the same section; they are explicit loss options, not
changes to the inference architecture.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import LossWeights


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    terms: dict[str, torch.Tensor]


def _luminance(image: torch.Tensor) -> torch.Tensor:
    return 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]


def _smoothstep(x: torch.Tensor, low: float = 0.70, high: float = 0.90) -> torch.Tensor:
    t = ((x - low) / (high - low)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def spatial_consistency_loss(output: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """L_spa: preserve local contrast on 4x4 average-pooled regions."""
    if min(output.shape[-2:]) < 4:
        return output.new_zeros(())
    pooled_output = F.avg_pool2d(output, 4, 4)
    pooled_source = F.avg_pool2d(source, 4, 4)
    dx_output = pooled_output[:, :, :, 1:] - pooled_output[:, :, :, :-1]
    dx_source = pooled_source[:, :, :, 1:] - pooled_source[:, :, :, :-1]
    dy_output = pooled_output[:, :, 1:, :] - pooled_output[:, :, :-1, :]
    dy_source = pooled_source[:, :, 1:, :] - pooled_source[:, :, :-1, :]
    return (
        (dx_output.abs() - dx_source.abs()).pow(2).mean()
        + (dy_output.abs() - dy_source.abs()).pow(2).mean()
    )


def exposure_loss(output: torch.Tensor, target: float = 0.6) -> torch.Tensor:
    """L_exp: pull 16x16 local mean luminance toward ``target``."""
    if min(output.shape[-2:]) < 16:
        return output.new_zeros(())
    local = F.avg_pool2d(output, 16, 16)
    return (local - target).abs().mean()


def color_constancy_loss(output: torch.Tensor) -> torch.Tensor:
    """L_col: channel-mean pair differences (gray-world term)."""
    means = output.mean(dim=(2, 3))
    return (
        (means[:, 0] - means[:, 1]).pow(2).mean()
        + (means[:, 0] - means[:, 2]).pow(2).mean()
        + (means[:, 1] - means[:, 2]).pow(2).mean()
    )


def illumination_smoothness_loss(alpha: torch.Tensor) -> torch.Tensor:
    """L_tvA: total variation of the predicted curve map, squared."""
    if min(alpha.shape[-2:]) < 2:
        return alpha.new_zeros(())
    dx = alpha[:, :, :, 1:] - alpha[:, :, :, :-1]
    dy = alpha[:, :, 1:, :] - alpha[:, :, :-1, :]
    return (dx.abs().mean() + dy.abs().mean()).pow(2)


def highlight_loss(output: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """L_hl: smoothstep-masked penalty for increasing already-bright pixels."""
    source_y = _luminance(source)
    output_y = _luminance(output)
    mask = _smoothstep(source_y)
    return (mask * F.relu(output_y - source_y)).mean()


def chromaticity_loss(output: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """L_hue: penalise changes in RGB direction."""
    output_unit = output / output.norm(dim=1, keepdim=True).clamp_min(1e-6)
    source_unit = source / source.norm(dim=1, keepdim=True).clamp_min(1e-6)
    cosine = (output_unit * source_unit).sum(dim=1, keepdim=True)
    return (1.0 - cosine).mean()


def zero_dce_loss(
    output: torch.Tensor,
    alpha: torch.Tensor,
    source: torch.Tensor,
    weights: LossWeights,
) -> LossBreakdown:
    """Combine the documented terms; all values are differentiable tensors."""
    terms = {
        "spatial": spatial_consistency_loss(output, source),
        "exposure": exposure_loss(output, weights.exposure_target),
        "color": color_constancy_loss(output),
        "tv_alpha": illumination_smoothness_loss(alpha),
        "highlight": highlight_loss(output, source),
        "hue": chromaticity_loss(output, source),
    }
    total = (
        weights.spatial * terms["spatial"]
        + weights.exposure * terms["exposure"]
        + weights.color * terms["color"]
        + weights.tv_alpha * terms["tv_alpha"]
        + weights.highlight * terms["highlight"]
        + weights.hue * terms["hue"]
    )
    return LossBreakdown(total=total, terms=terms)
