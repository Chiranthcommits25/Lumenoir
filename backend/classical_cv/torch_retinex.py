"""GPU LumenRetinex.

The algorithm is identical to :mod:`backend.classical_cv.retinex`; only the
array operations are tensor operations. The CLAHE LUT path mirrors OpenCV's
``clahe.cpp``, and RGB→LAB uses the OpenCV 5 33³ trilinear LUT so CLAHE bins do
not drift.
"""

from __future__ import annotations

from dataclasses import dataclass

from .retinex import (
    LAB_MAX,
    UINT16_MAX,
    LumenRetinexConfig,
)
from .scan import LumenParameters, LumenScanResult, scan


@dataclass(frozen=True)
class TorchRetinexResult:
    image: object
    gain: object | None
    illumination: object | None
    bypassed: bool
    parameters: LumenParameters


def compute_gain_tensor(
    rgb,
    illumination,
    parameters: LumenParameters,
    config: LumenRetinexConfig | None = None,
    *,
    ops,
):
    config = config or LumenRetinexConfig()
    torch = ops.torch
    if illumination.shape != rgb.shape[:2]:
        raise ValueError(
            f"illumination shape {tuple(illumination.shape)} must match the "
            f"image {tuple(rgb.shape[:2])}."
        )
    if not 0.0 < parameters.gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {parameters.gamma!r}.")
    if parameters.gain_cap < 1.0:
        raise ValueError(f"gain_cap must be >= 1, got {parameters.gain_cap!r}.")
    if config.highlight_high <= config.highlight_low:
        raise ValueError("highlight_high must be greater than highlight_low.")
    if config.gain_epsilon <= 0.0:
        raise ValueError("gain_epsilon must be positive.")

    raw_gain = torch.pow(illumination, float(parameters.gamma - 1.0))
    gain = torch.clamp(raw_gain, min=1.0, max=float(parameters.gain_cap))

    max_channel = torch.amax(rgb, dim=2)
    shoulder = 1.0 - ops.smoothstep(
        float(config.highlight_low), float(config.highlight_high), max_channel
    )
    attenuated = 1.0 + (gain - 1.0) * shoulder
    limited = torch.minimum(
        attenuated, 1.0 / (max_channel + float(config.gain_epsilon))
    )
    return limited.to(rgb.dtype)


def _shoulder_tensor(values, config: LumenRetinexConfig, *, ops):
    if config.shoulder_width <= 0.0:
        raise ValueError("shoulder_width must be positive.")
    return ops.highlight_shoulder(
        values,
        start=float(config.shoulder_start),
        width=float(config.shoulder_width),
    )


def _clahe_tensor(image, parameters: LumenParameters, *, ops):
    torch = ops.torch
    if parameters.clahe_clip <= 0.0:
        raise ValueError(f"clahe_clip must be positive, got {parameters.clahe_clip!r}.")

    lab = ops.rgb_to_lab(image)
    lightness = lab[..., 0].to(torch.float32)
    lightness_u16 = (
        torch.round(lightness * (UINT16_MAX / LAB_MAX))
        .clamp(0.0, UINT16_MAX)
        .to(torch.uint16)
        .to(torch.int64)
    )
    equalised = ops.clahe_uint16(
        lightness_u16,
        float(parameters.clahe_clip),
        tuple(int(value) for value in parameters.clahe_tiles),
    ).to(torch.float32) * (LAB_MAX / UINT16_MAX)

    blend = float(parameters.clahe_blend)
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"clahe_blend must be in [0, 1], got {blend!r}.")
    blended = (1.0 - blend) * lightness + blend * equalised
    parts = [
        blended[..., None],
        lab[..., 1:2].to(torch.float32),
        lab[..., 2:3].to(torch.float32),
    ]
    return ops.lab_to_rgb(torch.cat(parts, dim=2)).to(image.dtype)


def enhance_tensor(
    image,
    *,
    scan_result: LumenScanResult | None = None,
    config: LumenRetinexConfig | None = None,
    illumination=None,
    ops,
) -> TorchRetinexResult:
    """Run LumenRetinex on a float RGB tensor."""
    config = config or LumenRetinexConfig()
    torch = ops.torch
    rgb = image.to(ops.dtype)
    result = scan_result or scan(ops.to_numpy(rgb), None)

    if config.respect_bypass_gate and result.parameters.bypass_gate:
        return TorchRetinexResult(
            image=rgb,
            gain=None,
            illumination=None,
            bypassed=True,
            parameters=result.parameters,
        )

    parameters = result.parameters
    if illumination is None:
        illum = ops.estimate_illumination(
            rgb,
            eps=parameters.guided_eps,
            downsample=config.illumination_downsample,
        )
    else:
        illum = illumination.to(device=rgb.device, dtype=rgb.dtype)
        if illum.shape != rgb.shape[:2]:
            raise ValueError(
                f"illumination shape {tuple(illum.shape)} must match the "
                f"image {tuple(rgb.shape[:2])}."
            )

    gain = compute_gain_tensor(rgb, illum, parameters, config, ops=ops)
    corrected = torch.clamp(rgb * gain[..., None], min=0.0, max=1.0)

    if config.apply_clahe:
        corrected = _clahe_tensor(corrected, parameters, ops=ops)
    if config.apply_shoulder:
        corrected = _shoulder_tensor(corrected, config, ops=ops)
    corrected = torch.clamp(corrected, min=0.0, max=1.0).to(rgb.dtype)

    if corrected.shape != rgb.shape:
        raise RuntimeError(
            f"LumenRetinex changed the image shape: {tuple(rgb.shape)} -> "
            f"{tuple(corrected.shape)}."
        )
    return TorchRetinexResult(
        image=corrected,
        gain=gain,
        illumination=illum,
        bypassed=False,
        parameters=parameters,
    )


__all__ = [
    "TorchRetinexResult",
    "compute_gain_tensor",
    "enhance_tensor",
]
