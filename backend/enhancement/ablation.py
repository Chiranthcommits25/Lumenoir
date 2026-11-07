"""Ablation study configuration and fixed-parameter fallbacks.

This module provides the formal leave-one-module-out ablation configurations
for the LUMENOIR enhancement pipeline:

1. Baseline: all 5 modules enabled
2. Without LumenScan: uses fixed scan parameters (bypasses adaptive scan)
3. Without LumenRetinex: uses original input as classical candidate
4. Without EdgeGuard: bypasses gain-aware denoising
5. Without HueLock: bypasses CIELAB colour correction
6. Without LumenFuse: uses a simple 50/50 blend of classical and DL candidates

All switches operate cleanly at the pipeline orchestration layer. Algorithm
implementations, model weights and core parameters remain untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..classical_cv.scan import (
    DarknessClass,
    LumenParameters,
    LumenScanFeatures,
    LumenScanResult,
)

AblationName = Literal[
    "baseline",
    "without_lumen_scan",
    "without_lumen_retinex",
    "without_edge_guard",
    "without_hue_lock",
    "without_lumen_fuse",
]


@dataclass(frozen=True)
class AblationConfig:
    """Switches controlling which module in the pipeline is ablated.

    At most one module is disabled in a controlled leave-one-out study,
    though multiple flags can be combined for custom experiments.
    """

    disable_lumen_scan: bool = False
    disable_lumen_retinex: bool = False
    disable_edge_guard: bool = False
    disable_hue_lock: bool = False
    disable_lumen_fuse: bool = False

    @property
    def is_active(self) -> bool:
        """True if any ablation switch is enabled."""
        return (
            self.disable_lumen_scan
            or self.disable_lumen_retinex
            or self.disable_edge_guard
            or self.disable_hue_lock
            or self.disable_lumen_fuse
        )


# Canonical 6 experimental configurations
ABLATION_EXPERIMENTS: dict[AblationName, AblationConfig] = {
    "baseline": AblationConfig(),
    "without_lumen_scan": AblationConfig(disable_lumen_scan=True),
    "without_lumen_retinex": AblationConfig(disable_lumen_retinex=True),
    "without_edge_guard": AblationConfig(disable_edge_guard=True),
    "without_hue_lock": AblationConfig(disable_hue_lock=True),
    "without_lumen_fuse": AblationConfig(disable_lumen_fuse=True),
}

ABLATION_TITLES: dict[AblationName, str] = {
    "baseline": "Baseline (All Modules Enabled)",
    "without_lumen_scan": "Without LumenScan (Fixed Parameters)",
    "without_lumen_retinex": "Without LumenRetinex (Pass-through Classical)",
    "without_edge_guard": "Without EdgeGuard (No Denoising)",
    "without_hue_lock": "Without HueLock (No Colour Correction)",
    "without_lumen_fuse": "Without LumenFuse (50/50 Simple Blend)",
}


def create_fixed_scan_result(
    image_shape: tuple[int, int],
    *,
    device_ops: object | None = None,
) -> LumenScanResult:
    """Create a fixed, non-adaptive LumenScanResult for Ablation 1.

    When LumenScan is ablated, the pipeline cannot adaptively measure
    percentiles or noise. This supplies frozen representative parameters
    derived from the baseline configuration defaults.
    """
    height, width = image_shape[:2]

    # Global fixed features
    features = LumenScanFeatures(
        mean=0.30,
        p5=0.05,
        p50=0.25,
        p95=0.70,
        dynamic_range=0.65,
        nonuniformity=1.0,
        noise_sigma=0.01,
        colour_cast_a=0.0,
        colour_cast_b=0.0,
        colour_cast_magnitude=0.0,
        highlight_fraction=0.0,
        black_fraction=0.0,
        darkness=DarknessClass.DARK,
    )

    # Global fixed parameters (representative defaults without image adaptation)
    parameters = LumenParameters(
        gamma=0.60,
        gain_cap=6.0,
        guided_eps=1e-3,
        clahe_clip=2.0,
        clahe_tiles=(8, 8),
        clahe_blend=0.6,
        dl_strength=0.8,
        fusion_contrast=1.0,
        fusion_saturation=0.5,
        fusion_exposure=1.0,
        fusion_noise=1.0,
        fusion_tau=0.08,
        hue_lock_beta=0.0,
        output_blend_alpha=1.0,
        bypass_gate=False,
    )

    # Uniform baseline illumination map and noise map
    illumination_np = np.full((height, width), 0.35, dtype=np.float32)
    noise_map_np = np.full((height, width), 0.01, dtype=np.float32)

    if device_ops is not None:
        illumination = device_ops.ensure(illumination_np)
        noise_map = device_ops.ensure(noise_map_np)
    else:
        illumination = illumination_np
        noise_map = noise_map_np

    return LumenScanResult(
        features=features,
        parameters=parameters,
        illumination=illumination,
        noise_map=noise_map,
        analysis_size=(min(width, 256), min(height, 256)),
    )

