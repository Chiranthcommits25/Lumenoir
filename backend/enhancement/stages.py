"""Pipeline stages.

Each class is a thin orchestration layer over an existing module:

- Analysis → LumenScan
- Classical CV → LumenRetinex
- Deep learning → the registered Zero-DCE++ model through ModelManager
- Fusion → LumenFuse
- Denoising → EdgeGuard
- Colour correction → HueLock
- Quality assessment → the adaptive alpha blend with the original image

The stage classes contain no algorithm logic of their own. ``PendingStage``
remains available for custom/partial pipelines, but the default pipeline uses
the implemented stages only.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from ..classical_cv.edge_guard import denoise as edge_guard_denoise
from ..classical_cv.hue_lock import correct as hue_lock_correct
from ..classical_cv.retinex import enhance as retinex_enhance
from ..classical_cv.scan import LumenScanResult, scan
from ..fusion.lumen_fuse import fuse as fuse_candidates
from ..inference.errors import InferenceFailedError, ModelNotAvailableError
from ..inference.manager import ModelManager
from .types import (
    ArtifactKind,
    EnhancementMode,
    StageName,
    StageStatus,
)


class StageNotImplemented(NotImplementedError):
    """Raised by a stage that has no implementation yet."""


@dataclass
class StageContext:
    """Mutable state passed along the pipeline for a single image."""

    image: Image.Image
    mode: EnhancementMode
    original_size: tuple[int, int]
    outputs: dict[ArtifactKind, Image.Image] = field(default_factory=dict)

    rgb: np.ndarray | None = None
    scan_result: LumenScanResult | None = None
    bypass: bool = False
    classical: np.ndarray | None = None
    deep_learning: np.ndarray | None = None
    fused: np.ndarray | None = None
    enhanced: np.ndarray | None = None
    model_manager: ModelManager | None = None
    capture_diagnostics: bool = False
    diagnostics: dict[str, object] = field(default_factory=dict)
    diagnostic_images: dict[str, Image.Image] = field(default_factory=dict)

    def publish(self, kind: ArtifactKind, image: Image.Image) -> None:
        """Record an intermediate result so it can be reported to the client."""
        self.outputs[kind] = image


class PipelineStage(ABC):
    """One step of the enhancement pipeline."""

    name: StageName
    description: str
    produces: tuple[ArtifactKind, ...] = ()

    @property
    @abstractmethod
    def status(self) -> StageStatus:
        """Whether this stage can run in the current build."""

    @abstractmethod
    def run(self, context: StageContext) -> StageContext:
        """Process the context and return it (stages may publish artifacts)."""


class ImplementedStage(PipelineStage):
    """Base class for a stage backed by a real module."""

    @property
    def status(self) -> StageStatus:
        return StageStatus.IMPLEMENTED


class PendingStage(PipelineStage):
    """A documented stage whose algorithm has not been decided yet."""

    def __init__(
        self,
        name: StageName,
        description: str,
        produces: tuple[ArtifactKind, ...] = (),
        pending_reason: str = "",
    ) -> None:
        self.name = name
        self.description = description
        self.produces = produces
        self.pending_reason = pending_reason or (
            "This stage is not implemented: its algorithm is pending research."
        )

    @property
    def status(self) -> StageStatus:
        return StageStatus.NOT_IMPLEMENTED

    def run(self, context: StageContext) -> StageContext:
        raise StageNotImplemented(self.pending_reason)


def _to_array(image: Image.Image, *, name: str = "image") -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if array.ndim != 3 or array.shape[2] != 3:
        raise InferenceFailedError(f"{name} is not an RGB image.")
    return np.ascontiguousarray(array)


def _to_pil(array: np.ndarray, *, name: str = "output") -> Image.Image:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3:
        raise RuntimeError(f"{name} is not HxWx3, got {values.shape}.")
    if not np.isfinite(values).all():
        raise RuntimeError(f"{name} contains NaN or infinite values.")
    encoded = np.clip(np.round(values * 255.0), 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(encoded)


def _map_to_pil(array: np.ndarray, *, vmax: float, name: str) -> Image.Image:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 2:
        raise RuntimeError(f"{name} is not a 2-D map, got {values.shape}.")
    if not np.isfinite(values).all():
        raise RuntimeError(f"{name} contains NaN or infinite values.")
    scale = float(vmax) if vmax > 0.0 else 1.0
    encoded = np.clip(values / scale, 0.0, 1.0)
    return Image.fromarray(np.round(encoded * 255.0).astype(np.uint8))


def _require(value: object, name: str) -> None:
    if value is None:
        raise RuntimeError(f"Pipeline stage dependency '{name}' is missing.")


class AnalysisStage(ImplementedStage):
    name = StageName.ANALYSIS
    description = (
        "Measure illumination, dynamic range, colour cast and noise with LumenScan."
    )
    produces = (
        ArtifactKind.ORIGINAL,
        ArtifactKind.ILLUMINATION_MAP,
        ArtifactKind.NOISE_MAP,
    )

    def run(self, context: StageContext) -> StageContext:
        context.rgb = _to_array(context.image)
        context.scan_result = scan(context.rgb)
        context.bypass = bool(context.scan_result.parameters.bypass_gate)
        if context.capture_diagnostics:
            context.diagnostics["scan"] = context.scan_result.as_dict()

        context.publish(ArtifactKind.ORIGINAL, context.image.copy())
        context.publish(
            ArtifactKind.ILLUMINATION_MAP,
            _map_to_pil(context.scan_result.illumination, vmax=1.0, name="illumination"),
        )
        noise = context.scan_result.noise_map
        context.publish(
            ArtifactKind.NOISE_MAP,
            _map_to_pil(noise, vmax=max(float(noise.max()), 1e-6), name="noise map"),
        )
        return context


class ClassicalStage(ImplementedStage):
    name = StageName.CLASSICAL_CV
    description = "LumenRetinex adaptive illumination correction, CLAHE and shoulder."
    produces = (ArtifactKind.CLASSICAL_OUTPUT,)

    def run(self, context: StageContext) -> StageContext:
        _require(context.rgb, "rgb")
        _require(context.scan_result, "scan_result")

        if context.bypass:
            context.classical = context.rgb
            context.enhanced = context.rgb
            return context

        result = retinex_enhance(context.rgb, scan_result=context.scan_result)
        context.classical = result.image
        context.enhanced = result.image
        context.publish(ArtifactKind.CLASSICAL_OUTPUT, _to_pil(result.image))
        return context


class DeepLearningStage(ImplementedStage):
    name = StageName.DEEP_LEARNING
    description = "Zero-DCE++ inference through the load-once ModelManager."
    produces = (ArtifactKind.DEEP_LEARNING_OUTPUT,)

    def run(self, context: StageContext) -> StageContext:
        _require(context.rgb, "rgb")

        if context.bypass:
            context.deep_learning = context.rgb
            context.enhanced = context.rgb
            return context

        manager = context.model_manager
        if manager is None:
            raise ModelNotAvailableError(
                "No inference model manager is configured for this pipeline."
            )

        outcome = manager.infer(context.image)
        try:
            array = _to_array(outcome.image, name="model output")
        finally:
            outcome.image.close()

        if array.shape != context.rgb.shape:
            raise InferenceFailedError(
                f"The model returned {array.shape}, expected {context.rgb.shape}."
            )

        context.deep_learning = array
        context.enhanced = array
        context.publish(ArtifactKind.DEEP_LEARNING_OUTPUT, _to_pil(array))
        return context


class FusionStage(ImplementedStage):
    name = StageName.FUSION
    description = "LumenFuse noise-aware Laplacian-pyramid fusion of I0, C and D."
    produces = (
        ArtifactKind.FUSION_OUTPUT,
        ArtifactKind.ENHANCEMENT_STRENGTH_MAP,
    )

    def run(self, context: StageContext) -> StageContext:
        _require(context.rgb, "rgb")
        _require(context.scan_result, "scan_result")
        _require(context.classical, "classical")
        _require(context.deep_learning, "deep_learning")

        if context.bypass:
            context.fused = context.rgb
            context.enhanced = context.rgb
            return context

        result = fuse_candidates(
            context.rgb,
            context.classical,
            context.deep_learning,
            scan_result=context.scan_result,
            backend=os.getenv("LUMENOIR_FUSE_BACKEND", "auto"),
            device=os.getenv("LUMENOIR_FUSE_DEVICE") or None,
            precision=os.getenv("LUMENOIR_FUSE_PRECISION", "float32"),
        )
        context.fused = result.image
        context.enhanced = result.image
        if context.capture_diagnostics:
            context.diagnostics["fusion"] = result.as_dict()
            context.diagnostics["fusion_weights"] = result.weights.copy()
            if result.initial_weights is not None:
                context.diagnostics["fusion_weights_initial"] = (
                    result.initial_weights.copy()
                )
        context.publish(ArtifactKind.FUSION_OUTPUT, _to_pil(result.image))
        strength = 1.0 - result.weights[..., 0]
        context.publish(
            ArtifactKind.ENHANCEMENT_STRENGTH_MAP,
            _map_to_pil(strength, vmax=1.0, name="enhancement strength"),
        )
        return context


class DenoisingStage(ImplementedStage):
    name = StageName.DENOISING
    description = "EdgeGuard gain-aware luma guided filter and chroma bilateral filter."

    def run(self, context: StageContext) -> StageContext:
        _require(context.rgb, "rgb")
        _require(context.scan_result, "scan_result")
        _require(context.enhanced, "enhanced")

        if context.bypass:
            context.enhanced = context.rgb
            return context

        result = edge_guard_denoise(
            context.enhanced,
            context.rgb,
            scan_result=context.scan_result,
        )
        context.enhanced = result.image
        if context.capture_diagnostics:
            context.diagnostics["edge_guard"] = result.as_dict()
            context.diagnostic_images["edge_guard"] = _to_pil(result.image)
        return context


class ColourCorrectionStage(ImplementedStage):
    name = StageName.COLOUR_CORRECTION
    description = "HueLock guarded LAB colour-cast correction."

    def run(self, context: StageContext) -> StageContext:
        _require(context.scan_result, "scan_result")
        _require(context.enhanced, "enhanced")

        if context.bypass:
            return context

        result = hue_lock_correct(
            context.enhanced, scan_result=context.scan_result
        )
        context.enhanced = result.image
        if context.capture_diagnostics:
            context.diagnostics["hue_lock"] = result.as_dict()
            context.diagnostic_images["hue_lock"] = _to_pil(result.image)
        return context


class QualityAssessmentStage(ImplementedStage):
    name = StageName.QUALITY_ASSESSMENT
    description = (
        "Adaptive alpha blend with the original image and final output validation."
    )
    produces = (ArtifactKind.FINAL_OUTPUT,)

    def run(self, context: StageContext) -> StageContext:
        _require(context.rgb, "rgb")
        _require(context.scan_result, "scan_result")
        _require(context.enhanced, "enhanced")

        if context.bypass:
            final_image = context.image.copy()
            if context.capture_diagnostics:
                context.diagnostics["output_blend_alpha"] = 0.0
        else:
            alpha = float(context.scan_result.parameters.output_blend_alpha)
            if context.capture_diagnostics:
                context.diagnostics["output_blend_alpha"] = alpha
            if alpha <= 0.0:
                final_image = context.image.copy()
            else:
                blended = (1.0 - alpha) * context.rgb + alpha * context.enhanced
                blended = np.ascontiguousarray(
                    np.clip(blended, 0.0, 1.0).astype(np.float32)
                )
                if blended.shape != context.rgb.shape:
                    raise RuntimeError(
                        f"Final blend shape {blended.shape} != {context.rgb.shape}."
                    )
                final_image = _to_pil(blended, name="final output")

        if final_image.size != context.original_size:
            final_image.close()
            raise RuntimeError(
                f"Final image size {final_image.size} != {context.original_size}."
            )

        context.publish(ArtifactKind.FINAL_OUTPUT, final_image)
        return context


def default_stages() -> tuple[PipelineStage, ...]:
    """The ordered stage set used by the production pipeline."""
    return (
        AnalysisStage(),
        ClassicalStage(),
        DeepLearningStage(),
        FusionStage(),
        DenoisingStage(),
        ColourCorrectionStage(),
        QualityAssessmentStage(),
    )
