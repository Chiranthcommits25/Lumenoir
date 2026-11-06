"""Pipeline vocabulary: modes, stages, artifacts and result shapes.

This module is architecture, not algorithm code. It names the parts of the
future enhancement pipeline so the API, the frontend and the tests can be
written against a stable contract while the research decisions are still open.

Nothing here selects a method, a weight or a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from PIL import Image

EnhancementMode = Literal["classical", "deep_learning", "hybrid"]

ENHANCEMENT_MODES: tuple[EnhancementMode, ...] = (
    "classical",
    "deep_learning",
    "hybrid",
)

MODE_LABELS: dict[str, str] = {
    "classical": "Classical CV",
    "deep_learning": "Deep learning",
    "hybrid": "Hybrid",
}


class StageName(StrEnum):
    """Coarse pipeline stages. Internal algorithm choices stay inside a stage."""

    ANALYSIS = "analysis"
    CLASSICAL_CV = "classical_cv"
    DEEP_LEARNING = "deep_learning"
    FUSION = "fusion"
    DENOISING = "denoising"
    COLOUR_CORRECTION = "colour_correction"
    QUALITY_ASSESSMENT = "quality_assessment"


class StageStatus(StrEnum):
    """Whether a stage can actually run today."""

    IMPLEMENTED = "implemented"
    NOT_IMPLEMENTED = "not_implemented"


class ProcessingStatus(StrEnum):
    """Outcome of a processing request (stateless; never persisted)."""

    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class ArtifactKind(StrEnum):
    """Visual outputs the explainable dashboard will display."""

    ORIGINAL = "original"
    ILLUMINATION_MAP = "illumination_map"
    NOISE_MAP = "noise_map"
    ENHANCEMENT_STRENGTH_MAP = "enhancement_strength_map"
    CLASSICAL_OUTPUT = "classical_output"
    DEEP_LEARNING_OUTPUT = "deep_learning_output"
    FUSION_OUTPUT = "fusion_output"
    FINAL_OUTPUT = "final_output"


@dataclass(frozen=True)
class ArtifactDescriptor:
    """Declares an intermediate image and the stage responsible for it."""

    kind: ArtifactKind
    label: str
    produced_by: StageName
    description: str


# The dashboard's visual set (Context §8). Each entry is reported as unavailable
# until the producing stage actually publishes it for the current request.
PLANNED_ARTIFACTS: tuple[ArtifactDescriptor, ...] = (
    ArtifactDescriptor(
        ArtifactKind.ORIGINAL,
        "Original",
        StageName.ANALYSIS,
        "The uploaded image as decoded, before any processing.",
    ),
    ArtifactDescriptor(
        ArtifactKind.ILLUMINATION_MAP,
        "Illumination map",
        StageName.ANALYSIS,
        "Estimated spatial illumination from LumenScan.",
    ),
    ArtifactDescriptor(
        ArtifactKind.NOISE_MAP,
        "Noise map",
        StageName.ANALYSIS,
        "Estimated local noise sigma from the Immerkaer estimator.",
    ),
    ArtifactDescriptor(
        ArtifactKind.ENHANCEMENT_STRENGTH_MAP,
        "Enhancement strength",
        StageName.FUSION,
        "Per-region contribution of the enhanced candidates in the LumenFuse weights.",
    ),
    ArtifactDescriptor(
        ArtifactKind.CLASSICAL_OUTPUT,
        "Classical CV output",
        StageName.CLASSICAL_CV,
        "LumenRetinex illumination correction, CLAHE and highlight shoulder.",
    ),
    ArtifactDescriptor(
        ArtifactKind.DEEP_LEARNING_OUTPUT,
        "Deep learning output",
        StageName.DEEP_LEARNING,
        "Zero-DCE++ curve-network output.",
    ),
    ArtifactDescriptor(
        ArtifactKind.FUSION_OUTPUT,
        "Fusion output",
        StageName.FUSION,
        "LumenFuse noise-aware pyramid blend of the original, classical and DL candidates.",
    ),
    ArtifactDescriptor(
        ArtifactKind.FINAL_OUTPUT,
        "Final enhanced image",
        StageName.QUALITY_ASSESSMENT,
        "Delivered result after EdgeGuard, HueLock and the adaptive alpha blend.",
    ),
)


@dataclass(frozen=True)
class StageCapability:
    """Public description of one stage's implementation status."""

    name: StageName
    status: StageStatus
    description: str


@dataclass(frozen=True)
class ModeCapability:
    """Whether a selectable mode can produce a result today."""

    mode: EnhancementMode
    label: str
    available: bool
    required_stages: tuple[StageName, ...]


@dataclass(frozen=True)
class StageReport:
    """Per-stage outcome for one request, including measured duration."""

    name: StageName
    status: StageStatus
    duration_ms: float | None = None
    detail: str | None = None


@dataclass(frozen=True)
class Artifact:
    """An intermediate output: either real pixels, or a documented absence."""

    descriptor: ArtifactDescriptor
    available: bool
    reason: str | None = None
    image: Image.Image | None = None


@dataclass(frozen=True)
class PipelineResult:
    """Everything a caller needs to explain what happened."""

    status: ProcessingStatus
    mode: EnhancementMode
    stages: tuple[StageReport, ...]
    artifacts: tuple[Artifact, ...]
    timings_ms: dict[str, float] = field(default_factory=dict)
    total_ms: float = 0.0
    final_image: Image.Image | None = None
    message: str = ""
    diagnostics: dict[str, object] = field(default_factory=dict)
    diagnostic_images: dict[str, Image.Image] = field(default_factory=dict)

    @property
    def completed(self) -> bool:
        return self.status is ProcessingStatus.COMPLETED and self.final_image is not None

    def release(self) -> None:
        """Close every image handle produced by this run (stateless requests)."""
        for artifact in self.artifacts:
            if artifact.image is not None:
                artifact.image.close()
        for image in self.diagnostic_images.values():
            image.close()
        if self.final_image is not None:
            self.final_image.close()
