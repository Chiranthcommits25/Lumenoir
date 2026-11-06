"""Enhancement pipeline package.

The default pipeline has seven implemented stages that delegate to the existing
LumenScan, LumenRetinex, Zero-DCE++, LumenFuse, EdgeGuard and HueLock modules.
``PendingStage`` remains available for custom partial pipelines.
"""

from __future__ import annotations

from .pipeline import REQUIRED_STAGES_BY_MODE, EnhancementPipeline, PipelineCapabilities
from .shared import SHARED_STAGES, SharedStageCache
from .stages import (
    PendingStage,
    PipelineStage,
    StageContext,
    StageNotImplemented,
    default_stages,
)
from .types import (
    ENHANCEMENT_MODES,
    MODE_LABELS,
    PLANNED_ARTIFACTS,
    Artifact,
    ArtifactDescriptor,
    ArtifactKind,
    EnhancementMode,
    ModeCapability,
    PipelineResult,
    ProcessingStatus,
    StageCapability,
    StageName,
    StageReport,
    StageStatus,
)

__all__ = [
    "ENHANCEMENT_MODES",
    "MODE_LABELS",
    "PLANNED_ARTIFACTS",
    "REQUIRED_STAGES_BY_MODE",
    "SHARED_STAGES",
    "Artifact",
    "ArtifactDescriptor",
    "ArtifactKind",
    "EnhancementMode",
    "EnhancementPipeline",
    "ModeCapability",
    "PendingStage",
    "PipelineCapabilities",
    "PipelineResult",
    "PipelineStage",
    "ProcessingStatus",
    "SharedStageCache",
    "StageCapability",
    "StageContext",
    "StageName",
    "StageNotImplemented",
    "StageReport",
    "StageStatus",
    "default_stages",
]
