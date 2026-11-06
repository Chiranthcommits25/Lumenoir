"""Enhancement pipeline orchestration.

The orchestrator runs the implemented stages in order and reports per-stage
timings, artifacts and failures. The stage classes delegate to the existing
LumenScan, LumenRetinex, Zero-DCE++, LumenFuse, EdgeGuard and HueLock modules;
no algorithm logic lives here.

The pipeline is stateless. Slow CPU work is executed by the API in a thread
pool; the Zero-DCE++ ModelManager serialises model inference internally.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image

from ..config import Settings
from ..inference.errors import InferenceError
from ..inference.manager import ModelManager
from ..timing import Stopwatch
from .stages import PipelineStage, StageContext, StageNotImplemented, default_stages
from .types import (
    ENHANCEMENT_MODES,
    MODE_LABELS,
    PLANNED_ARTIFACTS,
    Artifact,
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

logger = logging.getLogger(__name__)

# Which stages each selectable mode needs. The remaining stages are still
# reported as pending so the status endpoint never hides missing work.
REQUIRED_STAGES_BY_MODE: dict[EnhancementMode, tuple[StageName, ...]] = {
    "classical": (
        StageName.ANALYSIS,
        StageName.CLASSICAL_CV,
        StageName.DENOISING,
        StageName.COLOUR_CORRECTION,
        StageName.QUALITY_ASSESSMENT,
    ),
    "deep_learning": (
        StageName.ANALYSIS,
        StageName.DEEP_LEARNING,
        StageName.DENOISING,
        StageName.COLOUR_CORRECTION,
        StageName.QUALITY_ASSESSMENT,
    ),
    "hybrid": (
        StageName.ANALYSIS,
        StageName.CLASSICAL_CV,
        StageName.DEEP_LEARNING,
        StageName.FUSION,
        StageName.DENOISING,
        StageName.COLOUR_CORRECTION,
        StageName.QUALITY_ASSESSMENT,
    ),
}


@dataclass(frozen=True)
class PipelineCapabilities:
    """Static description of the pipeline for the status endpoint."""

    stages: tuple[StageCapability, ...]
    modes: tuple[ModeCapability, ...]

    @property
    def ready(self) -> bool:
        return any(mode.available for mode in self.modes)


class EnhancementPipeline:
    """Drives the ordered stages and reports what is missing."""

    def __init__(
        self,
        *,
        stages: Sequence[PipelineStage] | None = None,
        model_manager: ModelManager | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._stages: tuple[PipelineStage, ...] = tuple(
            stages if stages is not None else default_stages()
        )
        self.model_manager = model_manager
        self.settings = settings

        self._by_name = {stage.name: stage for stage in self._stages}
        if len(self._by_name) != len(self._stages):
            raise ValueError("Pipeline stages must be unique.")

    # -- introspection -----------------------------------------------------

    @property
    def stages(self) -> tuple[PipelineStage, ...]:
        return self._stages

    def stages_for(self, mode: EnhancementMode) -> tuple[PipelineStage, ...]:
        """Return the stages required by ``mode`` in pipeline order."""
        required = set(REQUIRED_STAGES_BY_MODE[mode])
        return tuple(stage for stage in self._stages if stage.name in required)

    def capabilities(self) -> PipelineCapabilities:
        stage_caps = tuple(
            StageCapability(
                name=stage.name,
                status=stage.status,
                description=stage.description,
            )
            for stage in self._stages
        )

        model_ready = (
            True if self.model_manager is None else self.model_manager.status.ready
        )

        modes = tuple(
            ModeCapability(
                mode=mode,
                label=MODE_LABELS[mode],
                available=(
                    all(
                        stage.status is StageStatus.IMPLEMENTED
                        for stage in self.stages_for(mode)
                    )
                    and (
                        StageName.DEEP_LEARNING not in REQUIRED_STAGES_BY_MODE[mode]
                        or model_ready
                    )
                ),
                required_stages=REQUIRED_STAGES_BY_MODE[mode],
            )
            for mode in ENHANCEMENT_MODES
        )

        return PipelineCapabilities(stages=stage_caps, modes=modes)

    # -- execution ---------------------------------------------------------

    def run(
        self, image: Image.Image, *, mode: EnhancementMode = "hybrid"
    ) -> PipelineResult:
        """Run the stages required by ``mode`` and describe the outcome.

        A stage that raises :class:`StageNotImplemented` is recorded as skipped,
        not as an error: its absence is expected in the current build. Any other
        exception is a genuine failure and is reported as such.
        """
        if mode not in ENHANCEMENT_MODES:
            raise ValueError(f"Unsupported enhancement mode: {mode!r}.")

        watches = Stopwatch()
        context = StageContext(
            image=image,
            mode=mode,
            original_size=(image.width, image.height),
            model_manager=self.model_manager,
        )
        reports: list[StageReport] = []

        failed: str | None = None

        for stage in self.stages_for(mode):
            if stage.status is not StageStatus.IMPLEMENTED:
                reports.append(
                    StageReport(
                        name=stage.name,
                        status=stage.status,
                        detail=self._pending_reason(stage),
                    )
                )
                continue

            try:
                with watches.measure(stage.name.value):
                    stage.run(context)
            except StageNotImplemented as exc:
                reports.append(
                    StageReport(name=stage.name, status=stage.status, detail=str(exc))
                )
            except InferenceError:
                raise
            except Exception as exc:
                logger.exception("Pipeline stage %s failed", stage.name.value)
                failed = f"The '{stage.name.value}' stage failed: {exc}"
                reports.append(
                    StageReport(
                        name=stage.name,
                        status=stage.status,
                        duration_ms=watches.durations_ms.get(stage.name.value),
                        detail=failed,
                    )
                )
                break
            else:
                reports.append(
                    StageReport(
                        name=stage.name,
                        status=stage.status,
                        duration_ms=watches.durations_ms.get(stage.name.value),
                    )
                )

        artifacts = self._artifacts(context.outputs)
        final_image = context.outputs.get(ArtifactKind.FINAL_OUTPUT)

        status, message = self._outcome(reports, failed, mode)
        if status is ProcessingStatus.COMPLETED and final_image is None:
            status = ProcessingStatus.FAILED
            message = "The pipeline completed without producing a final image."

        return PipelineResult(
            status=status,
            mode=mode,
            stages=tuple(reports),
            artifacts=artifacts,
            timings_ms=dict(watches.durations_ms),
            total_ms=watches.total_ms,
            final_image=final_image,
            message=message,
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _pending_reason(stage: PipelineStage) -> str:
        return getattr(
            stage,
            "pending_reason",
            f"The '{stage.name.value}' stage is not implemented yet.",
        )

    def artifact_capabilities(self) -> tuple[Artifact, ...]:
        """Describe every planned artifact before a run (nothing produced)."""
        return self._artifacts({})

    def _artifacts(
        self, outputs: dict[ArtifactKind, Image.Image]
    ) -> tuple[Artifact, ...]:
        artifacts: list[Artifact] = []
        for descriptor in PLANNED_ARTIFACTS:
            image = outputs.get(descriptor.kind)
            if image is not None:
                artifacts.append(
                    Artifact(descriptor=descriptor, available=True, image=image)
                )
                continue

            producer = self._by_name.get(descriptor.produced_by)
            if producer is None:
                reason = "No stage produces this output yet."
            elif producer.status is StageStatus.IMPLEMENTED:
                reason = "This output was not produced for this image."
            else:
                reason = self._pending_reason(producer)
            artifacts.append(
                Artifact(descriptor=descriptor, available=False, reason=reason)
            )

        return tuple(artifacts)

    @staticmethod
    def _outcome(
        reports: list[StageReport], failed: str | None, mode: EnhancementMode
    ) -> tuple[ProcessingStatus, str]:
        if failed is not None:
            return ProcessingStatus.FAILED, failed

        pending = [
            report.name.value
            for report in reports
            if report.status is not StageStatus.IMPLEMENTED
        ]
        if pending:
            message = (
                "Enhancement is not available yet. Pending stages for "
                f"'{MODE_LABELS[mode]}': {', '.join(pending)}. "
                "No output was generated."
            )
            return ProcessingStatus.UNAVAILABLE, message

        return ProcessingStatus.COMPLETED, "Enhancement completed."

