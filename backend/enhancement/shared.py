"""Per-image reuse of the mode-independent pipeline prefix stages.

The analysis, classical and deep-learning stages depend only on the input image
and the loaded model, never on the selected mode. When several modes are
evaluated for the same image, their results are identical, so they can be
measured once and replayed. Fusion is not cached because only the hybrid mode
uses it; it still consumes the reused classical and deep-learning candidates.
This module stores results for exactly one image and contains no algorithm
logic of its own.

A cache instance must be used for one image only. The caller owns its lifetime
and should call :meth:`SharedStageCache.close` when the image is finished.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

from .stages import StageContext
from .types import ArtifactKind, StageName

SHARED_STAGES: frozenset[StageName] = frozenset(
    {
        StageName.ANALYSIS,
        StageName.CLASSICAL_CV,
        StageName.DEEP_LEARNING,
    }
)
"""Prefix stages reused by more than one mode for the same image."""

_STAGE_STATE_FIELDS: dict[StageName, tuple[str, ...]] = {
    StageName.ANALYSIS: ("rgb", "scan_result", "bypass"),
    StageName.CLASSICAL_CV: ("classical", "enhanced"),
    StageName.DEEP_LEARNING: ("deep_learning", "enhanced"),
}
"""Context attributes each shared stage owns.

Only these are replayed, so a later stage's cached snapshot cannot clobber a
candidate produced by an earlier stage (for example ``classical`` when the
deep-learning stage ran in a mode that did not need the classical stage).
"""


@dataclass(frozen=True)
class CachedStage:
    """One shared stage's result, ready to replay into a new context."""

    duration_ms: float
    state: dict[str, object]
    outputs: dict[ArtifactKind, Image.Image]
    diagnostics: dict[str, object]
    diagnostic_images: dict[str, Image.Image]
    error: BaseException | None = None


@dataclass
class SharedStageCache:
    """Memoised prefix stage results for a single input image."""

    stages: dict[StageName, CachedStage] = field(default_factory=dict)

    def get(self, name: StageName) -> CachedStage | None:
        return self.stages.get(name)

    def record(
        self,
        name: StageName,
        *,
        duration_ms: float,
        context: StageContext,
        produced: tuple[ArtifactKind, ...],
    ) -> None:
        """Store a successful stage result.

        Images are copied so a caller releasing the first mode's
        :class:`~backend.enhancement.types.PipelineResult` cannot invalidate the
        cache used by the remaining modes.
        """
        self.stages[name] = CachedStage(
            duration_ms=duration_ms,
            state={
                field_name: getattr(context, field_name)
                for field_name in _STAGE_STATE_FIELDS[name]
            },
            outputs={
                kind: image.copy()
                for kind, image in context.outputs.items()
                if kind in produced
            },
            diagnostics=dict(context.diagnostics) if context.capture_diagnostics else {},
            diagnostic_images=(
                {key: image.copy() for key, image in context.diagnostic_images.items()}
                if context.capture_diagnostics
                else {}
            ),
        )

    def record_error(
        self, name: StageName, *, duration_ms: float, error: BaseException
    ) -> None:
        """Store a stage failure so later modes replay it without recomputing."""
        self.stages[name] = CachedStage(
            duration_ms=duration_ms,
            state={},
            outputs={},
            diagnostics={},
            diagnostic_images={},
            error=error,
        )

    def restore(self, name: StageName, context: StageContext) -> CachedStage:
        """Copy a cached result into ``context`` and return its entry."""
        cached = self.stages[name]
        for field_name, value in cached.state.items():
            setattr(context, field_name, value)
        for kind, image in cached.outputs.items():
            context.outputs[kind] = image.copy()
        if cached.diagnostics:
            context.diagnostics.update(cached.diagnostics)
        for key, image in cached.diagnostic_images.items():
            context.diagnostic_images[key] = image.copy()
        return cached

    def close(self) -> None:
        """Close cached images and drop every retained array."""
        for cached in self.stages.values():
            for image in cached.outputs.values():
                image.close()
            for image in cached.diagnostic_images.values():
                image.close()
        self.stages.clear()
