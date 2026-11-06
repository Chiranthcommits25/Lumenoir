"""Pipeline orchestration: real stages, bypass, modes and failure handling."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from backend.enhancement.pipeline import REQUIRED_STAGES_BY_MODE, EnhancementPipeline
from backend.enhancement.stages import PendingStage, PipelineStage, StageContext
from backend.enhancement.types import (
    ArtifactKind,
    ProcessingStatus,
    StageName,
    StageStatus,
)
from backend.inference.errors import ModelNotAvailableError
from tests.conftest import StubModelManager, brighten


class _PassThroughStage(PipelineStage):
    """Implemented test stage: proves the orchestrator without any algorithm."""

    name = StageName.QUALITY_ASSESSMENT
    description = "Test double that forwards the input as the final output."
    produces = (ArtifactKind.FINAL_OUTPUT,)

    def __init__(self) -> None:
        self.calls = 0

    @property
    def status(self) -> StageStatus:
        return StageStatus.IMPLEMENTED

    def run(self, context: StageContext) -> StageContext:
        self.calls += 1
        context.publish(ArtifactKind.FINAL_OUTPUT, context.image)
        return context


class _ExplodingStage(_PassThroughStage):
    def run(self, context: StageContext) -> StageContext:
        raise RuntimeError("synthetic stage failure")


def _dark_image(size: tuple[int, int] = (96, 80), level: int = 10) -> Image.Image:
    return Image.new("RGB", size, (level, level, level))


def _artifacts(result) -> set[ArtifactKind]:
    return {
        artifact.descriptor.kind
        for artifact in result.artifacts
        if artifact.available
    }


def test_every_default_stage_is_implemented() -> None:
    pipeline = EnhancementPipeline(model_manager=StubModelManager())
    capabilities = pipeline.capabilities()

    assert len(capabilities.stages) == 7
    assert all(
        stage.status is StageStatus.IMPLEMENTED for stage in capabilities.stages
    )
    assert capabilities.ready is True


def test_modes_report_required_stages_and_availability() -> None:
    capabilities = EnhancementPipeline(
        model_manager=StubModelManager()
    ).capabilities()

    assert tuple(mode.mode for mode in capabilities.modes) == (
        "classical",
        "deep_learning",
        "hybrid",
    )
    assert all(mode.available is True for mode in capabilities.modes)

    hybrid = next(mode for mode in capabilities.modes if mode.mode == "hybrid")
    classical = next(mode for mode in capabilities.modes if mode.mode == "classical")

    assert hybrid.required_stages == REQUIRED_STAGES_BY_MODE["hybrid"]
    assert StageName.FUSION in hybrid.required_stages
    assert StageName.FUSION not in classical.required_stages


def test_modes_require_a_ready_model() -> None:
    capabilities = EnhancementPipeline(
        model_manager=StubModelManager(ready=False)
    ).capabilities()

    by_mode = {mode.mode: mode.available for mode in capabilities.modes}
    assert by_mode["classical"] is True
    assert by_mode["deep_learning"] is False
    assert by_mode["hybrid"] is False
    assert capabilities.ready is True


def test_planned_artifacts_are_unavailable_before_a_run() -> None:
    artifacts = EnhancementPipeline().artifact_capabilities()

    assert len(artifacts) == len(ArtifactKind)
    assert {artifact.descriptor.kind for artifact in artifacts} == set(ArtifactKind)
    assert all(artifact.available is False for artifact in artifacts)
    assert all(artifact.reason for artifact in artifacts)


def test_hybrid_pipeline_completes_with_real_artifacts() -> None:
    stub = StubModelManager(transform=lambda image: brighten(image, 2.5))
    pipeline = EnhancementPipeline(model_manager=stub)
    image = _dark_image()

    result = pipeline.run(image, mode="hybrid")

    try:
        assert result.status is ProcessingStatus.COMPLETED
        assert result.completed is True
        assert result.final_image is not None
        assert result.final_image.size == image.size
        assert result.final_image.mode == "RGB"
        assert stub.calls == 1
        assert len(result.stages) == len(REQUIRED_STAGES_BY_MODE["hybrid"])
        assert all(stage.status is StageStatus.IMPLEMENTED for stage in result.stages)
        assert set(result.timings_ms) == {
            stage.value for stage in REQUIRED_STAGES_BY_MODE["hybrid"]
        }
        assert result.total_ms >= 0.0
        assert _artifacts(result) == set(ArtifactKind)

        array = np.asarray(result.final_image, dtype=np.float32) / 255.0
        assert np.isfinite(array).all()
        assert array.min() >= 0.0
        assert array.max() <= 1.0
    finally:
        result.release()
        image.close()


def test_bypass_gate_skips_the_model_and_returns_the_input() -> None:
    ramp = np.tile(
        np.linspace(0.25, 0.85, 128, dtype=np.float32)[None, :, None], (128, 1, 3)
    )
    image = Image.fromarray(np.round(ramp * 255.0).astype(np.uint8))
    stub = StubModelManager(
        transform=lambda _image: (_ for _ in ()).throw(
            AssertionError("the model must not run when the bypass gate fires")
        )
    )
    pipeline = EnhancementPipeline(model_manager=stub)

    result = pipeline.run(image, mode="hybrid")

    try:
        assert result.status is ProcessingStatus.COMPLETED
        assert result.final_image is not None
        assert stub.calls == 0
        assert np.array_equal(np.asarray(result.final_image), np.asarray(image))
        assert _artifacts(result) == {
            ArtifactKind.ORIGINAL,
            ArtifactKind.ILLUMINATION_MAP,
            ArtifactKind.NOISE_MAP,
            ArtifactKind.FINAL_OUTPUT,
        }
    finally:
        result.release()
        image.close()


def test_classical_mode_does_not_require_a_ready_model() -> None:
    stub = StubModelManager(ready=False)
    pipeline = EnhancementPipeline(model_manager=stub)
    image = _dark_image()

    result = pipeline.run(image, mode="classical")

    try:
        assert result.status is ProcessingStatus.COMPLETED
        assert result.final_image is not None
        assert result.final_image.size == image.size
        assert stub.calls == 0
        assert ArtifactKind.CLASSICAL_OUTPUT in _artifacts(result)
        assert ArtifactKind.DEEP_LEARNING_OUTPUT not in _artifacts(result)
    finally:
        result.release()
        image.close()


def test_deep_learning_mode_uses_the_model() -> None:
    stub = StubModelManager(transform=lambda image: brighten(image, 2.0))
    pipeline = EnhancementPipeline(model_manager=stub)
    image = _dark_image()

    result = pipeline.run(image, mode="deep_learning")

    try:
        assert result.status is ProcessingStatus.COMPLETED
        assert result.final_image is not None
        assert stub.calls == 1
        assert ArtifactKind.DEEP_LEARNING_OUTPUT in _artifacts(result)
        assert ArtifactKind.CLASSICAL_OUTPUT not in _artifacts(result)
        assert ArtifactKind.FUSION_OUTPUT not in _artifacts(result)
    finally:
        result.release()
        image.close()


def test_hybrid_requires_a_ready_model() -> None:
    pipeline = EnhancementPipeline(model_manager=StubModelManager(ready=False))
    image = _dark_image()

    try:
        with pytest.raises(ModelNotAvailableError):
            pipeline.run(image, mode="hybrid")
    finally:
        image.close()


def test_dark_image_is_brightened() -> None:
    stub = StubModelManager(transform=lambda image: brighten(image, 3.0))
    pipeline = EnhancementPipeline(model_manager=stub)
    image = _dark_image((96, 80), level=10)

    result = pipeline.run(image, mode="hybrid")

    try:
        output = np.asarray(result.final_image, dtype=np.float32) / 255.0
        source = np.asarray(image, dtype=np.float32) / 255.0
        assert output.mean() > source.mean()
        assert result.final_image.size == image.size
    finally:
        result.release()
        image.close()


def test_nonuniform_image_is_processed() -> None:
    ramp = np.tile(
        np.linspace(0.02, 0.60, 128, dtype=np.float32)[None, :, None], (128, 1, 3)
    )
    image = Image.fromarray(np.round(ramp * 255.0).astype(np.uint8))
    pipeline = EnhancementPipeline(
        model_manager=StubModelManager(transform=lambda image: brighten(image, 2.0))
    )

    result = pipeline.run(image, mode="hybrid")

    try:
        output = np.asarray(result.final_image, dtype=np.float32) / 255.0
        source = np.asarray(image, dtype=np.float32) / 255.0
        assert result.final_image.size == image.size
        assert np.isfinite(output).all()
        assert output[:, :40].mean() > source[:, :40].mean()
        assert output.max() <= 1.0
    finally:
        result.release()
        image.close()


def test_noisy_image_is_processed() -> None:
    rng = np.random.default_rng(3)
    noisy = np.clip(
        0.04 + rng.normal(0.0, 0.03, (96, 80, 3)), 0.0, 1.0
    )
    image = Image.fromarray(np.round(noisy * 255.0).astype(np.uint8))
    pipeline = EnhancementPipeline(
        model_manager=StubModelManager(transform=lambda image: brighten(image, 2.5))
    )

    result = pipeline.run(image, mode="hybrid")

    try:
        output = np.asarray(result.final_image, dtype=np.float32) / 255.0
        assert result.final_image.size == image.size
        assert np.isfinite(output).all()
        assert output.min() >= 0.0
        assert output.max() <= 1.0
    finally:
        result.release()
        image.close()


def test_colour_distorted_image_is_processed() -> None:
    image = Image.new("RGB", (96, 80), (153, 128, 102))
    pipeline = EnhancementPipeline(
        model_manager=StubModelManager(transform=lambda image: brighten(image, 1.5))
    )

    result = pipeline.run(image, mode="hybrid")

    try:
        assert result.status is ProcessingStatus.COMPLETED
        assert result.final_image is not None
        assert result.final_image.size == image.size
        output = np.asarray(result.final_image, dtype=np.float32) / 255.0
        assert np.isfinite(output).all()
        assert output.min() >= 0.0
        assert output.max() <= 1.0
    finally:
        result.release()
        image.close()


def test_odd_dimensions_are_preserved() -> None:
    image = Image.new("RGB", (101, 77), (12, 14, 16))
    pipeline = EnhancementPipeline(
        model_manager=StubModelManager(transform=lambda image: brighten(image, 2.0))
    )

    result = pipeline.run(image, mode="hybrid")

    try:
        assert result.final_image.size == image.size
        assert result.final_image.mode == "RGB"
        output = np.asarray(result.final_image, dtype=np.float32) / 255.0
        assert output.shape == (77, 101, 3)
        assert output.min() >= 0.0
        assert output.max() <= 1.0
    finally:
        result.release()
        image.close()


def test_pipeline_does_not_mutate_the_input() -> None:
    image = _dark_image()
    before = np.asarray(image).copy()
    pipeline = EnhancementPipeline(
        model_manager=StubModelManager(transform=lambda image: brighten(image, 2.0))
    )

    result = pipeline.run(image, mode="hybrid")

    try:
        assert np.array_equal(np.asarray(image), before)
    finally:
        result.release()
        image.close()


def test_mode_selects_a_smaller_stage_set() -> None:
    pipeline = EnhancementPipeline()

    classical = pipeline.stages_for("classical")
    deep = pipeline.stages_for("deep_learning")

    assert len(classical) == len(REQUIRED_STAGES_BY_MODE["classical"])
    assert len(deep) == len(REQUIRED_STAGES_BY_MODE["deep_learning"])
    assert StageName.CLASSICAL_CV in {stage.name for stage in classical}
    assert StageName.CLASSICAL_CV not in {stage.name for stage in deep}


def test_unsupported_mode_rejected() -> None:
    image = Image.new("RGB", (16, 16))
    with pytest.raises(ValueError):
        EnhancementPipeline().run(image, mode="retinex_magic")  # type: ignore[arg-type]
    image.close()


def test_duplicate_stage_names_rejected() -> None:
    duplicated = (
        PendingStage(StageName.ANALYSIS, "first"),
        PendingStage(StageName.ANALYSIS, "second"),
    )

    with pytest.raises(ValueError):
        EnhancementPipeline(stages=duplicated)


def test_implemented_stage_produces_a_real_result() -> None:
    stage = _PassThroughStage()
    pipeline = EnhancementPipeline(stages=(stage,))
    image = Image.new("RGB", (40, 30), (5, 10, 15))

    result = pipeline.run(image, mode="hybrid")

    assert result.status is ProcessingStatus.COMPLETED
    assert result.completed is True
    assert result.final_image is image
    assert stage.calls == 1
    assert result.timings_ms["quality_assessment"] >= 0

    final = next(
        artifact
        for artifact in result.artifacts
        if artifact.descriptor.kind is ArtifactKind.FINAL_OUTPUT
    )
    assert final.available is True
    assert final.image is image

    image.close()


def test_failing_stage_is_reported_as_failure() -> None:
    pipeline = EnhancementPipeline(stages=(_ExplodingStage(),))
    image = Image.new("RGB", (16, 16))

    result = pipeline.run(image, mode="hybrid")

    assert result.status is ProcessingStatus.FAILED
    assert result.final_image is None
    assert "synthetic stage failure" in result.message
    assert result.stages[0].detail is not None

    image.close()
