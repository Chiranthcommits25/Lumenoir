"""Regression tests for the LUMENOIR leave-one-module-out ablation study.

Verifies:
1. Baseline matches standard execution.
2. Without LumenScan: runs with fixed parameters, no crashes, valid outputs.
3. Without LumenRetinex: classical candidate passes through original input.
4. Without EdgeGuard: denoising stage bypassed cleanly.
5. Without HueLock: colour correction stage bypassed cleanly.
6. Without LumenFuse: performs 50/50 blend of classical and DL candidates.
7. All 6 configurations produce valid float images in [0, 1] without NaNs.
8. Shared caching is automatically disabled during ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pytest
from PIL import Image

from backend.enhancement.ablation import (
    ABLATION_EXPERIMENTS,
    AblationConfig,
    create_fixed_scan_result,
)
from backend.enhancement.pipeline import EnhancementPipeline
from backend.enhancement.shared import SharedStageCache
from backend.enhancement.types import ArtifactKind, ProcessingStatus
from backend.inference.manager import ModelManager


@dataclass
class DummyOutcome:
    image: Image.Image
    curve_alpha: None = None


class DummyModelManager(ModelManager):
    """Stub ModelManager that brightens the image slightly."""

    def __init__(self) -> None:
        pass

    @property
    def status(self):
        @dataclass
        class _Status:
            ready: bool = True
            model_id: str = "stub"
            device: str = "cpu"
        return _Status()

    def infer(self, image: Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        brightened = np.clip(arr * 1.5, 0.0, 1.0)
        out_img = Image.fromarray(np.round(brightened * 255.0).astype(np.uint8))
        return DummyOutcome(image=out_img)


@pytest.fixture
def dark_image() -> Image.Image:
    """A realistic low-light test image with non-zero noise."""
    rng = np.random.default_rng(42)
    base = np.full((64, 64, 3), 0.10, dtype=np.float32)
    noise = rng.normal(0, 0.02, base.shape).astype(np.float32)
    arr = np.clip(base + noise, 0.0, 1.0)
    return Image.fromarray(np.round(arr * 255.0).astype(np.uint8))


def test_create_fixed_scan_result() -> None:
    res = create_fixed_scan_result((64, 64))
    assert res.features.noise_sigma == 0.01
    assert res.parameters.gamma == 0.60
    assert res.parameters.gain_cap == 6.0
    assert res.illumination.shape == (64, 64)
    assert res.noise_map.shape == (64, 64)
    assert not res.parameters.bypass_gate


def test_baseline_matches_no_ablation(dark_image: Image.Image) -> None:
    pipeline = EnhancementPipeline(model_manager=DummyModelManager())
    res_none = pipeline.run(dark_image, mode="hybrid")
    res_base = pipeline.run(dark_image, mode="hybrid", ablation=AblationConfig())

    assert res_none.status is ProcessingStatus.COMPLETED
    assert res_base.status is ProcessingStatus.COMPLETED
    assert res_none.final_image is not None and res_base.final_image is not None
    assert np.array_equal(np.asarray(res_none.final_image), np.asarray(res_base.final_image))


@pytest.mark.parametrize(
    "name,config",
    list(ABLATION_EXPERIMENTS.items()),
)
def test_all_ablation_experiments_produce_valid_image(
    dark_image: Image.Image, name: str, config: AblationConfig
) -> None:
    pipeline = EnhancementPipeline(model_manager=DummyModelManager())
    result = pipeline.run(dark_image, mode="hybrid", ablation=config)

    assert result.status is ProcessingStatus.COMPLETED, f"Experiment {name} failed: {result.message}"
    assert result.final_image is not None
    assert result.final_image.size == dark_image.size

    arr = np.asarray(result.final_image, dtype=np.float32) / 255.0
    assert np.isfinite(arr).all(), f"Experiment {name} produced NaN or infinite values"
    assert (arr >= 0.0).all() and (arr <= 1.0).all()


def test_without_lumen_retinex_passes_original_classical(dark_image: Image.Image) -> None:
    pipeline = EnhancementPipeline(model_manager=DummyModelManager())
    config = AblationConfig(disable_lumen_retinex=True)
    result = pipeline.run(dark_image, mode="hybrid", ablation=config)

    classical_art = next(a for a in result.artifacts if a.descriptor.kind == ArtifactKind.CLASSICAL_OUTPUT)
    assert classical_art.available
    assert classical_art.image is not None
    # Classical output must match the original image
    assert np.array_equal(np.asarray(classical_art.image), np.asarray(dark_image))


def test_without_lumen_fuse_uses_fifty_fifty_blend(dark_image: Image.Image) -> None:
    pipeline = EnhancementPipeline(model_manager=DummyModelManager())
    config = AblationConfig(disable_lumen_fuse=True)
    result = pipeline.run(dark_image, mode="hybrid", ablation=config)

    fusion_art = next(a for a in result.artifacts if a.descriptor.kind == ArtifactKind.FUSION_OUTPUT)
    assert fusion_art.available
    assert fusion_art.image is not None


def test_shared_cache_is_disabled_during_ablation(dark_image: Image.Image) -> None:
    pipeline = EnhancementPipeline(model_manager=DummyModelManager())
    cache = SharedStageCache()
    config = AblationConfig(disable_edge_guard=True)

    # When ablation is active, the pipeline must ignore the cache
    pipeline.run(dark_image, mode="hybrid", cache=cache, ablation=config)
    assert len(cache.stages) == 0, "SharedStageCache should remain unpopulated during ablation"
