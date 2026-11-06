"""Shared prefix-stage reuse: call counts, output equivalence and isolation."""

from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import backend.enhancement.stages as stage_module
from backend.enhancement.pipeline import EnhancementPipeline
from backend.enhancement.shared import SharedStageCache
from backend.inference.errors import InferenceFailedError
from tests.conftest import StubModelManager, brighten

MODES = ("classical", "deep_learning", "hybrid")


def _scene(size: tuple[int, int] = (80, 64), seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0.02, 0.35, size[0], dtype=np.float32)[None, :, None]
    noise = rng.normal(0.0, 0.01, (size[1], size[0], 3)).astype(np.float32)
    array = np.clip(ramp + noise, 0.0, 1.0)
    return Image.fromarray(np.round(array * 255.0).astype(np.uint8))


def _as_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _run_modes(pipeline: EnhancementPipeline, image: Image.Image, cache=None):
    results = {}
    try:
        for mode in MODES:
            results[mode] = pipeline.run(image, mode=mode, cache=cache)  # type: ignore[arg-type]
        return {
            mode: _as_float(result.final_image)  # type: ignore[arg-type]
            for mode, result in results.items()
        }
    finally:
        for result in results.values():
            result.release()


class _FailingModel(StubModelManager):
    def infer(self, image: Image.Image):
        self.calls += 1
        raise InferenceFailedError("synthetic inference failure")


def _count(monkeypatch, name: str, counts: Counter) -> None:
    original = getattr(stage_module, name)

    def wrapper(*args, **kwargs):
        counts[name] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(stage_module, name, wrapper)


def test_prefix_stages_run_once_per_image(monkeypatch) -> None:
    counts: Counter[str] = Counter()
    for name in (
        "scan",
        "retinex_enhance",
        "edge_guard_denoise",
        "hue_lock_correct",
        "fuse_candidates",
    ):
        _count(monkeypatch, name, counts)

    stub = StubModelManager(transform=brighten)
    pipeline = EnhancementPipeline(model_manager=stub)
    image = _scene()
    cache = SharedStageCache()
    results = []

    try:
        for mode in MODES:
            results.append(pipeline.run(image, mode=mode, cache=cache))  # type: ignore[arg-type]

        assert stub.calls == 1
        assert counts["scan"] == 1
        assert counts["retinex_enhance"] == 1
        assert counts["fuse_candidates"] == 1
        assert counts["edge_guard_denoise"] == len(MODES)
        assert counts["hue_lock_correct"] == len(MODES)
    finally:
        for result in results:
            result.release()
        cache.close()
        image.close()


def test_cached_outputs_match_uncached_outputs_exactly() -> None:
    image = _scene(seed=3)
    baseline = _run_modes(
        EnhancementPipeline(model_manager=StubModelManager(transform=brighten)),
        image,
    )

    cache = SharedStageCache()
    try:
        shared = _run_modes(
            EnhancementPipeline(model_manager=StubModelManager(transform=brighten)),
            image,
            cache=cache,
        )
    finally:
        cache.close()
        image.close()

    for mode in MODES:
        mae = float(np.mean(np.abs(shared[mode] - baseline[mode])))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            psnr = peak_signal_noise_ratio(
                baseline[mode], shared[mode], data_range=1.0
            )
        ssim = structural_similarity(
            baseline[mode], shared[mode], channel_axis=-1, data_range=1.0
        )
        assert mae == 0.0
        assert np.isinf(psnr)
        assert ssim == pytest.approx(1.0)
        assert np.array_equal(shared[mode], baseline[mode])


def test_shared_stage_timings_are_replayed_per_mode() -> None:
    image = _scene(seed=4)
    cache = SharedStageCache()
    results = {}

    try:
        pipeline = EnhancementPipeline(model_manager=StubModelManager(transform=brighten))
        for mode in MODES:
            results[mode] = pipeline.run(image, mode=mode, cache=cache)  # type: ignore[arg-type]

        for mode, result in results.items():
            expected = {
                stage.name.value
                for stage in pipeline.stages_for(mode)  # type: ignore[arg-type]
            }
            assert set(result.timings_ms) == expected
            assert result.total_ms == pytest.approx(
                sum(result.timings_ms.values()), abs=1e-6
            )
            assert result.total_ms >= 0.0
    finally:
        for result in results.values():
            result.release()
        cache.close()
        image.close()


def test_bypass_gate_respects_sharing_without_model_calls() -> None:
    ramp = np.tile(
        np.linspace(0.25, 0.85, 96, dtype=np.float32)[None, :, None], (96, 1, 3)
    )
    image = Image.fromarray(np.round(ramp * 255.0).astype(np.uint8))
    stub = StubModelManager(
        transform=lambda _image: (_ for _ in ()).throw(
            AssertionError("the model must not run when the bypass gate fires")
        )
    )
    pipeline = EnhancementPipeline(model_manager=stub)
    cache = SharedStageCache()
    results = []

    try:
        for mode in MODES:
            results.append(pipeline.run(image, mode=mode, cache=cache))  # type: ignore[arg-type]

        assert stub.calls == 0
        for result in results:
            assert result.completed is True
            assert np.array_equal(
                np.asarray(result.final_image.convert("RGB")), np.asarray(image)
            )
    finally:
        for result in results:
            result.release()
        cache.close()
        image.close()


def test_inference_failure_is_replayed_without_invalidating_other_modes() -> None:
    image = _scene(seed=5)
    stub = _FailingModel()
    pipeline = EnhancementPipeline(model_manager=stub)
    cache = SharedStageCache()
    classical = pipeline.run(image, mode="classical", cache=cache)

    try:
        assert classical.completed is True
        with pytest.raises(InferenceFailedError, match="synthetic inference failure"):
            pipeline.run(image, mode="deep_learning", cache=cache)
        with pytest.raises(InferenceFailedError, match="synthetic inference failure"):
            pipeline.run(image, mode="hybrid", cache=cache)

        # The failing inference is attempted once and replayed for hybrid.
        assert stub.calls == 1
    finally:
        classical.release()
        cache.close()
        image.close()


def _write_image(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _scene(seed=seed).save(path)


def _fixture(tmp_path: Path) -> None:
    for index in range(2):
        _write_image(tmp_path / "input" / "lime" / f"img_{index}.png", seed=10 + index)


def test_runner_uses_one_inference_per_image(tmp_path, monkeypatch) -> None:
    import training.kaggle_eval.runner as runner

    _fixture(tmp_path)
    counts: Counter[str] = Counter()
    _count(monkeypatch, "scan", counts)
    _count(monkeypatch, "retinex_enhance", counts)
    stub = StubModelManager(transform=brighten)
    monkeypatch.setattr(
        runner,
        "_build_manager",
        lambda config, modes: (stub, stub.status, None, ()),
    )

    config = runner.EvalConfig(
        input_root=tmp_path / "input",
        additional_roots=(),
        output_dir=tmp_path / "out",
        selected_datasets=("lime",),
        modes=("classical", "deep_learning", "hybrid"),
        skip_metrics=("lpips", "niqe"),
        save_visuals=False,
        latency_warmups=0,
        latency_runs=0,
    )
    summary = runner.run_evaluation(config)

    assert summary.failure_count == 0
    assert summary.record_count == 6
    assert stub.calls == 2
    assert counts["scan"] == 2
    assert counts["retinex_enhance"] == 2


def test_runner_outputs_are_identical_with_and_without_sharing(
    tmp_path, monkeypatch
) -> None:
    import training.kaggle_eval.runner as runner

    _fixture(tmp_path)
    stub = StubModelManager(transform=brighten)
    monkeypatch.setattr(
        runner,
        "_build_manager",
        lambda config, modes: (stub, stub.status, None, ()),
    )

    base = runner.EvalConfig(
        input_root=tmp_path / "input",
        additional_roots=(),
        output_dir=tmp_path / "with-sharing",
        selected_datasets=("lime",),
        modes=("original", "classical", "deep_learning", "hybrid"),
        skip_metrics=("lpips", "niqe"),
        save_visuals=True,
        visual_limit=10,
        latency_warmups=0,
        latency_runs=0,
    )
    runner.run_evaluation(base)
    runner.run_evaluation(
        base.replace(output_dir=tmp_path / "without-sharing", share_stages=False)
    )

    def stable(output_dir: Path) -> list[dict]:
        rows = [
            json.loads(line)
            for line in (output_dir / "records" / "records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        for row in rows:
            row.pop("processing_time_ms", None)
            row.pop("stage_timings_ms", None)
        return rows

    assert stable(tmp_path / "with-sharing") == stable(tmp_path / "without-sharing")


def test_runner_can_disable_sharing_for_old_behaviour(tmp_path, monkeypatch) -> None:
    import training.kaggle_eval.runner as runner

    _fixture(tmp_path)
    counts: Counter[str] = Counter()
    _count(monkeypatch, "scan", counts)
    _count(monkeypatch, "retinex_enhance", counts)
    stub = StubModelManager(transform=brighten)
    monkeypatch.setattr(
        runner,
        "_build_manager",
        lambda config, modes: (stub, stub.status, None, ()),
    )

    config = runner.EvalConfig(
        input_root=tmp_path / "input",
        additional_roots=(),
        output_dir=tmp_path / "out",
        selected_datasets=("lime",),
        modes=("classical", "deep_learning", "hybrid"),
        skip_metrics=("lpips", "niqe"),
        save_visuals=False,
        latency_warmups=0,
        latency_runs=0,
        share_stages=False,
    )
    summary = runner.run_evaluation(config)

    assert summary.failure_count == 0
    assert stub.calls == 4
    assert counts["scan"] == 6
    assert counts["retinex_enhance"] == 4


def test_runner_isolates_inference_failure_across_modes(tmp_path, monkeypatch) -> None:
    import training.kaggle_eval.runner as runner

    _fixture(tmp_path)
    stub = _FailingModel()
    monkeypatch.setattr(
        runner,
        "_build_manager",
        lambda config, modes: (stub, stub.status, None, ()),
    )

    config = runner.EvalConfig(
        input_root=tmp_path / "input",
        additional_roots=(),
        output_dir=tmp_path / "out",
        selected_datasets=("lime",),
        modes=("classical", "deep_learning", "hybrid"),
        skip_metrics=("lpips", "niqe"),
        save_visuals=False,
        latency_warmups=0,
        latency_runs=0,
    )
    summary = runner.run_evaluation(config)

    records = [
        json.loads(line)
        for line in (config.output_dir / "records" / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    failures = [
        json.loads(line)
        for line in (config.output_dir / "failures" / "failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert {record["mode"] for record in records} == {"classical"}
    assert all(record["success"] for record in records)
    failed_modes = {failure["mode"] for failure in failures}
    assert failed_modes == {"deep_learning", "hybrid"}
    assert all(failure["code"] == "inference_failed" for failure in failures)
    assert summary.failure_count == 4  # two modes on each of two images
    assert stub.calls == 2  # one failed attempt per image, replayed for hybrid
