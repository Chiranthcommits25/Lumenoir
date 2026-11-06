"""Metric calculations, registry extensibility and aggregation."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from backend.evaluation.aggregate import select_hard_slice, summarise, worst
from backend.evaluation.harness import evaluate_arrays
from backend.evaluation.latency import LatencyBenchmark
from backend.evaluation.registry import register_metric, unregister_metric
from backend.evaluation.types import MetricDefinition, MetricInput, MetricInputError


def _image(seed: int = 0, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None]
    cols = np.linspace(0.0, 1.0, size, dtype=np.float32)[None, :]
    base = (rows + cols) / 2.0
    noise = rng.normal(0.0, 0.02, (size, size)).astype(np.float32)
    channel = np.clip(base + noise, 0.0, 1.0)
    return np.stack([channel, channel**1.5, 1.0 - channel], axis=2).astype(np.float32)


def test_identical_images_are_perfect() -> None:
    image = _image()

    values = evaluate_arrays(
        image,
        reference=image.copy(),
        original=image.copy(),
        keys=[
            "psnr",
            "ssim",
            "delta_e00",
            "edge_preservation",
            "texture_retention",
            "noise_residual",
            "new_clipping",
            "halo_rate",
        ],
    )

    assert values["psnr"].value == math.inf
    assert values["ssim"].value == pytest.approx(1.0, abs=1e-5)
    assert values["delta_e00"].value == pytest.approx(0.0, abs=1e-5)
    assert values["edge_preservation"].value == pytest.approx(1.0, abs=1e-4)
    assert values["texture_retention"].value == pytest.approx(1.0, rel=1e-3)
    assert values["noise_residual"].value == pytest.approx(0.0, abs=1e-6)
    assert values["new_clipping"].value == pytest.approx(0.0)
    assert values["halo_rate"].value == pytest.approx(0.0)


def test_assume_valid_matches_default_validation() -> None:
    original = _image(seed=1)
    reference = _image(seed=2)
    output = np.clip(_image(seed=3) * 1.1, 0.0, 1.0).astype(np.float32)
    keys = ("psnr", "ssim", "delta_e00", "edge_preservation", "new_clipping")

    default = evaluate_arrays(output, reference=reference, original=original, keys=keys)
    trusted = evaluate_arrays(
        output,
        reference=reference,
        original=original,
        keys=keys,
        assume_valid=True,
    )

    for key in keys:
        assert trusted[key].value == default[key].value
        assert trusted[key].notes == default[key].notes


def test_psnr_matches_the_closed_form() -> None:
    reference = np.full((16, 16, 3), 0.4, dtype=np.float32)
    output = np.full((16, 16, 3), 0.5, dtype=np.float32)

    values = evaluate_arrays(output, reference=reference, keys=["psnr"])

    assert values["psnr"].value == pytest.approx(20.0, abs=1e-4)


def test_new_clipping_and_highlight_push() -> None:
    original = np.full((10, 10, 3), 0.2, dtype=np.float32)
    output = original.copy()
    output[:5] = 1.0

    values = evaluate_arrays(output, original=original, keys=["new_clipping"])

    assert values["new_clipping"].value == pytest.approx(0.5)

    near_highlight = np.full((10, 10, 3), 0.9, dtype=np.float32)
    pushed = np.ones((10, 10, 3), dtype=np.float32)

    values = evaluate_arrays(pushed, original=near_highlight, keys=["highlight_push"])

    assert values["highlight_push"].value == pytest.approx(1.0)


def test_noise_metric_is_finite_on_a_noisy_image() -> None:
    rng = np.random.default_rng(1)
    noisy = np.clip(0.5 + rng.normal(0.0, 0.05, (64, 64, 3)), 0.0, 1.0).astype(np.float32)

    values = evaluate_arrays(noisy, keys=["noise_relative"])

    assert values["noise_relative"].value is not None
    assert math.isfinite(values["noise_relative"].value)


def test_missing_reference_is_reported_not_fabricated() -> None:
    values = evaluate_arrays(_image(), keys=["psnr", "ssim"])

    assert values["psnr"].value is None
    assert values["psnr"].notes is not None
    assert "reference" in values["psnr"].notes


def test_metric_registry_is_extensible() -> None:
    definition = MetricDefinition(
        key="mean_level",
        label="Mean level",
        function=lambda metric_input: float(metric_input.image.mean()),
        requires_reference=False,
        unit="intensity",
    )
    register_metric(definition)

    try:
        image = _image()
        values = evaluate_arrays(image, keys=["mean_level"])
        assert values["mean_level"].value == pytest.approx(float(image.mean()))
        assert values["mean_level"].unit == "intensity"
    finally:
        unregister_metric("mean_level")


def test_aggregate_summary_and_worst() -> None:
    summary = summarise([1.0, 2.0, 3.0, 4.0])

    assert summary is not None
    assert summary.count == 4
    assert summary.mean == pytest.approx(2.5)
    assert summary.median == pytest.approx(2.5)
    assert summary.p10 == pytest.approx(1.3)
    assert summary.p90 == pytest.approx(3.7)

    ascending = worst([3.0, 1.0, 2.0, None], higher_is_better=True, n=2)
    assert [value for _, value in ascending] == [1.0, 2.0]

    descending = worst([3.0, 1.0, 2.0], higher_is_better=False, n=1)
    assert descending == [(0, 3.0)]


def test_summarise_ignores_non_finite_values() -> None:
    assert summarise([None, math.inf]) is None
    assert summarise([]) is None


def test_hard_slice_selection() -> None:
    rows = [
        {"name": "a", "p50": 0.02, "nu": 0.5},
        {"name": "b", "p50": 0.30, "nu": 0.5},
        {"name": "c", "p50": 0.50, "nu": 2.5},
        {"name": "d", "p50": 0.60, "nu": 0.5},
    ]

    selected = select_hard_slice(rows)

    assert {row["name"] for row in selected} == {"a", "c"}


def test_latency_benchmark_reports_median_and_stages() -> None:
    def work() -> dict[str, float]:
        time.sleep(0.001)
        return {"stage_a": 2.0}

    result = LatencyBenchmark(work, warmups=1, runs=3, sample_memory=False).run()

    assert result.runs == 3
    assert result.median_ms > 0
    assert result.p95_ms >= result.median_ms
    assert result.per_stage_ms["stage_a"] == pytest.approx(2.0)
    assert result.peak_rss_mb is None


def test_models_are_not_constructed_on_import() -> None:
    """Importing the harness must not build LPIPS or NIQE."""
    import backend.evaluation.no_reference_metrics as no_reference
    import backend.evaluation.reference_metrics as reference

    assert reference._LPIPS_CACHE == {}
    assert no_reference._NIQE_CACHE == {}


def test_lpips_is_lazy_and_calls_the_loaded_model_once(monkeypatch) -> None:
    """The LPIPS branch is exercised without downloading pretrained weights."""
    torch = pytest.importorskip("torch")
    import backend.evaluation.reference_metrics as reference

    created: list[str] = []

    def fake_model(device: str):
        created.append(device)

        def model(left, right):
            return ((left - right) ** 2).mean().reshape(1)

        return model

    monkeypatch.setattr(reference, "_lpips_model", fake_model)

    image = _image(size=64)
    values = evaluate_arrays(image, reference=image.copy(), keys=["lpips"])

    assert values["lpips"].value == pytest.approx(0.0, abs=1e-6)
    assert len(created) == 1
    assert isinstance(torch.zeros(1), torch.Tensor)


def test_lpips_requires_a_reference_without_building_the_model(monkeypatch) -> None:
    import backend.evaluation.reference_metrics as reference

    def fail(_device: str):
        raise AssertionError("The model must not be built without a reference.")

    monkeypatch.setattr(reference, "_lpips_model", fail)

    values = evaluate_arrays(_image(), keys=["lpips"])

    assert values["lpips"].value is None
    assert "reference" in (values["lpips"].notes or "")


def test_niqe_uses_the_cached_metric_without_network(monkeypatch) -> None:
    """NIQE plumbing is tested with a stub metric, not loaded model files."""
    torch = pytest.importorskip("torch")
    import backend.evaluation.no_reference_metrics as no_reference

    calls: list[object] = []

    def fake_metric(tensor):
        calls.append(tensor)
        return torch.tensor(3.5)

    monkeypatch.setattr(no_reference, "_NIQE_CACHE", {"cpu": fake_metric})
    monkeypatch.setenv("LUMENOIR_EVAL_DEVICE", "cpu")

    values = evaluate_arrays(_image(), keys=["niqe"])

    assert values["niqe"].value == pytest.approx(3.5)
    assert len(calls) == 1
    assert tuple(calls[0].shape) == (1, 3, 64, 64)


def test_loe_and_black_crushing() -> None:
    image = _image()

    identical = evaluate_arrays(image, original=image, keys=["loe", "black_crushing"])
    assert identical["loe"].value == pytest.approx(0.0)
    assert identical["black_crushing"].value == pytest.approx(0.0)

    flat = np.random.default_rng(9).random((64, 64, 1)).astype(np.float32)
    gray = np.repeat(flat, 3, axis=2)
    inverted = evaluate_arrays(1.0 - gray, original=gray, keys=["loe"])
    assert inverted["loe"].value is not None
    assert inverted["loe"].value > 0.9

    original = np.full((10, 10, 3), 0.2, dtype=np.float32)
    crushed = original.copy()
    crushed[:2] = 0.0
    values = evaluate_arrays(crushed, original=original, keys=["black_crushing"])
    assert values["black_crushing"].value == pytest.approx(0.2, abs=1e-6)


def test_loe_without_original_is_reported() -> None:
    values = evaluate_arrays(_image(), keys=["loe", "black_crushing"])

    assert values["loe"].value is None
    assert values["loe"].notes is not None
    assert values["black_crushing"].value is None


def test_compute_metrics_adapter_without_reference(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from PIL import Image

    import backend.evaluation.no_reference_metrics as no_reference
    from backend.evaluation.metrics import MetricName, compute_metrics

    monkeypatch.setattr(no_reference, "_NIQE_CACHE", {"cpu": lambda _tensor: torch.tensor(4.0)})
    monkeypatch.setenv("LUMENOIR_EVAL_DEVICE", "cpu")

    image = Image.fromarray((_image() * 255).astype(np.uint8))
    report = compute_metrics(image=image, original=image, processing_time_ms=12.5)

    assert report.reference_available is False
    assert report.value(MetricName.PSNR) is None
    assert report.value(MetricName.NIQE) == pytest.approx(4.0)
    assert report.value(MetricName.LATENCY) == pytest.approx(12.5)
    assert report.value(MetricName.COLOUR_DISTORTION) == pytest.approx(0.0, abs=1e-4)


def test_metric_input_factory() -> None:
    image = _image()
    metric_input = MetricInput(image=image)

    assert metric_input.reference is None
    with pytest.raises(MetricInputError):
        metric_input.require_reference()
