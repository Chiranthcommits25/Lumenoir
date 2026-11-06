"""Invalid metric inputs are rejected, never silently coerced."""

from __future__ import annotations

import numpy as np
import pytest

from backend.evaluation.harness import evaluate_arrays
from backend.evaluation.images import to_float32_rgb
from backend.evaluation.reference_metrics import compute_psnr, compute_ssim
from backend.evaluation.types import MetricInput, MetricInputError


def _valid(size: int = 32) -> np.ndarray:
    return np.full((size, size, 3), 0.5, dtype=np.float32)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((10, 10), dtype=np.float32),
        np.zeros((10, 10, 4), dtype=np.float32),
        np.zeros((0, 10, 3), dtype=np.float32),
        np.full((10, 10, 3), np.nan, dtype=np.float32),
        np.full((10, 10, 3), 1.5, dtype=np.float32),
        np.full((10, 10, 3), -0.1, dtype=np.float32),
        np.zeros((10, 10, 3), dtype=np.int32),
    ],
)
def test_invalid_arrays_are_rejected(bad: np.ndarray) -> None:
    with pytest.raises(MetricInputError):
        to_float32_rgb(bad)


def test_uint8_is_converted_not_rejected() -> None:
    converted = to_float32_rgb(np.full((8, 8, 3), 255, dtype=np.uint8))

    assert converted.dtype == np.float32
    assert converted.max() == pytest.approx(1.0)


def test_reference_is_required() -> None:
    with pytest.raises(MetricInputError):
        compute_psnr(MetricInput(image=_valid()))

    with pytest.raises(MetricInputError):
        compute_ssim(MetricInput(image=_valid()))


def test_small_images_are_rejected_by_ssim() -> None:
    small = np.full((4, 4, 3), 0.5, dtype=np.float32)

    with pytest.raises(MetricInputError):
        compute_ssim(MetricInput(image=small, reference=small.copy()))


def test_shape_mismatch_is_reported_per_metric() -> None:
    values = evaluate_arrays(
        _valid(32),
        reference=_valid(16),
        keys=["psnr", "ssim"],
    )

    assert values["psnr"].value is None
    assert values["psnr"].notes is not None
    assert values["ssim"].value is None


def test_small_ssim_window_is_reported_not_crashing_the_harness() -> None:
    small = np.full((4, 4, 3), 0.5, dtype=np.float32)

    values = evaluate_arrays(small, reference=small.copy(), keys=["ssim", "psnr"])

    assert values["ssim"].value is None
    assert values["psnr"].value is not None


def test_unknown_metric_key_raises() -> None:
    with pytest.raises(KeyError):
        evaluate_arrays(_valid(), keys=["definitely_not_a_metric"])


def test_non_reference_metric_without_inputs_is_reported() -> None:
    values = evaluate_arrays(_valid(), keys=["new_clipping", "halo_rate"])

    assert values["new_clipping"].value is None
    assert values["new_clipping"].notes is not None
    assert values["halo_rate"].value is None
