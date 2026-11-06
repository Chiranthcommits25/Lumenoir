"""Box and guided filter tests."""

from __future__ import annotations

import numpy as np
import pytest

from backend.classical_cv.filters import box_filter, guided_filter


def test_box_filter_preserves_a_constant_image() -> None:
    image = np.full((32, 32), 0.4, dtype=np.float32)

    result = box_filter(image, 3)

    assert result.shape == image.shape
    assert result.dtype == np.float32
    assert np.allclose(result, 0.4, atol=1e-6)


def test_box_filter_of_an_impulse_sums_to_one() -> None:
    impulse = np.zeros((41, 41), dtype=np.float32)
    impulse[20, 20] = 1.0

    result = box_filter(impulse, 2)

    assert result.sum() == pytest.approx(1.0, abs=1e-5)
    assert result[20, 20] == pytest.approx(1.0 / 25.0, abs=1e-6)


def test_box_filter_supports_rgb() -> None:
    image = np.random.default_rng(0).random((16, 24, 3)).astype(np.float32)

    result = box_filter(image, 2)

    assert result.shape == image.shape
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_guided_filter_identity_returns_the_input() -> None:
    image = np.random.default_rng(1).random((64, 64)).astype(np.float32)

    result = guided_filter(image, image, radius=4, eps=1e-4)

    assert np.allclose(result, image, atol=5e-3)


def test_guided_filter_keeps_a_flat_image_flat() -> None:
    image = np.full((48, 48), 0.3, dtype=np.float32)

    result = guided_filter(image, image, radius=5, eps=1e-3)

    assert np.allclose(result, 0.3, atol=1e-6)


def test_guided_filter_preserves_a_step_edge_better_than_box() -> None:
    step = np.zeros((64, 64), dtype=np.float32)
    step[:, 32:] = 1.0

    guided = guided_filter(step, step, radius=8, eps=1e-3)
    boxed = box_filter(step, 8)

    guided_intermediate = int(((guided > 0.05) & (guided < 0.95)).sum())
    box_intermediate = int(((boxed > 0.05) & (boxed < 0.95)).sum())

    assert guided_intermediate < box_intermediate
    assert guided_intermediate <= 4


def test_guided_filter_rejects_invalid_arguments() -> None:
    image = np.zeros((16, 16), dtype=np.float32)

    with pytest.raises(ValueError):
        guided_filter(image, image, radius=2, eps=0.0)
    with pytest.raises(ValueError):
        guided_filter(image, np.zeros((8, 8), dtype=np.float32), radius=2, eps=1e-3)
    with pytest.raises(ValueError):
        guided_filter(image[None, ...], image[None, ...], radius=2, eps=1e-3)
    with pytest.raises(ValueError):
        box_filter(image, -1)
    with pytest.raises(ValueError):
        box_filter(image, 1.5)  # type: ignore[arg-type]
