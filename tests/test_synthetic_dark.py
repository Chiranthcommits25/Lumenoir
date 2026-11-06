"""The specified extreme-dark synthesis formula."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from training.data_pipeline.synthetic import (
    SyntheticDarkError,
    generate_synthetic_darks,
    synth_dark,
)


def _gradient(height: int = 32, width: int = 32) -> np.ndarray:
    rows = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    cols = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    base = (rows + cols) / 2.0
    return np.stack([base, base**2, np.sqrt(base)], axis=2).astype(np.float32)


def test_output_is_float32_in_range_and_quantised() -> None:
    darkened = synth_dark(_gradient(), np.random.default_rng(0))

    assert darkened.dtype == np.float32
    assert darkened.shape == (32, 32, 3)
    assert darkened.min() >= 0.0
    assert darkened.max() <= 1.0
    assert np.allclose(darkened * 255.0, np.round(darkened * 255.0), atol=1e-5)


def test_output_is_darker_than_the_input() -> None:
    image = _gradient()
    darkened = synth_dark(image, np.random.default_rng(1))

    assert darkened.mean() < image.mean()


def test_same_seed_reproduces_exactly() -> None:
    image = _gradient()

    first = synth_dark(image, np.random.default_rng(42))
    second = synth_dark(image, np.random.default_rng(42))
    third = synth_dark(image, np.random.default_rng(43))

    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


def test_input_is_not_modified() -> None:
    image = _gradient()
    before = image.copy()

    synth_dark(image, np.random.default_rng(0))

    assert np.array_equal(image, before)


def test_uint8_input_is_accepted() -> None:
    image = (_gradient() * 255).astype(np.uint8)

    darkened = synth_dark(image, np.random.default_rng(0))

    assert darkened.dtype == np.float32
    assert darkened.mean() < image.mean() / 255.0


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((10, 10), dtype=np.float32),
        np.zeros((10, 10, 4), dtype=np.float32),
        np.full((10, 10, 3), 2.0, dtype=np.float32),
        np.full((10, 10, 3), np.nan, dtype=np.float32),
    ],
)
def test_invalid_inputs_are_rejected(bad: np.ndarray) -> None:
    with pytest.raises(SyntheticDarkError):
        synth_dark(bad, np.random.default_rng(0))


def test_generator_requires_an_explicit_count(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.fromarray((_gradient() * 255).astype(np.uint8)).save(source)

    with pytest.raises(SyntheticDarkError):
        generate_synthetic_darks([source], tmp_path / "out", count=0)

    written = generate_synthetic_darks([source], tmp_path / "out", count=3, seed=0)

    assert len(written) == 3
    assert all(path.is_file() for path in written)


def test_generator_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.fromarray((_gradient() * 255).astype(np.uint8)).save(source)

    first = generate_synthetic_darks([source], tmp_path / "a", count=2, seed=5)
    second = generate_synthetic_darks([source], tmp_path / "b", count=2, seed=5)

    for left, right in zip(first, second, strict=True):
        assert np.array_equal(np.asarray(Image.open(left)), np.asarray(Image.open(right)))
