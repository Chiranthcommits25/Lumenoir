"""Immerkaer noise estimation tests."""

from __future__ import annotations

import numpy as np
import pytest

from backend.classical_cv.noise import (
    flat_mask,
    noise_map,
    noise_sigma,
    relative_noise,
)


def _uniform(level: float = 0.5, size: int = 256) -> np.ndarray:
    return np.full((size, size), level, dtype=np.float32)


def _noisy(level: float, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(
        level + rng.normal(0.0, sigma, (256, 256)), 0.0, 1.0
    ).astype(np.float32)


def test_flat_image_has_no_noise() -> None:
    assert noise_sigma(_uniform()) == pytest.approx(0.0, abs=1e-8)
    assert relative_noise(_uniform()) == pytest.approx(0.0, abs=1e-8)


def test_gaussian_noise_is_estimated_in_the_right_range() -> None:
    estimate = noise_sigma(_noisy(0.5, 0.05, seed=42))

    # A single-channel image at sigma=0.05; the estimator should be close.
    assert 0.035 < estimate < 0.065


def test_flat_mask_selects_about_half_the_pixels() -> None:
    mask = flat_mask(_noisy(0.5, 0.02, seed=1), percentile=50.0)

    assert mask.shape == (254, 254)
    assert mask.mean() == pytest.approx(0.5, abs=0.02)


def test_noise_map_shape_and_clipping() -> None:
    gray = _noisy(0.5, 0.05, seed=2)
    sigma = noise_sigma(gray)

    local = noise_map(gray, radius=7, sigma=sigma)

    assert local.shape == gray.shape
    assert local.dtype == np.float32
    assert local.min() >= 0.5 * sigma - 1e-6
    assert local.max() <= 2.0 * sigma + 1e-6


def test_evaluation_module_reuses_the_same_implementation() -> None:
    from backend.classical_cv.noise import noise_sigma as canonical
    from backend.evaluation.noise import noise_sigma as evaluated

    assert evaluated is canonical


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        noise_sigma(np.zeros((16, 16, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        noise_sigma(np.zeros((1, 16), dtype=np.float32))
    with pytest.raises(ValueError):
        flat_mask(_uniform(), percentile=140.0)
