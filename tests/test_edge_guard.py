"""EdgeGuard: gain-aware luma and chroma denoising."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.classical_cv.arrays import luminance
from backend.classical_cv.edge_guard import (
    EdgeGuardConfig,
    denoise,
    effective_gain_map,
)
from backend.classical_cv.noise import noise_sigma
from backend.classical_cv.scan import scan


def _flat(level: float, size: int = 128) -> np.ndarray:
    return np.full((size, size, 3), level, dtype=np.float32)


def _noisy(level: float, sigma: float, seed: int = 0, size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((size, size, 3), level, dtype=np.float32)
    return np.clip(base + rng.normal(0.0, sigma, base.shape), 0.0, 1.0).astype(np.float32)


def _chroma_std(image: np.ndarray) -> tuple[float, float]:
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    return float(ycrcb[..., 1].std()), float(ycrcb[..., 2].std())


def test_output_shape_dtype_and_range() -> None:
    fused = _noisy(0.3, 0.02)
    original = _flat(0.05)

    result = denoise(fused, original, sigma=0.02)

    assert result.image.shape == fused.shape
    assert result.image.dtype == np.float32
    assert result.image.min() >= 0.0
    assert result.image.max() <= 1.0
    assert result.effective_gain.shape == fused.shape[:2]
    assert result.luma_weight.shape == fused.shape[:2]


def test_effective_gain_and_reference() -> None:
    fused = _flat(0.30)
    original = _flat(0.05)

    gain = effective_gain_map(fused, original)

    assert float(gain.mean()) == pytest.approx(0.30 / 0.07, rel=0.01)
    assert float(gain.min()) >= 1.0
    assert float(gain.max()) <= 30.0

    result = denoise(fused, original, sigma=0.0)
    assert result.gain_reference == pytest.approx(float(np.percentile(gain, 75)), rel=1e-5)


def test_luma_noise_is_reduced() -> None:
    fused = _noisy(0.30, 0.02, seed=1)
    original = _flat(0.05)
    before = noise_sigma(luminance(fused))

    result = denoise(fused, original, sigma=0.02)
    after = noise_sigma(luminance(result.image))

    assert before > 0.005
    assert after < 0.5 * before
    assert float(result.luma_weight.mean()) > 0.8


def test_chroma_noise_is_reduced() -> None:
    fused = _noisy(0.30, 0.02, seed=2)
    original = _flat(0.05)
    before_cr, before_cb = _chroma_std(fused)

    result = denoise(fused, original, sigma=0.02)
    after_cr, after_cb = _chroma_std(result.image)

    assert after_cr < 0.5 * before_cr
    assert after_cb < 0.5 * before_cb
    assert result.sigma_color > 8.0


def test_edges_are_preserved() -> None:
    fused = np.full((128, 128, 3), 0.30, dtype=np.float32)
    fused[:, 64:] = 0.90
    original = _flat(0.05)
    input_contrast = float(fused[:, :56].mean() - fused[:, 72:].mean())

    result = denoise(fused, original, sigma=0.01)
    output_contrast = float(result.image[:, :56].mean() - result.image[:, 72:].mean())

    assert abs(output_contrast) > 0.9 * abs(input_contrast)
    assert result.image[:, 40].mean() < result.image[:, 90].mean()


def test_texture_is_retained() -> None:
    checker = ((np.indices((128, 128)).sum(axis=0) // 16) % 2).astype(np.float32)
    fused = np.stack([0.30 + 0.10 * checker] * 3, axis=2).astype(np.float32)
    original = _flat(0.05)
    before = float(luminance(fused).std())

    result = denoise(fused, original, sigma=0.01)
    after = float(luminance(result.image).std())

    assert after > 0.5 * before


def test_well_lit_image_skips_luma_denoising() -> None:
    fused = _flat(0.60)
    original = _flat(0.60)

    result = denoise(fused, original, sigma=0.0)

    assert result.gain_reference <= 1.5
    assert float(result.luma_weight.max()) == 0.0
    assert np.allclose(luminance(result.image), luminance(fused), atol=1e-5)
    assert result.image.shape == fused.shape


def test_gain_is_higher_in_the_dark_region() -> None:
    original = np.full((128, 128, 3), 0.02, dtype=np.float32)
    original[:, 64:] = 0.50
    fused = np.full((128, 128, 3), 0.20, dtype=np.float32)
    fused[:, 64:] = 0.55

    result = denoise(fused, original, sigma=0.01)

    assert float(result.effective_gain[:, :56].mean()) > float(
        result.effective_gain[:, 72:].mean()
    )
    assert float(result.luma_weight[:, :56].mean()) > float(
        result.luma_weight[:, 72:].mean()
    )


def test_sigma_can_come_from_lumen_scan() -> None:
    original = _noisy(0.05, 0.05, seed=3)
    fused = _noisy(0.30, 0.02, seed=4)
    scan_result = scan(original)

    from_scan = denoise(fused, original, scan_result=scan_result)
    explicit = denoise(fused, original, sigma=scan_result.features.noise_sigma)

    assert scan_result.features.noise_sigma > 0.02
    assert from_scan.sigma == pytest.approx(scan_result.features.noise_sigma)
    assert np.allclose(from_scan.image, explicit.image, atol=1e-6)


def test_gain_is_capped() -> None:
    fused = _flat(1.0)
    original = _flat(0.001)

    gain = effective_gain_map(fused, original)

    assert float(gain.min()) == pytest.approx(30.0)
    assert float(gain.max()) == pytest.approx(30.0)


def test_large_image_uses_the_larger_luma_radius() -> None:
    fused = _flat(0.30, 1100)
    original = _flat(0.05, 1100)

    result = denoise(fused, original, sigma=0.0)

    assert result.luma_radius == 6


def test_invalid_inputs_are_rejected() -> None:
    fused = _flat(0.3)
    original = _flat(0.05)

    with pytest.raises(ValueError):
        denoise(np.zeros((32, 32), dtype=np.float32), original, sigma=0.01)
    with pytest.raises(ValueError):
        denoise(fused, np.zeros((64, 64), dtype=np.float32), sigma=0.01)
    with pytest.raises(ValueError):
        denoise(fused, original, sigma=-0.01)
    with pytest.raises(ValueError):
        denoise(fused, original)
    with pytest.raises(ValueError):
        denoise(fused, original, sigma=0.01, config=EdgeGuardConfig(effective_gain_kernel=4))
