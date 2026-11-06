"""LumenFuse: weights, pyramids, normalization and fusion behaviour."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from backend.classical_cv.filters import guided_filter
from backend.classical_cv.scan import scan
from backend.fusion.lumen_fuse import (
    CANDIDATE_COUNT,
    LumenFuseConfig,
    _refine_weight_channels,
    collapse_pyramid,
    compute_weights,
    fuse,
    gaussian_pyramid,
    laplacian_pyramid,
    normalize_weights,
    pyramid_levels,
)


def _textured(
    level: float,
    *,
    amp: float = 0.05,
    size: int = 128,
    seed: int = 0,
    tint: tuple[float, float, float] = (1.0, 0.92, 0.85),
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    texture = (rng.random((size, size, 1)) - 0.5) * 2.0 * amp
    base = np.asarray(tint, dtype=np.float32) * level
    return np.clip(base + texture, 0.0, 1.0).astype(np.float32)


def _uniform(level: float, size: int = 128) -> np.ndarray:
    return np.full((size, size, 3), level, dtype=np.float32)


def _mean_weights(weights: np.ndarray) -> list[float]:
    return [float(weights[..., index].mean()) for index in range(CANDIDATE_COUNT)]


def _reversal_rate(reference: np.ndarray, result: np.ndarray) -> float:
    ref = cv2.GaussianBlur(reference, (0, 0), 2.0)
    out = cv2.GaussianBlur(result, (0, 0), 2.0)
    ref_dx = cv2.Sobel(ref, cv2.CV_32F, 1, 0, ksize=3)
    ref_dy = cv2.Sobel(ref, cv2.CV_32F, 0, 1, ksize=3)
    out_dx = cv2.Sobel(out, cv2.CV_32F, 1, 0, ksize=3)
    out_dy = cv2.Sobel(out, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(ref_dx, ref_dy)
    mask = magnitude >= np.percentile(magnitude, 95.0)
    flips = (np.sign(out_dx[mask]) != np.sign(ref_dx[mask])) | (
        np.sign(out_dy[mask]) != np.sign(ref_dy[mask])
    )
    return float(flips.mean())


# ---------------------------------------------------------------------------
# Weights and normalization
# ---------------------------------------------------------------------------


def test_normalize_weights_handles_all_zero_and_negative_values() -> None:
    zeros = normalize_weights(np.zeros((8, 8, CANDIDATE_COUNT)))

    assert np.allclose(zeros, 1.0 / CANDIDATE_COUNT)
    assert np.allclose(zeros.sum(axis=2), 1.0)

    values = np.zeros((8, 8, CANDIDATE_COUNT))
    values[..., 0] = -1.0
    values[..., 1] = 2.0
    values[..., 2] = 1.0
    normalized = normalize_weights(values, epsilon=1e-6)

    assert normalized.min() >= 0.0
    assert np.allclose(normalized.sum(axis=2), 1.0)
    assert np.allclose(normalized, [0.0, 2.0 / 3.0, 1.0 / 3.0], atol=1e-5)


def test_normalize_weights_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        normalize_weights(np.zeros((8, 8, 2)))
    with pytest.raises(ValueError):
        normalize_weights(np.zeros((8, 8, 3)) + np.nan)


def test_weights_are_normalized_and_non_negative() -> None:
    original = _textured(0.05)
    classical = _textured(0.4, seed=1)
    deep = _textured(0.3, seed=2)

    weights = compute_weights(original, classical, deep)

    assert weights.refined.shape == (128, 128, CANDIDATE_COUNT)
    assert weights.refined.dtype == np.float32
    assert np.allclose(weights.refined.sum(axis=2), 1.0, atol=1e-5)
    assert weights.refined.min() >= 0.0
    assert np.allclose(weights.initial.sum(axis=2), 1.0, atol=1e-6)


def test_zero_refined_weights_fall_back_to_uniform(monkeypatch) -> None:
    original = _textured(0.2)
    classical = _textured(0.4, seed=1)
    deep = _textured(0.3, seed=2)

    monkeypatch.setattr(
        "backend.fusion.lumen_fuse.guided_filter",
        lambda guide, source, radius, eps: np.zeros_like(source),
    )

    weights = compute_weights(original, classical, deep)

    assert np.allclose(weights.refined, 1.0 / CANDIDATE_COUNT)


# ---------------------------------------------------------------------------
# Fusion behaviour
# ---------------------------------------------------------------------------


def test_identical_candidates_return_the_input() -> None:
    image = _textured(0.3)

    result = fuse(image, image.copy(), image.copy())

    assert result.image.shape == image.shape
    assert result.image.dtype == np.float32
    assert float(np.abs(result.image - image).max()) < 1e-5
    assert np.allclose(result.weights.sum(axis=2), 1.0, atol=1e-5)


def test_well_exposed_region_prefers_the_original() -> None:
    original = _textured(0.50)
    classical = _textured(0.90, seed=1)
    deep = _textured(0.85, seed=2)

    result = fuse(original, classical, deep)
    weights = _mean_weights(result.weights)

    assert weights[0] > weights[1]
    assert weights[0] > weights[2]
    assert weights[0] > 0.6
    assert result.image.min() >= 0.0
    assert result.image.max() <= 1.0


def test_dark_region_benefits_from_enhanced_candidates() -> None:
    original = _textured(0.05)
    classical = _textured(0.40, seed=1)
    deep = _textured(0.30, seed=2)

    result = fuse(original, classical, deep)
    weights = _mean_weights(result.weights)

    assert weights[1] + weights[2] > weights[0]
    assert float(result.image.mean()) > 2.0 * float(original.mean())
    assert result.image.min() >= 0.0
    assert result.image.max() <= 1.0


def test_noisy_region_penalizes_high_gain_candidates() -> None:
    rng = np.random.default_rng(5)
    original = np.clip(
        _textured(0.05, amp=0.01) + rng.normal(0, 0.05, (128, 128, 3)), 0, 1
    ).astype(np.float32)
    classical = _textured(0.60, seed=1)
    deep = _textured(0.20, seed=2)
    scan_result = scan(original)

    result = fuse(original, classical, deep, scan_result=scan_result)
    weights = _mean_weights(result.weights)

    assert scan_result.features.noise_sigma > 0.02
    assert scan_result.parameters.fusion_noise == 2.0
    # The strongly amplified candidate is penalised most.
    assert weights[1] < weights[2]
    assert weights[1] < weights[0]


def test_nonuniform_image_selects_regionally() -> None:
    ramp = np.tile(np.linspace(0.02, 0.60, 128, dtype=np.float32)[None, :, None], (128, 1, 3))
    original = np.clip(ramp * np.array([1.0, 0.9, 0.8], np.float32), 0, 1).astype(np.float32)
    classical = np.clip(original**0.6, 0, 1).astype(np.float32)
    deep = np.clip(original**0.7, 0, 1).astype(np.float32)

    result = fuse(original, classical, deep)

    dark_original = float(result.weights[:, :40, 0].mean())
    bright_original = float(result.weights[:, 88:, 0].mean())
    dark_classical = float(result.weights[:, :40, 1].mean())
    bright_classical = float(result.weights[:, 88:, 1].mean())

    assert bright_original > dark_original
    assert dark_classical > bright_classical
    assert result.image[:, :40].mean() > original[:, :40].mean()
    assert result.image.max() <= 1.0


def test_highlight_patch_protects_the_original() -> None:
    original = _uniform(0.4)
    original[40:80, 40:80] = 0.95
    classical = _uniform(0.60)
    deep = _uniform(0.55)

    result = fuse(original, classical, deep)
    patch = result.weights[40:80, 40:80]
    original_weight = float(patch[..., 0].mean())

    assert original_weight > float(patch[..., 1].mean())
    assert original_weight > float(patch[..., 2].mean())
    assert result.image.max() <= 1.0
    assert float(result.image[40:80, 40:80].mean()) < 0.99


def test_colour_distorted_candidates_remain_valid() -> None:
    original = _textured(0.15, tint=(0.9, 0.75, 0.55))
    classical = _textured(0.35, seed=1, tint=(0.8, 0.65, 0.45))
    deep = _textured(0.30, seed=2, tint=(0.7, 0.6, 0.5))

    result = fuse(original, classical, deep)

    assert result.image.shape == original.shape
    assert np.isfinite(result.image).all()
    assert np.allclose(result.weights.sum(axis=2), 1.0, atol=1e-5)
    assert result.image.min() >= 0.0
    assert result.image.max() <= 1.0


def test_extremely_dark_noisy_image_is_stable() -> None:
    rng = np.random.default_rng(7)
    original = np.clip(
        _uniform(0.01) + rng.normal(0, 0.03, (128, 128, 3)), 0, 1
    ).astype(np.float32)
    classical = _textured(0.25, seed=1)
    deep = _textured(0.20, seed=2)

    result = fuse(original, classical, deep)

    assert np.isfinite(result.image).all()
    assert np.isfinite(result.weights).all()
    assert np.allclose(result.weights.sum(axis=2), 1.0, atol=1e-5)
    assert result.image.min() >= 0.0
    assert result.image.max() <= 1.0


def test_all_black_candidates_are_stable() -> None:
    black = np.zeros((64, 64, 3), dtype=np.float32)

    result = fuse(black, black, black)

    assert np.isfinite(result.image).all()
    assert float(result.image.max()) == 0.0
    assert np.allclose(result.weights, 1.0 / CANDIDATE_COUNT)


def test_halo_rate_is_low_at_a_strong_edge() -> None:
    original = np.full((128, 128, 3), 0.20, dtype=np.float32)
    original[:, 64:] = 0.80
    classical = np.full((128, 128, 3), 0.35, dtype=np.float32)
    classical[:, 64:] = 0.90
    deep = np.full((128, 128, 3), 0.30, dtype=np.float32)
    deep[:, 64:] = 0.85

    result = fuse(original, classical, deep)

    assert _reversal_rate(original, result.image) < 0.2
    assert result.image[:, 40].mean() < result.image[:, 90].mean()


def test_odd_dimensions_are_preserved() -> None:
    original = _textured(0.1, size=101)[:77]
    classical = _textured(0.4, size=101, seed=1)[:77]
    deep = _textured(0.3, size=101, seed=2)[:77]

    result = fuse(original, classical, deep)

    assert result.image.shape == original.shape
    assert result.levels == pyramid_levels(101, 77)


def test_batched_guided_filter_matches_single_channel_path() -> None:
    rng = np.random.default_rng(11)
    guide = rng.random((64, 96)).astype(np.float32)
    weights = rng.random((64, 96, CANDIDATE_COUNT)).astype(np.float32)

    batched = _refine_weight_channels(guide, weights, 8, 1e-2)
    single = np.stack(
        [
            guided_filter(guide, weights[..., index], 8, 1e-2)
            for index in range(CANDIDATE_COUNT)
        ],
        axis=2,
    )

    assert np.allclose(batched, single, atol=1e-5)
    assert np.allclose(batched.sum(axis=2), single.sum(axis=2), atol=1e-5)


def test_fuse_does_not_mutate_candidates() -> None:
    original = _textured(0.1)
    classical = _textured(0.4, seed=1)
    deep = _textured(0.3, seed=2)
    copies = (original.copy(), classical.copy(), deep.copy())

    fuse(original, classical, deep)

    for current, before in zip((original, classical, deep), copies, strict=True):
        assert np.array_equal(current, before)


# ---------------------------------------------------------------------------
# Pyramid helpers and levels
# ---------------------------------------------------------------------------


def test_pyramid_round_trip_is_exact() -> None:
    image = np.random.default_rng(0).random((128, 128, 3)).astype(np.float32)

    pyramid = laplacian_pyramid(image, 4)
    restored = collapse_pyramid(pyramid)

    assert len(pyramid) == 5
    assert np.allclose(restored, image, atol=1e-5)


def test_gaussian_pyramid_level_count() -> None:
    image = np.zeros((64, 64), dtype=np.float32)

    pyramid = gaussian_pyramid(image, 3)

    assert len(pyramid) == 4
    assert pyramid[0].shape == (64, 64)
    assert pyramid[-1].shape == (8, 8)


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (128, 128, 4),
        (64, 64, 3),
        (16, 16, 1),
        (8, 8, 0),
        (1920, 1080, 5),
        (4, 4, 0),
    ],
)
def test_pyramid_levels_formula(width: int, height: int, expected: int) -> None:
    assert pyramid_levels(width, height) == expected


# ---------------------------------------------------------------------------
# Invalid inputs and parameters
# ---------------------------------------------------------------------------


def test_invalid_candidates_are_rejected() -> None:
    original = _uniform(0.2)
    other = _uniform(0.3)

    with pytest.raises(ValueError):
        fuse(original, other[:64], other)
    with pytest.raises(ValueError):
        fuse(original, other, other[:64])
    with pytest.raises(ValueError):
        fuse(np.zeros((32, 32), dtype=np.float32), other, other)
    with pytest.raises(ValueError):
        fuse(np.full((32, 32, 3), 2.0, dtype=np.float32), other, other)


def test_invalid_configs_are_rejected() -> None:
    original = _uniform(0.2)
    other = _uniform(0.3)

    for config in (
        LumenFuseConfig(epsilon=0.0),
        LumenFuseConfig(contrast_blur_sigma=0.0),
        LumenFuseConfig(exposure_sigma=0.0),
        LumenFuseConfig(effective_gain_kernel=4),
        LumenFuseConfig(weight_guided_radius=-1),
        LumenFuseConfig(weight_guided_eps=0.0),
        LumenFuseConfig(max_levels=-1),
    ):
        with pytest.raises(ValueError):
            fuse(original, other, other, config=config)


def test_noise_map_shape_mismatch_is_rejected() -> None:
    original = _uniform(0.2)
    scan_result = scan(original)
    broken = replace(scan_result, noise_map=np.zeros((64, 64), dtype=np.float32))

    with pytest.raises(ValueError, match="noise_map"):
        fuse(original, _uniform(0.3), _uniform(0.3), scan_result=broken)


def test_invalid_fusion_parameters_are_rejected() -> None:
    original = _uniform(0.2)
    scan_result = scan(original)
    broken = replace(
        scan_result,
        parameters=replace(scan_result.parameters, fusion_tau=0.0),
    )

    with pytest.raises(ValueError, match="fusion_tau"):
        fuse(original, _uniform(0.3), _uniform(0.3), scan_result=broken)
