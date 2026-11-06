"""LumenRetinex: gain formula, highlight protection, CLAHE and do-no-harm."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from backend.classical_cv.retinex import (
    LumenRetinexConfig,
    apply_clahe,
    apply_highlight_shoulder,
    compute_gain,
    enhance,
)
from backend.classical_cv.scan import scan


def _uniform(level: float, size: int = 128) -> np.ndarray:
    return np.full((size, size, 3), level, dtype=np.float32)


def _ramp(low: float, high: float, size: int = 128) -> np.ndarray:
    row = np.linspace(low, high, size, dtype=np.float32)
    plane = np.tile(row, (size, 1))
    return np.stack([plane, plane, plane], axis=2)


def _noisy(level: float, sigma: float, seed: int = 0, size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((size, size, 3), level, dtype=np.float32)
    return np.clip(base + rng.normal(0.0, sigma, base.shape), 0.0, 1.0).astype(np.float32)


def _expected_gain(
    max_channel: np.ndarray,
    illumination: np.ndarray,
    *,
    gamma: float,
    gain_cap: float,
    low: float = 0.65,
    high: float = 0.95,
    epsilon: float = 1e-3,
) -> np.ndarray:
    gain = np.clip(illumination ** (gamma - 1.0), 1.0, gain_cap)
    t = np.clip((max_channel - low) / (high - low), 0.0, 1.0)
    shoulder = 1.0 - t * t * (3.0 - 2.0 * t)
    attenuated = 1.0 + (gain - 1.0) * shoulder
    return np.minimum(attenuated, 1.0 / (max_channel + epsilon))


def test_gain_matches_the_manual_formula() -> None:
    max_values = np.array([[0.2, 0.7], [0.9, 1.0]], dtype=np.float32)
    image = np.repeat(max_values[..., None], 3, axis=2)
    illumination = np.array([[0.1, 0.5], [0.8, 0.9]], dtype=np.float32)
    params = replace(scan(_uniform(0.5, 32)).parameters, gamma=0.5, gain_cap=6.0)

    gain = compute_gain(image, illumination, params)

    expected = _expected_gain(
        max_values, illumination, gamma=0.5, gain_cap=6.0
    )
    assert np.allclose(gain, expected, atol=1e-6)


def test_dark_image_uses_the_exact_gain() -> None:
    image = _uniform(0.05)
    params = scan(image).parameters
    illumination = np.full((128, 128), 0.05, dtype=np.float32)
    config = LumenRetinexConfig(apply_clahe=False, apply_shoulder=False)

    result = enhance(image, config=config, illumination=illumination)

    expected_gain = float(0.05 ** (params.gamma - 1.0))
    assert result.bypassed is False
    assert result.gain is not None
    assert result.gain[0, 0] == pytest.approx(expected_gain, rel=1e-5)
    assert result.image.mean() == pytest.approx(0.05 * expected_gain, rel=1e-4)
    assert result.image.shape == image.shape


def test_extremely_dark_image_is_gain_capped() -> None:
    image = _uniform(0.01)
    params = scan(image).parameters
    config = LumenRetinexConfig(apply_clahe=False, apply_shoulder=False)

    result = enhance(
        image,
        config=config,
        illumination=np.full((128, 128), 0.01, dtype=np.float32),
    )

    assert params.gain_cap == pytest.approx(10.0)
    assert float(result.gain.min()) == pytest.approx(10.0)
    assert float(result.gain.max()) == pytest.approx(10.0)
    assert result.image.mean() == pytest.approx(0.10, abs=1e-4)


def test_noisy_dark_image_uses_the_noise_limited_cap() -> None:
    image = _noisy(0.05, 0.05, seed=7)
    scan_result = scan(image)
    config = LumenRetinexConfig(apply_clahe=False, apply_shoulder=False)

    result = enhance(
        image,
        scan_result=scan_result,
        config=config,
        illumination=np.full((128, 128), 0.05, dtype=np.float32),
    )

    assert scan_result.features.noise_sigma > 0.02
    assert result.parameters.gain_cap == pytest.approx(3.0)
    assert float(result.gain.min()) == pytest.approx(3.0)
    assert float(result.gain.max()) == pytest.approx(3.0)
    assert result.image.max() <= 1.0
    assert result.image.min() >= 0.0


def test_nonuniform_dark_dominant_image_is_enhanced_regionally() -> None:
    image = np.full((128, 128, 3), 0.02, dtype=np.float32)
    image[:, 90:] = 0.95

    scan_result = scan(image)
    result = enhance(image, scan_result=scan_result)

    assert scan_result.parameters.bypass_gate is False
    assert result.gain is not None
    dark_gain = float(result.gain[:, :80].mean())
    bright_gain = float(result.gain[:, 100:].mean())

    assert dark_gain > 5.0
    assert bright_gain < 1.2
    assert result.image[:, :80].mean() > image[:, :80].mean()
    assert result.image.max() <= 1.0
    assert result.image.shape == image.shape


def test_bimodal_image_meeting_the_gate_is_bypassed() -> None:
    image = np.full((128, 128, 3), 0.02, dtype=np.float32)
    image[:, 64:] = 0.90

    result = enhance(image)

    # p50 ~0.46, DR ~0.86, no clipped highlights -> §5.8 bypass.
    assert result.bypassed is True
    assert result.gain is None
    assert np.array_equal(result.image, image)


def test_well_lit_image_is_bypassed() -> None:
    image = _ramp(0.25, 0.85)

    result = enhance(image)

    assert result.bypassed is True
    assert np.array_equal(result.image, image)


def test_well_lit_uniform_image_is_nearly_unchanged() -> None:
    image = _uniform(0.6)

    result = enhance(image, scan_result=scan(image))
    assert result.bypassed is False
    assert result.gain is not None
    assert float(result.gain.min()) == pytest.approx(1.0)
    assert float(result.gain.max()) == pytest.approx(1.0)
    assert np.abs(result.image - image).max() < 0.01


def test_highlight_shoulder_compresses_only_the_top_end() -> None:
    row = np.array([[0.4, 0.4, 0.4], [0.85, 0.85, 0.85], [0.99, 0.99, 0.99]])
    values = np.tile(row[None, ...], (2, 1, 1)).astype(np.float32)

    result = apply_highlight_shoulder(values)

    assert result[0, 0, 0] == pytest.approx(0.4)
    assert result[0, 1, 0] == pytest.approx(0.85)
    assert 0.85 < result[0, 2, 0] < 0.99
    assert result.max() <= 1.0


def test_highlight_protection_in_the_full_pipeline() -> None:
    image = np.full((128, 128, 3), 0.35, dtype=np.float32)
    image[40:80, 40:80] = 1.0

    result = enhance(image)

    assert result.bypassed is False
    assert result.image.max() < 0.99
    assert result.image.min() >= 0.0
    assert result.image.shape == image.shape


def test_clahe_increases_contrast_and_preserves_dimensions() -> None:
    image = _ramp(0.35, 0.45)
    params = scan(image).parameters
    before = np.asarray(image, dtype=np.float32)
    after = apply_clahe(image, params)

    assert after.shape == before.shape
    luma_before = before.mean(axis=2)
    luma_after = after.mean(axis=2)
    assert float(luma_after.std()) > float(luma_before.std())
    assert after.min() >= 0.0
    assert after.max() <= 1.0


def test_clahe_preserves_hue_for_mid_tone_pixels() -> None:
    base = np.zeros((128, 128, 3), dtype=np.float32)
    base[..., 0] = np.linspace(0.30, 0.50, 128, dtype=np.float32)[None, :]
    base[..., 1] = np.linspace(0.25, 0.40, 128, dtype=np.float32)[None, :]
    base[..., 2] = 0.20
    params = scan(base).parameters

    result = apply_clahe(base, params)

    lab_before = cv2.cvtColor(base, cv2.COLOR_RGB2LAB)
    lab_after = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
    chroma = np.hypot(lab_before[..., 1], lab_before[..., 2])
    mask = chroma > 10.0
    assert mask.any()

    hue_before = np.arctan2(lab_before[..., 2], lab_before[..., 1])
    hue_after = np.arctan2(lab_after[..., 2], lab_after[..., 1])
    wrapped = np.angle(np.exp(1j * (hue_after - hue_before)))
    assert float(np.degrees(np.abs(wrapped[mask])).mean()) < 1.0


def test_gain_is_identical_across_channels_and_preserves_size() -> None:
    image = np.zeros((96, 128, 3), dtype=np.float32)
    image[..., 0] = np.linspace(0.1, 0.3, 128, dtype=np.float32)[None, :]
    image[..., 1] = 0.2
    image[..., 2] = 0.15
    config = LumenRetinexConfig(apply_clahe=False, apply_shoulder=False)

    result = enhance(image, config=config)

    assert result.gain is not None
    assert result.image.shape == image.shape
    assert np.allclose(
        result.image, np.clip(image * result.gain[..., None], 0.0, 1.0), atol=1e-6
    )


def test_scan_result_is_computed_when_omitted() -> None:
    result = enhance(_uniform(0.1))

    assert result.bypassed is False
    assert result.gain is not None
    assert result.illumination is not None
    assert result.illumination.shape == (128, 128)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        enhance(np.zeros((32, 32), dtype=np.float32))
    with pytest.raises(ValueError):
        enhance(np.full((32, 32, 3), 2.0, dtype=np.float32))

    image = _uniform(0.1)
    params = scan(image).parameters
    with pytest.raises(ValueError):
        compute_gain(image, np.full((64, 64), 0.1, dtype=np.float32), params)
    with pytest.raises(ValueError):
        compute_gain(image, np.zeros((128, 128), dtype=np.float32), params)
    with pytest.raises(ValueError):
        enhance(image, illumination=np.zeros((128, 128), dtype=np.float32))
