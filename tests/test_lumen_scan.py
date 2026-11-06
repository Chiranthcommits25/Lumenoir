"""LumenScan statistics and rule-based parameter selection."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from backend.classical_cv.arrays import smoothstep
from backend.classical_cv.scan import (
    DarknessClass,
    LumenScanConfig,
    scan,
)


def _uniform(level: float, size: int = 256) -> np.ndarray:
    return np.full((size, size, 3), level, dtype=np.float32)


def _ramp(low: float, high: float, size: int = 256) -> np.ndarray:
    row = np.linspace(low, high, size, dtype=np.float32)
    plane = np.tile(row, (size, 1))
    return np.stack([plane, plane, plane], axis=2)


def _noisy(level: float, sigma: float, seed: int = 0, size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((size, size, 3), level, dtype=np.float32)
    return np.clip(base + rng.normal(0.0, sigma, base.shape), 0.0, 1.0).astype(np.float32)


def test_uniform_normal_image() -> None:
    result = scan(_uniform(0.6))
    features = result.features
    parameters = result.parameters

    assert features.darkness is DarknessClass.NORMAL
    assert features.mean == pytest.approx(0.6, abs=1e-4)
    assert features.p50 == pytest.approx(0.6, abs=1e-4)
    assert features.dynamic_range == pytest.approx(0.0, abs=1e-6)
    assert features.highlight_fraction == 0.0
    assert features.black_fraction == 0.0

    assert parameters.gamma == 1.0
    assert parameters.dl_strength == pytest.approx(0.4)
    assert parameters.gain_cap == 10.0
    assert parameters.guided_eps == pytest.approx(1e-3)
    assert parameters.output_blend_alpha == 0.0
    assert parameters.bypass_gate is False
    assert parameters.hue_lock_beta == 0.0


def test_percentiles_and_dynamic_range() -> None:
    features = scan(_ramp(0.0, 1.0)).features

    assert features.p5 == pytest.approx(0.05, abs=0.02)
    assert features.p50 == pytest.approx(0.50, abs=0.02)
    assert features.p95 == pytest.approx(0.95, abs=0.02)
    assert features.dynamic_range == pytest.approx(0.90, abs=0.03)


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        (0.03, DarknessClass.EXTREME),
        (0.10, DarknessClass.DARK),
        (0.30, DarknessClass.DIM),
        (0.60, DarknessClass.NORMAL),
    ],
)
def test_darkness_classes(level: float, expected: DarknessClass) -> None:
    assert scan(_uniform(level)).features.darkness is expected


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        (0.6, 1.0),
        (0.2, 0.4989),
        (0.03, 0.30),
    ],
)
def test_gamma_rule(level: float, expected: float) -> None:
    assert scan(_uniform(level)).parameters.gamma == pytest.approx(expected, abs=5e-3)


def test_gamma_matches_the_closed_form() -> None:
    for level in (0.12, 0.25, 0.40):
        expected = min(1.0, max(0.30, math.log(0.45) / math.log(level)))
        assert scan(_uniform(level)).parameters.gamma == pytest.approx(expected, abs=1e-3)


def test_noise_drives_gain_cap_and_guided_epsilon() -> None:
    clean = scan(_uniform(0.3))
    noisy = scan(_noisy(0.3, 0.05, seed=3))

    assert clean.features.noise_sigma < 0.008
    assert clean.parameters.gain_cap == 10.0
    assert clean.parameters.guided_eps == pytest.approx(1e-3)
    assert clean.parameters.fusion_contrast == 1.0

    assert noisy.features.noise_sigma > 0.02
    assert noisy.parameters.gain_cap == 3.0
    assert noisy.parameters.guided_eps == pytest.approx(3e-3)
    assert noisy.parameters.fusion_contrast == 0.5
    assert noisy.parameters.fusion_noise == 2.0
    assert noisy.parameters.fusion_tau == 0.06


def test_fusion_mid_noise_bucket() -> None:
    result = scan(_noisy(0.3, 0.015, seed=4))

    assert 0.008 <= result.features.noise_sigma < 0.02
    assert result.parameters.fusion_contrast == 0.7
    assert result.parameters.fusion_saturation == 0.5
    assert result.parameters.fusion_exposure == 1.0
    assert result.parameters.fusion_noise == 1.5
    assert result.parameters.fusion_tau == 0.08


def test_clahe_rule() -> None:
    flat = scan(_uniform(0.3))
    noisy = scan(_noisy(0.3, 0.05, seed=5))

    assert flat.parameters.clahe_clip == pytest.approx(4.0)
    assert noisy.parameters.clahe_clip < flat.parameters.clahe_clip
    assert 1.0 <= noisy.parameters.clahe_clip <= 4.0
    assert flat.parameters.clahe_blend == pytest.approx(0.6)


def test_clahe_tile_grid_depends_on_pixels() -> None:
    assert scan(_uniform(0.3, 256)).parameters.clahe_tiles == (8, 8)

    large = np.full((900, 1200, 3), 0.3, dtype=np.float32)
    assert scan(large).parameters.clahe_tiles == (12, 12)


def test_output_blend_alpha() -> None:
    assert scan(_uniform(0.1)).parameters.output_blend_alpha == pytest.approx(1.0)
    assert scan(_uniform(0.6)).parameters.output_blend_alpha == pytest.approx(0.0)

    expected = 1.0 - smoothstep(0.25, 0.50, 0.4)
    assert scan(_uniform(0.4)).parameters.output_blend_alpha == pytest.approx(expected)


def test_bypass_gate() -> None:
    normal = scan(_ramp(0.25, 0.85)).features
    assert normal.p50 >= 0.45
    assert normal.dynamic_range >= 0.45
    assert normal.highlight_fraction < 0.01
    assert scan(_ramp(0.25, 0.85)).parameters.bypass_gate is True

    assert scan(_uniform(0.6)).parameters.bypass_gate is False

    clipped = _ramp(0.25, 0.85).copy()
    clipped.reshape(-1, 3)[: int(0.05 * 256 * 256)] = 1.0
    assert scan(clipped).parameters.bypass_gate is False


def test_highlight_and_black_fractions() -> None:
    image = _uniform(0.5)
    flat = image.reshape(-1, 3)
    total = flat.shape[0]
    flat[: int(0.10 * total)] = 1.0
    flat[int(0.10 * total) : int(0.30 * total)] = 0.0

    features = scan(image).features

    assert features.highlight_fraction == pytest.approx(0.10, abs=1e-4)
    assert features.black_fraction == pytest.approx(0.20, abs=1e-4)


def test_illumination_nonuniformity() -> None:
    uniform = scan(_uniform(0.5)).features
    assert uniform.nonuniformity < 0.05
    assert uniform.illumination_label == "uniform"

    split = np.full((256, 256, 3), 0.02, dtype=np.float32)
    split[:, 128:] = 0.9
    nonuniform = scan(split).features

    assert nonuniform.nonuniformity > 2.0
    assert nonuniform.illumination_label == "strong"


def test_colour_cast_detection() -> None:
    warm = np.full((256, 256, 3), (0.60, 0.50, 0.40), dtype=np.float32)
    warm_result = scan(warm)

    assert warm_result.features.colour_cast_magnitude > 6.0
    assert warm_result.parameters.hue_lock_beta > 0.0
    assert warm_result.parameters.hue_lock_beta <= 0.5

    grey_result = scan(_uniform(0.5))
    assert grey_result.features.colour_cast_magnitude < 1.0
    assert grey_result.parameters.hue_lock_beta == 0.0


def test_result_shapes_and_analysis_downscale() -> None:
    small = scan(_uniform(0.3, 256))
    assert small.analysis_size == (256, 256)
    assert small.illumination.shape == (256, 256)
    assert small.noise_map.shape == (256, 256)
    assert small.illumination.dtype == np.float32
    assert small.noise_map.dtype == np.float32

    large = np.full((512, 640, 3), 0.3, dtype=np.float32)
    large_result = scan(large)
    assert large_result.analysis_size == (256, 205)
    assert large_result.noise_map.shape == (512, 640)
    assert large_result.illumination.shape == (205, 256)


def test_result_is_json_serialisable() -> None:
    payload = scan(_uniform(0.3)).as_dict()

    encoded = json.dumps(payload)

    assert "features" in encoded
    assert "parameters" in encoded


def test_config_can_be_overridden() -> None:
    config = LumenScanConfig(target_brightness=0.6, gamma_min=0.2)

    result = scan(_uniform(0.1), config=config)

    assert result.parameters.output_blend_alpha == pytest.approx(1.0)
    assert result.parameters.gamma >= 0.2


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        scan(np.zeros((32, 32), dtype=np.float32))
    with pytest.raises(ValueError):
        scan(np.full((32, 32, 3), 2.0, dtype=np.float32))
