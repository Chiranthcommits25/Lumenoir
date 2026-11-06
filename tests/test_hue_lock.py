"""HueLock: guarded LAB colour-cast correction."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.classical_cv.colour import MidToneLabStats, detect_colour_cast
from backend.classical_cv.hue_lock import HueLockConfig, correct
from backend.classical_cv.scan import scan


def _uniform(colour: tuple[float, float, float], size: int = 128) -> np.ndarray:
    return np.full((size, size, 3), colour, dtype=np.float32)


def _grey(level: float = 0.5, size: int = 128) -> np.ndarray:
    return _uniform((level, level, level), size)


def _lab(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)


def _cast(image: np.ndarray) -> float:
    scan_result = scan(image)
    return detect_colour_cast(image, p50=scan_result.features.p50).magnitude


def test_neutral_image_is_returned_unchanged() -> None:
    image = _grey(0.5)

    result = correct(image, scan_result=scan(image))

    assert result.cast_magnitude < 0.1
    assert result.beta_raw == 0.0
    assert result.beta_applied == 0.0
    assert np.array_equal(result.image, image)


def test_warm_cast_is_reduced() -> None:
    image = _uniform((0.60, 0.50, 0.40))
    before = _cast(image)

    result = correct(image, scan_result=scan(image))

    assert before > 6.0
    assert result.beta_raw > 0.0
    assert result.beta_applied > 0.0
    assert _cast(result.image) < before
    assert result.image.shape == image.shape
    assert result.image.min() >= 0.0
    assert result.image.max() <= 1.0


def test_shift_is_capped_at_eight_lab_units() -> None:
    image = _uniform((0.60, 0.50, 0.40))
    scan_result = scan(image)

    result = correct(image, scan_result=scan_result)

    assert result.cast_magnitude > 8.0
    assert result.shift_magnitude == pytest.approx(8.0, abs=1e-4)
    assert result.beta_applied == pytest.approx(8.0 / result.cast_magnitude, rel=1e-4)


def test_beta_raw_comes_from_lumen_scan() -> None:
    image = _uniform((0.62, 0.52, 0.42))
    scan_result = scan(image)

    result = correct(image, scan_result=scan_result)

    assert result.beta_raw == pytest.approx(scan_result.parameters.hue_lock_beta)
    assert result.cast_a == pytest.approx(scan_result.features.colour_cast_a)
    assert result.cast_b == pytest.approx(scan_result.features.colour_cast_b)


def test_colourful_scene_halves_beta(monkeypatch) -> None:
    image = _uniform((0.60, 0.50, 0.40))
    scan_result = scan(image)

    def fake_stats(_image, *, p50, **_kwargs):
        return MidToneLabStats(
            mean_a=0.0, mean_b=0.0, std_a=20.0, std_b=5.0, pixel_fraction=1.0
        )

    monkeypatch.setattr("backend.classical_cv.hue_lock.mid_tone_lab_stats", fake_stats)

    result = correct(image, scan_result=scan_result)

    assert result.spread_guarded is True
    expected = min(result.beta_raw / 2.0, 8.0 / result.cast_magnitude)
    assert result.beta_applied == pytest.approx(expected)


def test_high_spread_scene_is_detected_even_without_a_cast() -> None:
    image = np.zeros((128, 128, 3), dtype=np.float32)
    image[:, :64] = (0.60, 0.50, 0.40)
    image[:, 64:] = (0.40, 0.50, 0.60)

    result = correct(image, scan_result=scan(image))

    assert result.spread > 15.0
    assert result.spread_guarded is True
    assert result.beta_raw == 0.0
    assert np.array_equal(result.image, image)


def test_lightness_is_preserved() -> None:
    base = np.zeros((128, 128, 3), dtype=np.float32)
    base[..., 0] = np.linspace(0.45, 0.70, 128, dtype=np.float32)[None, :]
    base[..., 1] = np.linspace(0.40, 0.55, 128, dtype=np.float32)[None, :]
    base[..., 2] = 0.35

    result = correct(base, scan_result=scan(base))

    before = _lab(base)[..., 0]
    after = _lab(result.image)[..., 0]
    assert float(np.abs(before - after).max()) < 0.5


def test_chroma_is_not_boosted() -> None:
    image = _uniform((0.60, 0.50, 0.40))

    result = correct(image, scan_result=scan(image))

    before = _lab(image)
    after = _lab(result.image)
    chroma_before = np.hypot(before[..., 1], before[..., 2])
    chroma_after = np.hypot(after[..., 1], after[..., 2])
    assert float(chroma_after.mean()) <= float(chroma_before.mean()) + 0.5


def test_highlights_and_shadows_keep_their_lightness() -> None:
    image = _uniform((0.60, 0.50, 0.40))
    image[0:32, 0:32] = 0.05
    image[96:128, 96:128] = 0.95

    result = correct(image, scan_result=scan(image))

    before = _lab(image)[..., 0]
    after = _lab(result.image)[..., 0]
    assert abs(float(after[0:32, 0:32].mean()) - float(before[0:32, 0:32].mean())) < 0.5
    assert abs(float(after[96:128, 96:128].mean()) - float(before[96:128, 96:128].mean())) < 0.5


def test_scan_result_is_computed_when_omitted() -> None:
    image = _uniform((0.62, 0.52, 0.42))

    implicit = correct(image)
    explicit = correct(image, scan_result=scan(image))

    assert np.allclose(implicit.image, explicit.image, atol=1e-6)
    assert implicit.beta_applied == pytest.approx(explicit.beta_applied)


def test_config_can_disable_the_spread_guard(monkeypatch) -> None:
    image = _uniform((0.60, 0.50, 0.40))
    scan_result = scan(image)

    monkeypatch.setattr(
        "backend.classical_cv.hue_lock.mid_tone_lab_stats",
        lambda _image, *, p50, **_kwargs: MidToneLabStats(0, 0, 20, 5, 1.0),
    )

    guarded = correct(image, scan_result=scan_result)
    unguarded = correct(
        image,
        scan_result=scan_result,
        config=HueLockConfig(halve_on_spread=False),
    )

    assert guarded.beta_applied < unguarded.beta_applied


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        correct(np.zeros((32, 32), dtype=np.float32))
    with pytest.raises(ValueError):
        correct(np.full((32, 32, 3), 2.0, dtype=np.float32))
