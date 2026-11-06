"""Decoding, EXIF handling, resizing, cropping and augmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from training.data_pipeline.preprocessing import (
    PreprocessingError,
    apply_darkening,
    apply_orientation,
    augment_views,
    load_rgb_float32,
    random_crop,
    random_crop_pair,
    resize_long_edge,
)


def _save(tmp_path: Path, name: str, array: np.ndarray, **kwargs) -> Path:
    path = tmp_path / name
    Image.fromarray(array).save(path, **kwargs)
    return path


def test_bgr_to_rgb_conversion(tmp_path: Path) -> None:
    red = np.zeros((32, 32, 3), dtype=np.uint8)
    red[..., 0] = 255
    path = _save(tmp_path, "red.png", red)

    loaded = load_rgb_float32(path)

    assert loaded.dtype == np.float32
    assert loaded.shape == (32, 32, 3)
    assert loaded[0, 0, 0] == pytest.approx(1.0)
    assert loaded[0, 0, 1] == pytest.approx(0.0)
    assert loaded[0, 0, 2] == pytest.approx(0.0)
    assert 0.0 <= loaded.min() and loaded.max() <= 1.0


def test_exif_orientation_is_applied_once(tmp_path: Path) -> None:
    array = np.zeros((20, 40, 3), dtype=np.uint8)
    array[..., 0] = 200
    exif = Image.Exif()
    exif[274] = 6
    path = _save(tmp_path, "rotated.jpg", array, format="JPEG", exif=exif)

    loaded = load_rgb_float32(path)

    assert loaded.shape == (40, 20, 3)


def test_orientation_values() -> None:
    array = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)

    assert np.array_equal(apply_orientation(array, 1), array)
    assert np.array_equal(apply_orientation(array, 6), np.rot90(array, k=-1))
    assert np.array_equal(apply_orientation(array, 8), np.rot90(array, k=1))
    assert np.array_equal(apply_orientation(array, 2), array[:, ::-1])


def test_resize_long_edge_never_upscales() -> None:
    large = np.zeros((1200, 2400, 3), dtype=np.float32)
    resized = resize_long_edge(large, 1200)
    assert resized.shape == (600, 1200, 3)

    small = np.zeros((200, 300, 3), dtype=np.float32)
    assert resize_long_edge(small, 1200) is small


def test_random_crop_is_seeded_and_bounded() -> None:
    image = np.arange(500 * 500 * 3, dtype=np.float32).reshape(500, 500, 3)

    first = random_crop(image, 384, np.random.default_rng(0))
    second = random_crop(image, 384, np.random.default_rng(0))

    assert first.shape == (384, 384, 3)
    assert np.array_equal(first, second)


def test_random_crop_rejects_small_images() -> None:
    with pytest.raises(PreprocessingError):
        random_crop(np.zeros((100, 100, 3), dtype=np.float32), 384)


def test_pair_crop_uses_the_same_coordinates() -> None:
    size = 384
    low = np.zeros((500, 500, 3), dtype=np.float32)
    rows = np.arange(500, dtype=np.float32)[:, None]
    cols = np.arange(500, dtype=np.float32)[None, :]
    high = np.stack(
        [
            np.broadcast_to(rows, (500, 500)),
            np.broadcast_to(cols, (500, 500)),
            np.zeros((500, 500), dtype=np.float32),
        ],
        axis=2,
    )

    low_crop, high_crop = random_crop_pair(low, high, size, np.random.default_rng(4))
    top = round(float(high_crop[0, 0, 0]))
    left = round(float(high_crop[0, 0, 1]))

    assert np.array_equal(low_crop, low[top : top + size, left : left + size])
    assert np.array_equal(high_crop, high[top : top + size, left : left + size])


def test_pair_crop_rejects_mismatched_shapes() -> None:
    with pytest.raises(PreprocessingError):
        random_crop_pair(
            np.zeros((100, 100, 3), dtype=np.float32),
            np.zeros((100, 90, 3), dtype=np.float32),
            64,
        )


def test_augmentation_has_eight_views() -> None:
    image = np.random.default_rng(0).random((17, 23, 3)).astype(np.float32)

    views = augment_views(image)

    assert len(views) == 8
    assert np.array_equal(views[0], image)
    fingerprints = {view.tobytes() for view in views}
    assert len(fingerprints) == 8


def test_darkening_probability_bounds() -> None:
    image = np.full((32, 32, 3), 0.6, dtype=np.float32)

    unchanged = apply_darkening(image.copy(), np.random.default_rng(1), probability=0.0)
    darkened = apply_darkening(image.copy(), np.random.default_rng(1), probability=1.0)

    assert np.array_equal(unchanged, image)
    assert darkened.mean() < image.mean() * 0.9
    assert darkened.dtype == np.float32
    assert darkened.min() >= 0.0 and darkened.max() <= 1.0

    with pytest.raises(PreprocessingError):
        apply_darkening(image, np.random.default_rng(1), probability=1.5)
