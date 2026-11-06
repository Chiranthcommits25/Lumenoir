"""64-bit perceptual hashes and the exact Hamming index."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from training.data_pipeline.hashing import (
    HammingIndex,
    brute_force_pairs,
    hamming,
    pairs_within,
    phash64,
    phash64_from_array,
)


def _blocky(seed: int = 0, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    return cv2.resize(small, (size, size), interpolation=cv2.INTER_NEAREST)


def _save(tmp_path: Path, name: str, array: np.ndarray, **kwargs) -> Path:
    path = tmp_path / name
    Image.fromarray(array).save(path, **kwargs)
    return path


def test_identical_images_hash_to_zero_distance(tmp_path: Path) -> None:
    array = _blocky()
    left = _save(tmp_path, "a.png", array)
    right = _save(tmp_path, "b.png", array)

    assert hamming(phash64(left), phash64(right)) == 0


def test_resized_and_recompressed_copies_are_near_duplicates(tmp_path: Path) -> None:
    array = _blocky(seed=3)
    original = _save(tmp_path, "original.png", array)

    bigger = cv2.resize(array, (128, 128), interpolation=cv2.INTER_NEAREST)
    resized = _save(tmp_path, "resized.png", bigger)
    jpeg = _save(tmp_path, "copy.jpg", array, quality=90)

    assert hamming(phash64(original), phash64(resized)) <= 8
    assert hamming(phash64(original), phash64(jpeg)) <= 8


def test_hash_is_64_bit() -> None:
    value = phash64_from_array(_blocky())

    assert 0 <= value < 2**64


def test_index_matches_brute_force() -> None:
    rng = random.Random(7)
    values = [rng.getrandbits(64) for _ in range(150)]

    index = HammingIndex()
    index.add_many((value, str(position)) for position, value in enumerate(values))

    found: set[tuple[int, int, int]] = set()
    for position, value in enumerate(values):
        for item_id, distance in index.query(value):
            other = int(item_id)
            if other == position:
                continue
            found.add((min(position, other), max(position, other), distance))

    expected = set(brute_force_pairs(values))
    assert found == expected


def test_pairs_within_returns_identifier_pairs() -> None:
    values = [(str(index), value) for index, value in enumerate([0, 0b111, 0b11111111, 1 << 40])]

    pairs = pairs_within(values, max_distance=3)

    assert ("0", "1", 3) in pairs
    assert all(distance <= 3 for _, _, distance in pairs)


def test_distance_boundary_is_exact() -> None:
    index = HammingIndex()
    index.add(0, "zero")
    index.add(0b11111111, "eight-bits")
    index.add(0b111111111, "nine-bits")

    found = dict(index.query(0, max_distance=8))

    assert found["zero"] == 0
    assert found["eight-bits"] == 8
    assert "nine-bits" not in found
