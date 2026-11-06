"""Duplicate detection and cross-dataset leakage."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from training.data_pipeline.leakage import (
    HashedRecord,
    deduplicate,
    find_leakage,
    hash_records,
    ssim_similarity,
)
from training.data_pipeline.organisation import ImageRecord


def _blocky(seed: int, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    return cv2.resize(small, (size, size), interpolation=cv2.INTER_NEAREST)


def _save(tmp_path: Path, name: str, array: np.ndarray) -> Path:
    path = tmp_path / name
    Image.fromarray(array).save(path)
    return path


def _record(dataset: str, group: str, path: Path, name: str) -> ImageRecord:
    return ImageRecord(
        record_id=f"{dataset}:{group}:{name}",
        dataset=dataset,
        group=group,
        low_path=path,
        high_path=None,
        scene=None,
        has_gt=False,
    )


def test_ssim_identical_and_different(tmp_path: Path) -> None:
    array = _blocky(seed=1)
    left = _save(tmp_path, "left.png", array)
    right = _save(tmp_path, "right.png", array)

    noise = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    different = _save(tmp_path, "different.png", noise)

    assert ssim_similarity(left, right) == 1.0
    assert ssim_similarity(left, different) < 0.9


def test_exact_copy_across_datasets_is_leakage(tmp_path: Path) -> None:
    array = _blocky(seed=5)
    test_path = _save(tmp_path, "test.png", array)
    leaked_path = _save(tmp_path, "leaked.png", array)
    clean_path = _save(tmp_path, "clean.png", _blocky(seed=6))

    test = _record("lol_v2_real", "test", test_path, "test")
    leaked = _record("lol_v1", "train", leaked_path, "leaked")
    clean = _record("sice", "part1", clean_path, "clean")

    test_hashes = hash_records([test])
    train_hashes = hash_records([leaked, clean])

    report = find_leakage(train_hashes.hashed, test_hashes.hashed)

    dropped = {match.dropped_id for match in report.matches}
    assert leaked.record_id in dropped
    assert clean.record_id not in dropped
    assert all(match.kept_id == test.record_id for match in report.matches)
    assert report.candidates_reviewed >= 1


def test_close_hash_but_different_pixels_is_kept(tmp_path: Path) -> None:
    test_path = _save(tmp_path, "test.png", _blocky(seed=7))
    other_path = _save(tmp_path, "other.png", _blocky(seed=8))

    test = _record("lol_v1", "test", test_path, "test")
    other = _record("sice", "part1", other_path, "other")

    # Same 64-bit hash on purpose: the SSIM pass must reject the false positive.
    report = find_leakage(
        [HashedRecord(record=other, phash=0)],
        [HashedRecord(record=test, phash=0)],
    )

    assert report.matches == ()
    assert report.candidates_reviewed == 1


def test_deduplicate_keeps_the_priority_dataset(tmp_path: Path) -> None:
    array = _blocky(seed=9)
    first = _save(tmp_path, "sice.png", array)
    second = _save(tmp_path, "lol.png", array)

    sice = _record("sice", "part1", first, "sice")
    lol = _record("lol_v1", "train", second, "lol")

    hashed = hash_records([sice, lol]).hashed
    report = deduplicate(hashed, priority={"lol_v1": 0, "sice": 1})

    assert [record.dataset for record in report.kept] == ["lol_v1"]
    assert report.matches[0].dropped_id == sice.record_id
    assert report.matches[0].kept_id == lol.record_id
