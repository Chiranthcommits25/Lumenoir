"""Corruption, truncation and format validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from training.data_pipeline.organisation import ImageRecord
from training.data_pipeline.validation import (
    EMPTY_FILE,
    EXTENSION_MISMATCH,
    UNSUPPORTED_FORMAT,
    probe_image,
    validate_records,
)


def _array(size: tuple[int, int] = (32, 32), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)


def _save(tmp_path: Path, name: str, array: np.ndarray | None = None) -> Path:
    path = tmp_path / name
    Image.fromarray(array if array is not None else _array()).save(path)
    return path


def _record(path: Path, high: Path | None = None) -> ImageRecord:
    return ImageRecord(
        record_id="dataset:group:one",
        dataset="dataset",
        group="group",
        low_path=path,
        high_path=high,
        scene=None,
        has_gt=high is not None,
    )


def test_valid_image_probe(tmp_path: Path) -> None:
    probe = probe_image(_save(tmp_path, "ok.png"))

    assert probe.ok is True
    assert probe.code is None
    assert (probe.width, probe.height) == (32, 32)
    assert probe.channels == 3
    assert probe.detected_format == "PNG"
    assert probe.bit_depth == 8
    assert probe.size_bytes > 0


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.png"
    path.write_bytes(b"")

    probe = probe_image(path)

    assert probe.ok is False
    assert probe.code == EMPTY_FILE


def test_non_image_payload_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "fake.png"
    path.write_text("this is not an image", encoding="utf-8")

    probe = probe_image(path)

    assert probe.ok is False
    assert probe.code == UNSUPPORTED_FORMAT


def test_truncated_png_is_rejected(tmp_path: Path) -> None:
    path = _save(tmp_path, "truncated.png")
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])

    probe = probe_image(path)

    assert probe.ok is False
    assert probe.code in {"truncated_image", "corrupted_image", "unsupported_format"}


def test_extension_mismatch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "really_png.jpg"
    Image.fromarray(_array()).save(path, format="PNG")

    probe = probe_image(path)

    assert probe.ok is False
    assert probe.code == EXTENSION_MISMATCH


def test_missing_file_is_reported(tmp_path: Path) -> None:
    record = _record(tmp_path / "does-not-exist.png")

    outcome = validate_records([record])

    assert outcome.valid == ()
    assert outcome.problems[0].code == "missing_file"
    assert outcome.problems[0].is_reference is False


def test_invalid_records_are_excluded_but_not_deleted(tmp_path: Path) -> None:
    good = _save(tmp_path, "good.png", _array(seed=1))
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"broken")
    reference = _save(tmp_path, "ref.png", _array(seed=2))

    outcome = validate_records(
        [_record(good, reference), _record(bad, reference)]
    )

    assert len(outcome.valid) == 1
    assert outcome.valid[0].low_path == good
    assert len(outcome.problems) == 1
    assert outcome.problems[0].path == bad
    assert bad.exists(), "validation must never delete a raw image"
    assert good.exists()


def test_corrupt_reference_is_reported(tmp_path: Path) -> None:
    low = _save(tmp_path, "low.png")
    high = tmp_path / "high.png"
    high.write_bytes(b"not an image")

    outcome = validate_records([_record(low, high)])

    assert outcome.valid == ()
    assert any(problem.is_reference for problem in outcome.problems)
