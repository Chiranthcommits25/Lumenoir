"""Deterministic train/validation split and leakage tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from training.data_pipeline.validation_split import (
    DatasetLayout,
    build_split,
    write_outputs,
)
from training.experiments.stage_attribution import load_validation_pairs


def _image(path: Path, seed: int, size: tuple[int, int] = (32, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(array).save(path)


def _layout(root: Path) -> DatasetLayout:
    for index in range(10):
        _image(root / "LOL-v1" / "our485" / "low" / f"{index}.png", index)
        _image(root / "LOL-v1" / "our485" / "high" / f"{index}.png", index + 100)
    # One test image is a duplicate of a training image: it must be excluded
    # from train/validation.
    _image(root / "LOL-v1" / "eval15" / "low" / "0.png", 0)
    _image(root / "LOL-v1" / "eval15" / "high" / "0.png", 100)
    return DatasetLayout(
        key="lol_v1",
        label="LOL-v1",
        root="LOL-v1",
        low="our485/low",
        high="our485/high",
        expected_pairs=10,
        test_low="eval15/low",
        test_high="eval15/high",
        expected_test_pairs=1,
    )


def test_split_groups_pairs_and_excludes_test_leakage(tmp_path) -> None:
    layout = _layout(tmp_path)
    outcome = build_split(tmp_path, layouts=(layout,))

    count = outcome.counts["lol_v1"]
    assert count["source_pairs"] == 10
    assert count["eligible_pairs"] == 9
    assert count["train"] == 7
    assert count["validation"] == 2
    assert outcome.test_overlap == 0
    assert outcome.train_validation_overlap == 0
    assert outcome.exclusions
    assert outcome.exclusions[0]["record_id"].startswith("lol_v1:")
    assert len(outcome.train) == 7
    assert len(outcome.validation) == 2
    assert {pair.record_id for pair in outcome.train}.isdisjoint(
        {pair.record_id for pair in outcome.validation}
    )


def test_split_outputs_are_deterministic_and_stage_attribution_compatible(tmp_path) -> None:
    layout = _layout(tmp_path)
    first = build_split(tmp_path, layouts=(layout,))
    second = build_split(tmp_path, layouts=(layout,))
    assert [pair.record_id for pair in first.train] == [
        pair.record_id for pair in second.train
    ]
    assert [pair.record_id for pair in first.validation] == [
        pair.record_id for pair in second.validation
    ]

    output = tmp_path / "splits"
    paths = write_outputs(first, output, data_root=tmp_path)
    assert paths
    validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    assert validation["pool"] == "validation"
    assert all(record["split"] == "validation" for record in validation["records"])
    assert all(record["kind"] == "paired" for record in validation["records"])
    assert validation["records"][0]["low"].startswith("LOL-v1/")
    pairs, _ = load_validation_pairs(output / "validation.json", data_root=tmp_path)
    assert len(pairs) == 2


def test_split_does_not_overwrite_outputs_without_force(tmp_path) -> None:
    layout = _layout(tmp_path)
    outcome = build_split(tmp_path, layouts=(layout,))
    output = tmp_path / "splits"
    write_outputs(outcome, output, data_root=tmp_path)

    with pytest.raises(FileExistsError):
        write_outputs(outcome, output, data_root=tmp_path)

    write_outputs(outcome, output, data_root=tmp_path, overwrite=True)
    assert (output / "split_report.md").is_file()
