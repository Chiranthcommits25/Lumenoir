"""Split generation: exact counts, determinism and leakage filtering."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from training.data_pipeline.config import DATASET_SPECS, DATASET_SPECS_BY_KEY, SplitRules
from training.data_pipeline.organisation import DatasetScan, ImageRecord
from training.data_pipeline.splits import (
    SplitSpecMismatch,
    build_split_spec,
    build_splits,
    write_split_spec,
)


def _record(dataset: str, group: str, index: int, scene: str | None = None) -> ImageRecord:
    return ImageRecord(
        record_id=f"{dataset}:{group}:{index:05d}",
        dataset=dataset,
        group=group,
        low_path=Path(f"/tmp/{dataset}/{group}/{index:05d}.png"),
        high_path=None,
        scene=scene,
        has_gt=False,
    )


def _scan(dataset: str) -> DatasetScan:
    spec = DATASET_SPECS_BY_KEY[dataset]
    records: list[ImageRecord] = []
    for group in spec.groups:
        for index in range(group.expected_count or 0):
            scene = f"class{index % 10}" if dataset == "exdark" else None
            records.append(_record(dataset, group.name, index, scene))
    counts = Counter(record.group for record in records)
    return DatasetScan(
        spec=spec,
        root=Path("/tmp"),
        records=tuple(records),
        group_counts={group.name: counts.get(group.name, 0) for group in spec.groups},
        missing_groups=(),
        problems=(),
    )


@pytest.fixture(scope="module")
def full_scans() -> list[DatasetScan]:
    return [_scan(spec.key) for spec in DATASET_SPECS]


def test_pool_totals_match_the_specification(full_scans: list[DatasetScan]) -> None:
    bundle = build_splits(full_scans, detect_leakage=False)

    assert bundle.totals() == {
        "fine_tune": 3496,
        "validation": 800,
        "test": 415,
        "unseen": 459,
        "do_no_harm": 100,
    }


def test_per_dataset_composition(full_scans: list[DatasetScan]) -> None:
    counts = build_splits(full_scans, detect_leakage=False).counts()

    assert counts["fine_tune"] == {"lol_v1": 485, "lol_v2_real": 589, "sice": 2422}
    assert counts["validation"] == {"lol_v2_real": 100, "lol_v2_syn": 100, "sice": 600}
    assert counts["test"] == {
        "lol_v1": 15,
        "lol_v2_real": 100,
        "lol_v2_syn": 100,
        "sice": 200,
    }
    assert counts["unseen"] == {
        "dicm": 64,
        "exdark": 200,
        "lime": 10,
        "mef": 17,
        "npe": 84,
        "own": 60,
        "vv": 24,
    }
    assert counts["do_no_harm"] == {"sice": 100}


def test_splits_are_deterministic(full_scans: list[DatasetScan]) -> None:
    first = build_splits(full_scans, detect_leakage=False)
    second = build_splits(full_scans, detect_leakage=False)

    for pool in first.splits:
        assert [record.record_id for record in first.splits[pool]] == [
            record.record_id for record in second.splits[pool]
        ]


def test_pools_are_disjoint(full_scans: list[DatasetScan]) -> None:
    bundle = build_splits(full_scans, detect_leakage=False)

    seen: dict[str, str] = {}
    for pool, records in bundle.splits.items():
        for record in records:
            assert record.record_id not in seen
            seen[record.record_id] = pool


def test_count_mismatch_raises(full_scans: list[DatasetScan]) -> None:
    broken = [
        replace(scan, records=scan.records[1:], group_counts={"train": 484})
        if scan.spec.key == "lol_v1"
        else scan
        for scan in full_scans
    ]

    with pytest.raises(SplitSpecMismatch, match="lol_v1/train"):
        build_splits(broken, detect_leakage=False)


def test_count_mismatch_can_be_recorded(full_scans: list[DatasetScan]) -> None:
    broken = [
        replace(scan, records=scan.records[1:], group_counts={"train": 484})
        if scan.spec.key == "lol_v1"
        else scan
        for scan in full_scans
    ]

    bundle = build_splits(broken, detect_leakage=False, allow_count_mismatch=True)

    assert any(mismatch.dataset == "lol_v1" for mismatch in bundle.count_mismatches)
    assert any("explicitly allowed" in note for note in bundle.notes)


def test_split_spec_needs_no_data(tmp_path: Path) -> None:
    spec = build_split_spec()

    assert spec["seed"] == 42
    assert spec["pools"]["fine_tune"]["base_total"] == 3496
    assert spec["pools"]["test"]["composition"]["lol_v1/test"] == 15
    assert spec["pools"]["unseen"]["total"] == 459

    path = write_split_spec(tmp_path)
    assert path.is_file()


def _blocky(seed: int, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    return cv2.resize(small, (size, size), interpolation=cv2.INTER_NEAREST)


def _write_image(root: Path, dataset: str, group: str, name: str, seed: int) -> Path:
    path = root / dataset / group / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_blocky(seed)).save(path)
    return path


def _small_rules() -> SplitRules:
    return SplitRules(
        sice_part1_total=4,
        sice_part1_fine_tune=2,
        sice_part1_validation=2,
        lol_v1_train=2,
        lol_v1_test=1,
        lol_v2_real_train=1,
        lol_v2_real_test=1,
        lol_v2_real_validation=0,
        lol_v2_syn_train=1,
        lol_v2_syn_test=1,
        lol_v2_syn_validation=0,
        sice_part2_lowlight_total=2,
        sice_part2_test=1,
        exdark_total=1,
        exdark_per_class=1,
        exdark_classes=1,
        exdark_unseen=1,
        own_groups=("extremely_dark",),
        own_per_group=1,
        do_no_harm=1,
        unseen_counts={"dicm": 1},
    )


def _small_scan(dataset: str, records: list[ImageRecord]) -> DatasetScan:
    spec = DATASET_SPECS_BY_KEY[dataset]
    counts = Counter(record.group for record in records)
    return DatasetScan(
        spec=spec,
        root=Path("/tmp"),
        records=tuple(records),
        group_counts={group.name: counts.get(group.name, 0) for group in spec.groups},
        missing_groups=(),
        problems=(),
    )


def _file_record(dataset: str, group: str, path: Path, name: str) -> ImageRecord:
    return ImageRecord(
        record_id=f"{dataset}:{group}:{name}",
        dataset=dataset,
        group=group,
        low_path=path,
        high_path=None,
        scene="class0" if dataset == "exdark" else None,
        has_gt=False,
    )


def _image_record(
    root: Path, dataset: str, group: str, name: str, seed: int
) -> ImageRecord:
    return _file_record(dataset, group, _write_image(root, dataset, group, name, seed), name)


def test_leakage_is_removed_and_tests_are_kept(tmp_path: Path) -> None:
    test_image = _write_image(tmp_path, "lol_v1", "test", "test", seed=20)
    leaked_copy = _write_image(tmp_path, "lol_v1", "train", "leaked", seed=20)

    sice_records = [
        _image_record(tmp_path, "sice", "part1", f"p{index}", 100 + index)
        for index in range(4)
    ]
    sice_records += [
        _image_record(tmp_path, "sice", "part2_lowlight", f"q{index}", 110 + index)
        for index in range(2)
    ]
    sice_records.append(_image_record(tmp_path, "sice", "do_no_harm", "n0", 120))

    scans = [
        _small_scan("sice", sice_records),
        _small_scan(
            "lol_v1",
            [
                _file_record("lol_v1", "train", leaked_copy, "leaked"),
                _image_record(tmp_path, "lol_v1", "train", "clean", 21),
                _file_record("lol_v1", "test", test_image, "test"),
            ],
        ),
        _small_scan(
            "lol_v2_real",
            [
                _image_record(tmp_path, "lol_v2_real", "train", "r0", 130),
                _image_record(tmp_path, "lol_v2_real", "test", "r1", 131),
            ],
        ),
        _small_scan(
            "lol_v2_syn",
            [
                _image_record(tmp_path, "lol_v2_syn", "train", "s0", 140),
                _image_record(tmp_path, "lol_v2_syn", "test", "s1", 141),
            ],
        ),
        _small_scan("dicm", [_image_record(tmp_path, "dicm", "all", "d0", 150)]),
        _small_scan(
            "exdark",
            [
                _file_record(
                    "exdark",
                    "all",
                    _write_image(tmp_path, "exdark", "class0", "e0", 160),
                    "e0",
                )
            ],
        ),
        _small_scan(
            "own",
            [_image_record(tmp_path, "own", "extremely_dark", "o0", 170)],
        ),
    ]

    bundle = build_splits(scans, _small_rules())

    fine_ids = {record.record_id for record in bundle.splits["fine_tune"]}
    test_ids = {record.record_id for record in bundle.splits["test"]}

    assert "lol_v1:train:leaked" not in fine_ids
    assert "lol_v1:train:clean" in fine_ids
    assert "lol_v1:test:test" in test_ids
    assert len(bundle.leakage) == 1
    assert len(bundle.splits["fine_tune"]) == 4
    assert len(bundle.splits["test"]) == 4
