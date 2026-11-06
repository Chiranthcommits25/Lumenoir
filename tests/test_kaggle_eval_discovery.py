"""Kaggle dataset discovery and manifest tests."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from training.kaggle_eval.config import (
    DatasetKind,
    DatasetOverride,
    EvalConfig,
    PairLayout,
)
from training.kaggle_eval.discovery import discover, write_manifests


def _image(path, seed: int = 0, size: tuple[int, int] = (64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(array).save(path)


def _build_tree(root) -> None:
    for i in range(2):
        _image(root / "lol-v1" / "our485" / "low" / f"train_{i}.png", seed=10 + i)
        _image(root / "lol-v1" / "our485" / "high" / f"train_{i}.png", seed=10 + i)
    for i in range(2):
        _image(root / "lol-v1" / "eval15" / "low" / f"test_{i}.png", seed=20 + i)
        _image(root / "lol-v1" / "eval15" / "high" / f"test_{i}.png", seed=20 + i)
    # missing reference
    _image(root / "lol-v1" / "eval15" / "low" / "orphan.png", seed=30)

    _image(root / "lol-v2-real" / "Real_captured" / "Train" / "low" / "train.png", seed=40)
    _image(root / "lol-v2-real" / "Real_captured" / "Train" / "high" / "train.png", seed=40)
    _image(root / "lol-v2-real" / "Real_captured" / "Test" / "low" / "test.png", seed=41)
    _image(root / "lol-v2-real" / "Real_captured" / "Test" / "high" / "test.png", seed=41)

    for i in range(2):
        _image(root / "exdark" / f"class{i}" / f"img_{i}.png", seed=50 + i)
    _image(root / "lime" / "img.png", seed=60)

    # SICE with no valid low/high pairs: it must be skipped, not guessed.
    _image(root / "sice" / "part1" / "scene1" / "1.png", seed=70)


def _config(tmp_path) -> EvalConfig:
    return EvalConfig(
        input_root=tmp_path / "input",
        additional_roots=(),
        output_dir=tmp_path / "out",
        modes=("original", "classical"),
        seed=7,
        skip_metrics=("lpips", "niqe"),
    )


def test_auto_discovery_keeps_train_and_test_splits_separate(tmp_path) -> None:
    _build_tree(tmp_path / "input")
    result = discover(_config(tmp_path))
    manifests = {manifest.key: manifest for manifest in result.manifests}

    lol_v1 = manifests["lol_v1"]
    assert lol_v1.kind is DatasetKind.PAIRED
    assert len(lol_v1.records) == 4
    assert lol_v1.split_counts == {"test": 2, "train": 2}
    assert lol_v1.evaluation_count == 2
    assert lol_v1.training_count == 2
    assert any("eval15" in record.image_id for record in lol_v1.evaluation_records)
    assert not any("our485" in record.image_id for record in lol_v1.evaluation_records)
    assert any("orphan" in warning or "no matching" in warning for warning in lol_v1.warnings)

    lol_v2 = manifests["lol_v2_real"]
    assert len(lol_v2.records) == 2
    assert lol_v2.split_counts == {"test": 1, "train": 1}
    assert lol_v2.evaluation_count == 1
    assert all("Test" in record.image_id for record in lol_v2.evaluation_records)
    assert not any("Train" in record.image_id for record in lol_v2.evaluation_records)

    assert len(manifests["exdark"].records) == 2
    assert manifests["exdark"].kind is DatasetKind.UNPAIRED
    assert len(manifests["lime"].records) == 1
    assert manifests["lime"].evaluation_count == 1


def test_sice_without_reference_pairs_is_skipped(tmp_path) -> None:
    _build_tree(tmp_path / "input")
    result = discover(_config(tmp_path))
    manifests = {manifest.key: manifest for manifest in result.manifests}

    sice = manifests["sice_p1"]
    assert sice.records == ()
    assert sice.skipped_reason is not None


def test_explicit_override_wins(tmp_path) -> None:
    custom = tmp_path / "input" / "custom-name"
    _image(custom / "lows" / "a.png", seed=1)
    _image(custom / "highs" / "a.png", seed=1)
    config = _config(tmp_path)
    config = config.replace(
        datasets={
            "lol_v1": DatasetOverride(
                root=str(custom),
                pairs=(PairLayout(low="lows", high="highs", split="custom"),),
            )
        }
    )

    result = discover(config)
    manifest = next(item for item in result.manifests if item.key == "lol_v1")

    assert len(manifest.records) == 1
    assert manifest.records[0].split == "custom"


def test_discovery_is_deterministic_and_manifests_are_written(tmp_path) -> None:
    _build_tree(tmp_path / "input")
    config = _config(tmp_path)

    first = discover(config)
    second = discover(config)
    assert first.to_dict() == second.to_dict()

    paths = write_manifests(first, config.output_dir, config=config)

    assert all(path.is_file() for path in paths)
    assert (config.output_dir / "manifests" / "manifest.json").is_file()
    assert (config.output_dir / "manifests" / "lol_v1.json").is_file()


def test_max_images_is_deterministic(tmp_path) -> None:
    _build_tree(tmp_path / "input")
    config = _config(tmp_path).replace(max_images=1)

    first = discover(config)
    second = discover(config)

    assert first.to_dict() == second.to_dict()
    for manifest in first.manifests:
        assert len(manifest.records) <= 1


def _build_additional_tree(root) -> None:
    _image(root / "LIME" / "data" / "LIME" / "lime_1.bmp", seed=101)
    _image(root / "LIME" / "data" / "LIME" / "lime_2.BMP", seed=102)
    _image(root / "LIME" / "data" / "LIME" / "ignored.webp", seed=103)

    _image(root / "MEF" / "data" / "MEF" / "mef_1.png", seed=110)
    _image(root / "MEF" / "data" / "MEF" / "mef_2.jpg", seed=111)

    _image(root / "NPE" / "data" / "NPE" / "npe_1.jpg", seed=120)
    _image(root / "NPE-ex1" / "data" / "NPE-ex1" / "ex1_1.jpeg", seed=121)
    _image(root / "NPE-ex1" / "data" / "NPE-ex1" / "ex1_2.png", seed=122)
    _image(root / "NPE-ex2" / "data" / "NPE-ex2" / "ex2_1.bmp", seed=123)
    _image(root / "NPE-ex3" / "data" / "NPE-ex3" / "ex3_1.png", seed=124)

    _image(root / "VV" / "data" / "VV" / "vv_1.jpg", seed=130)


def _additional_config(tmp_path) -> EvalConfig:
    return EvalConfig(
        input_root=tmp_path / "input",
        additional_roots=(tmp_path / "additional" / "extracted",),
        output_dir=tmp_path / "out",
        modes=("original",),
        seed=7,
        skip_metrics=("lpips", "niqe"),
    )


def test_additional_nested_datasets_all_formats_and_npe_grouping(tmp_path) -> None:
    additional = tmp_path / "additional" / "extracted"
    _build_additional_tree(additional)
    config = _additional_config(tmp_path)

    result = discover(config)
    manifests = {manifest.key: manifest for manifest in result.manifests}

    assert result.additional_roots == (additional,)
    assert "npe_ex1" not in manifests

    lime = manifests["lime"]
    assert lime.kind is DatasetKind.UNPAIRED
    assert lime.present
    assert len(lime.records) == 2
    assert lime.format_counts == {"bmp": 2}
    assert all(record.high is None for record in lime.records)
    assert all("data/LIME" in str(record.low) for record in lime.records)
    assert any("unsupported extensions" in warning for warning in lime.warnings)

    mef = manifests["mef"]
    assert mef.kind is DatasetKind.UNPAIRED
    assert len(mef.records) == 2
    assert mef.format_counts == {"jpg": 1, "png": 1}
    assert all("data/MEF" in str(record.low) for record in mef.records)

    npe = manifests["npe"]
    assert npe.kind is DatasetKind.UNPAIRED
    assert npe.present
    assert len(npe.records) == 5
    assert npe.format_counts == {"bmp": 1, "jpeg": 1, "jpg": 1, "png": 2}
    assert set().union(
        lime.format_counts,
        mef.format_counts,
        npe.format_counts,
    ) >= {"jpg", "jpeg", "png", "bmp"}
    assert npe.subgroup_counts == {
        "npe": 1,
        "npe_ex1": 2,
        "npe_ex2": 1,
        "npe_ex3": 1,
    }
    assert len(npe.source_roots) == 4
    assert all(record.kind is DatasetKind.UNPAIRED for record in npe.records)
    assert {record.subgroup for record in npe.records} == {
        "npe",
        "npe_ex1",
        "npe_ex2",
        "npe_ex3",
    }
    assert all(record.image_id.startswith(f"{record.subgroup}/") for record in npe.records)

    vv = manifests["vv"]
    assert vv.kind is DatasetKind.UNPAIRED
    assert len(vv.records) == 1
    assert vv.format_counts == {"jpg": 1}
    assert all("data/VV" in str(record.low) for record in vv.records)


def test_npe_explicit_parent_and_manifest_counts(tmp_path) -> None:
    additional = tmp_path / "additional" / "extracted"
    _build_additional_tree(additional)
    config = _additional_config(tmp_path).replace(
        datasets={"npe": DatasetOverride(root=str(additional))}
    )

    result = discover(config)
    npe = next(manifest for manifest in result.manifests if manifest.key == "npe")
    assert len(npe.records) == 5
    assert len(npe.source_roots) == 4

    paths = write_manifests(result, config.output_dir, config=config)
    assert paths
    payload = json.loads(
        (config.output_dir / "manifests" / "manifest.json").read_text(encoding="utf-8")
    )
    npe_payload = next(
        dataset for dataset in payload["datasets"] if dataset["dataset"] == "npe"
    )
    assert npe_payload["image_count"] == 5
    assert npe_payload["subgroup_counts"]["npe_ex3"] == 1
    assert len(npe_payload["records"]) == 5
    assert npe_payload["format_counts"]["jpeg"] == 1


def test_additional_discovery_is_deterministic(tmp_path) -> None:
    additional = tmp_path / "additional" / "extracted"
    _build_additional_tree(additional)
    config = _additional_config(tmp_path)

    assert discover(config).to_dict() == discover(config).to_dict()


def test_additional_roots_config_round_trip(tmp_path) -> None:
    additional = tmp_path / "additional" / "extracted"
    config = EvalConfig.from_dict(
        {
            "input_root": str(tmp_path / "input"),
            "additional_roots": [str(additional)],
            "output_dir": str(tmp_path / "out"),
            "modes": ["original"],
            "skip_metrics": ["lpips", "niqe"],
        }
    )

    assert config.search_roots() == (tmp_path / "input", additional)
    assert config.to_dict()["additional_roots"] == [str(additional)]


def test_lol_v2_real_low_and_normal_prefix_pairing(tmp_path) -> None:
    root = tmp_path / "input" / "LOL-v2-Real"
    for split in ("Train", "Test"):
        for index in range(2):
            _image(root / split / "Low" / f"low{index:05d}.png", seed=400 + index)
            _image(root / split / "Normal" / f"normal{index:05d}.png", seed=500 + index)

    manifest = next(
        item for item in discover(_config(tmp_path)).manifests if item.key == "lol_v2_real"
    )

    assert len(manifest.records) == 4
    assert manifest.split_counts == {"test": 2, "train": 2}
    assert manifest.evaluation_count == 2
    assert all(record.high is not None for record in manifest.records)
    assert {record.high.name for record in manifest.records if record.high} == {
        "normal00000.png",
        "normal00001.png",
    }


def test_sice_huggingface_input_gt_samples(tmp_path) -> None:
    root = tmp_path / "input" / "SICE_Paired"
    for split in ("train", "test"):
        for index in range(2):
            sample = root / split / "samples" / f"sample_{index:03d}"
            _image(sample / "Input.jpg", seed=600 + index)
            _image(sample / "GT.jpg", seed=700 + index)
    # Make the other SICE root look matching but unusable.
    (tmp_path / "input" / "SICE-Part1").mkdir(parents=True)
    (tmp_path / "input" / "SICE-Part1" / "Dataset_Part1.rar").write_bytes(b"not extracted")

    manifest = next(item for item in discover(_config(tmp_path)).manifests if item.key == "sice_p1")

    assert manifest.present
    assert manifest.root.name == "SICE_Paired"
    assert len(manifest.records) == 4
    assert manifest.split_counts == {"test": 2, "train": 2}
    assert manifest.evaluation_count == 2
    assert manifest.training_count == 2
    assert all(record.high is not None for record in manifest.records)


def test_exdark_and_dicm_discovery_when_present(tmp_path) -> None:
    root = tmp_path / "input"
    _image(root / "ExDark" / "class_a" / "one.jpg", seed=800)
    _image(root / "ExDark" / "class_b" / "two.jpg", seed=801)
    _image(root / "DICM" / "one.jpg", seed=810)

    manifests = {item.key: item for item in discover(_config(tmp_path)).manifests}

    assert manifests["exdark"].present
    assert manifests["exdark"].kind is DatasetKind.UNPAIRED
    assert len(manifests["exdark"].records) == 2
    assert len(manifests["exdark"].evaluation_records) == 2
    assert manifests["dicm"].present
    assert len(manifests["dicm"].records) == 1
    assert len(manifests["dicm"].evaluation_records) == 1


def test_kaggle_mount_nested_under_datasets_user_slug(tmp_path) -> None:
    base = tmp_path / "input" / "datasets" / "chiranth25x"
    low_light = base / "lumenoir-low-light-datasets"
    additional = base / "lumenoir-additional-test-datasets"
    _image(low_light / "LOL-v1" / "eval15" / "low" / "1.png", seed=900)
    _image(low_light / "LOL-v1" / "eval15" / "high" / "1.png", seed=901)
    _image(additional / "data" / "LIME" / "1.bmp", seed=910)

    config = _config(tmp_path).replace(
        input_root=tmp_path / "input",
        additional_roots=(low_light, additional),
    )
    manifests = {item.key: item for item in discover(config).manifests}

    assert manifests["lol_v1"].present
    assert manifests["lol_v1"].evaluation_count == 1
    assert manifests["lime"].present
    assert manifests["lime"].image_count == 1


def test_exact_exdark_and_dicm_mount_roots_are_discovered(tmp_path) -> None:
    exdark = tmp_path / "mounts" / "exdark-dataset"
    dicm = tmp_path / "mounts" / "dicm-img"
    _image(exdark / "class_a" / "one.jpg", seed=920)
    _image(dicm / "one.jpg", seed=930)

    config = _config(tmp_path).replace(
        additional_roots=(exdark, dicm),
    )
    manifests = {item.key: item for item in discover(config).manifests}

    assert manifests["exdark"].present
    assert manifests["exdark"].image_count == 1
    assert manifests["dicm"].present
    assert manifests["dicm"].image_count == 1


def test_kaggle_working_root_discovers_transferred_sice_and_npe(tmp_path) -> None:
    working = tmp_path / "working"
    sice = working / "SICE_Paired"
    for split in ("train", "test"):
        sample = sice / split / "samples" / "sample_001"
        _image(sample / "Input.jpg", seed=940)
        _image(sample / "GT.jpg", seed=941)
    _image(working / "NPE" / "npe.jpg", seed=950)
    _image(working / "LIME" / "lime.bmp", seed=951)

    config = _config(tmp_path).replace(additional_roots=(working,))
    manifests = {item.key: item for item in discover(config).manifests}

    assert manifests["sice_p1"].present
    assert manifests["sice_p1"].evaluation_count == 1
    assert manifests["npe"].present
    assert manifests["npe"].image_count == 1
    assert manifests["lime"].present
    assert manifests["lime"].image_count == 1

