"""Kaggle evaluation runner and report tests with synthetic fixtures."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from training.kaggle_eval.cli import main
from training.kaggle_eval.config import EvalConfig
from training.kaggle_eval.runner import run_evaluation


def _image(path, seed: int = 0, size: tuple[int, int] = (64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(array).save(path)


def _build_fixture(root) -> None:
    for index in range(2):
        _image(root / "lol-v1" / "eval15" / "low" / f"pair_{index}.png", seed=index)
        _image(
            root / "lol-v1" / "eval15" / "high" / f"pair_{index}.png",
            seed=100 + index,
        )

    _image(root / "lol-v1" / "eval15" / "low" / "broken.png", seed=9)
    _image(root / "lol-v1" / "eval15" / "high" / "broken.png", seed=109)
    (root / "lol-v1" / "eval15" / "low" / "broken.png").write_bytes(b"not an image")

    _image(root / "lime" / "img_1.png", seed=20)
    _image(root / "lime" / "img_2.png", seed=21)


def _config(tmp_path) -> EvalConfig:
    return EvalConfig(
        input_root=tmp_path / "input",
        additional_roots=(),
        output_dir=tmp_path / "out",
        modes=("original", "classical"),
        seed=7,
        max_images=3,
        skip_metrics=("lpips", "niqe"),
        save_visuals=True,
        visual_limit=2,
    )


def _load_jsonl(path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_runner_writes_records_aggregates_failures_and_visuals(tmp_path) -> None:
    _build_fixture(tmp_path / "input")
    config = _config(tmp_path)

    summary = run_evaluation(config)
    output = config.output_dir

    assert summary.record_count == 8
    assert summary.failure_count >= 1
    assert summary.skipped_modes == ()

    assert (output / "manifests" / "manifest.json").is_file()
    assert (output / "records" / "records.jsonl").is_file()
    assert (output / "records" / "records.csv").is_file()
    assert (output / "failures" / "failures.jsonl").is_file()
    assert (output / "failures" / "failures.csv").is_file()
    assert (output / "aggregate.json").is_file()
    assert (output / "aggregate.csv").is_file()
    assert (output / "aggregate.md").is_file()
    assert (output / "latency.json").is_file()
    assert (output / "latency.csv").is_file()
    assert (output / "latency.md").is_file()
    assert (output / "device.json").is_file()
    assert (output / "run_config.json").is_file()
    assert (output / "reports" / "lol_v1" / "classical" / "image_metrics.json").is_file()

    records = _load_jsonl(output / "records" / "records.jsonl")
    paired = next(
        record
        for record in records
        if record["dataset"] == "lol_v1"
        and record["mode"] == "classical"
        and record["success"]
    )
    assert paired["kind"] == "paired"
    assert "psnr" in paired["metrics"]
    assert "ssim" in paired["metrics"]
    assert "lpips" not in paired["metrics"]

    unpaired = next(
        record
        for record in records
        if record["dataset"] == "lime"
        and record["mode"] == "classical"
        and record["success"]
    )
    assert unpaired["kind"] == "unpaired"
    assert {"loe", "noise_relative", "edge_preservation", "black_crushing"} <= set(
        unpaired["metrics"]
    )
    assert not {"psnr", "ssim", "lpips"} & set(unpaired["metrics"])

    aggregate = json.loads((output / "aggregate.json").read_text(encoding="utf-8"))
    assert any(
        row["dataset"] == "lol_v1"
        and row["mode"] == "classical"
        and row["metric"] == "psnr"
        and row["count"] == 2
        for row in aggregate["metrics"]
    )

    failures = _load_jsonl(output / "failures" / "failures.jsonl")
    assert any(failure["code"] == "load_failed" for failure in failures)

    visuals = list((output / "visuals").rglob("*.png"))
    assert visuals
    assert any("classical" in path.name for path in visuals)

    run_config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["config"]["seed"] == 7
    assert run_config["record_count"] == 8


def test_runner_is_deterministic_in_metric_values(tmp_path) -> None:
    _build_fixture(tmp_path / "input")
    first = run_evaluation(_config(tmp_path))
    second_config = _config(tmp_path).replace(output_dir=tmp_path / "out2")
    second = run_evaluation(second_config)

    def stable(records_path):
        rows = _load_jsonl(records_path)
        for row in rows:
            row.pop("processing_time_ms", None)
            row.pop("stage_timings_ms", None)
        return rows

    assert stable(first.output_dir / "records" / "records.jsonl") == stable(
        second.output_dir / "records" / "records.jsonl"
    )


def test_cli_discover_only(tmp_path) -> None:
    _build_fixture(tmp_path / "input")
    output = tmp_path / "cli-out"

    code = main(
        [
            "--discover-only",
            "--input-root",
            str(tmp_path / "input"),
            "--output-dir",
            str(output),
        ]
    )

    assert code == 0
    assert (output / "manifests" / "manifest.json").is_file()


def test_cli_runs_original_mode_on_one_dataset(tmp_path) -> None:
    _build_fixture(tmp_path / "input")
    output = tmp_path / "cli-run"

    code = main(
        [
            "--input-root",
            str(tmp_path / "input"),
            "--output-dir",
            str(output),
            "--datasets",
            "lime",
            "--modes",
            "original",
            "--skip-metrics",
            "niqe",
        ]
    )

    assert code == 0
    records = _load_jsonl(output / "records" / "records.jsonl")
    assert len(records) == 2
    assert all(record["mode"] == "original" for record in records)
    assert all(record["dataset"] == "lime" for record in records)


def test_cli_discover_only_with_additional_root(tmp_path) -> None:
    additional = tmp_path / "additional" / "extracted"
    _image(additional / "LIME" / "data" / "LIME" / "image.bmp", seed=1)
    output = tmp_path / "cli-additional"

    code = main(
        [
            "--discover-only",
            "--input-root",
            str(tmp_path / "input"),
            "--additional-root",
            str(additional),
            "--output-dir",
            str(output),
        ]
    )

    assert code == 0
    manifest = json.loads((output / "manifests" / "manifest.json").read_text(encoding="utf-8"))
    lime = next(dataset for dataset in manifest["datasets"] if dataset["dataset"] == "lime")
    assert lime["image_count"] == 1
    assert lime["format_counts"] == {"bmp": 1}


def test_npe_subgroups_are_reported_as_one_unpaired_dataset(tmp_path) -> None:
    additional = tmp_path / "additional" / "extracted"
    for index, subgroup in enumerate(("NPE", "NPE-ex1", "NPE-ex2", "NPE-ex3")):
        _image(
            additional / subgroup / "data" / subgroup / "image.png",
            seed=300 + index,
        )

    config = EvalConfig(
        input_root=tmp_path / "input",
        additional_roots=(additional,),
        output_dir=tmp_path / "npe-out",
        selected_datasets=("npe",),
        modes=("original",),
        seed=7,
        skip_metrics=("lpips", "niqe"),
    )

    summary = run_evaluation(config)
    output = config.output_dir

    assert summary.record_count == 4
    assert summary.failure_count == 0

    records = _load_jsonl(output / "records" / "records.jsonl")
    assert {record["dataset"] for record in records} == {"npe"}
    assert {record["subgroup"] for record in records} == {
        "npe",
        "npe_ex1",
        "npe_ex2",
        "npe_ex3",
    }
    assert all(record["kind"] == "unpaired" for record in records)
    assert all(
        not {"psnr", "ssim", "lpips"} & set(record["metrics"])
        for record in records
    )

    aggregate = json.loads((output / "aggregate.json").read_text(encoding="utf-8"))
    assert {row["dataset"] for row in aggregate["metrics"]} == {"npe"}

    report = json.loads(
        (output / "reports" / "npe" / "original" / "image_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["metadata"]["subgroup_counts"] == {
        "npe": 1,
        "npe_ex1": 1,
        "npe_ex2": 1,
        "npe_ex3": 1,
    }

    manifest = json.loads((output / "manifests" / "manifest.json").read_text(encoding="utf-8"))
    npe = next(dataset for dataset in manifest["datasets"] if dataset["dataset"] == "npe")
    assert npe["subgroup_counts"] == {
        "npe": 1,
        "npe_ex1": 1,
        "npe_ex2": 1,
        "npe_ex3": 1,
    }

