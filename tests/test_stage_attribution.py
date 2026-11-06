"""Stage attribution instrumentation and reporting tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.enhancement.pipeline import EnhancementPipeline
from backend.enhancement.stages import (
    AnalysisStage,
    ClassicalStage,
    DeepLearningStage,
    FusionStage,
    QualityAssessmentStage,
)
from backend.evaluation.harness import evaluate_arrays
from backend.evaluation.images import to_float32_rgb
from tests.conftest import StubModelManager, brighten
from training.experiments.stage_attribution import (
    AttributionConfig,
    ValidationError,
    load_validation_pairs,
    run_attribution,
)


def _image(path: Path, seed: int, size: tuple[int, int] = (96, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(array).save(path)


def _manifest(tmp_path: Path, *, split: str = "validation") -> Path:
    low = tmp_path / "low.png"
    high = tmp_path / "high.png"
    _image(low, 1)
    _image(high, 2)
    payload = {
        "pool": split,
        "records": [
            {
                "dataset": "fixture",
                "image_id": "low.png",
                "low": str(low),
                "high": str(high),
                "split": split,
            }
        ],
    }
    path = tmp_path / "validation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_attribution_writes_stages_metrics_weights_and_grid(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "experiment"
    pipeline = EnhancementPipeline(model_manager=StubModelManager(transform=brighten))
    config = AttributionConfig(
        validation_manifest=manifest,
        output_dir=output,
        metrics=("psnr", "ssim", "delta_e00"),
        skip_metrics=("lpips",),
        visual_limit=1,
        measure_memory=False,
    )

    summary = run_attribution(config, pipeline=pipeline)

    assert summary.image_count == 1
    assert summary.failure_count == 0
    for stage in (
        "original",
        "raw_zero_dce",
        "classical",
        "fusion",
        "fusion_edgeguard",
        "fusion_edgeguard_huelock",
        "final",
    ):
        assert (output / "stages" / "fixture" / "low.png" / f"{stage}.png").is_file()
    assert (output / "stages" / "fixture" / "low.png" / "ground_truth.png").is_file()
    assert list((output / "visuals").rglob("*_grid.png"))
    assert (output / "per_image_metrics.csv").is_file()
    assert (output / "stage_aggregate.csv").is_file()
    assert (output / "dataset_comparison.csv").is_file()
    assert (output / "fusion_weights.csv").is_file()
    assert (output / "stage_latency.csv").is_file()
    assert (output / "stage_latency_aggregate.csv").is_file()
    assert (output / "summary.md").is_file()

    with (output / "per_image_metrics.csv").open(encoding="utf-8") as handle:
        metric_rows = list(csv.DictReader(handle))
    assert len(metric_rows) == 7
    assert {row["stage"] for row in metric_rows} == {
        "original",
        "raw_zero_dce",
        "classical",
        "fusion",
        "fusion_edgeguard",
        "fusion_edgeguard_huelock",
        "final",
    }
    assert all(row["psnr"] for row in metric_rows)

    low_image = Image.open(tmp_path / "low.png").convert("RGB")
    high_image = Image.open(tmp_path / "high.png").convert("RGB")
    raw_image = Image.open(
        output / "stages" / "fixture" / "low.png" / "raw_zero_dce.png"
    ).convert("RGB")
    assert np.array_equal(np.asarray(raw_image), np.asarray(brighten(low_image)))
    expected_original = evaluate_arrays(
        to_float32_rgb(low_image),
        reference=to_float32_rgb(high_image),
        original=to_float32_rgb(low_image),
        keys=("psnr",),
    )["psnr"].value
    original_row = next(row for row in metric_rows if row["stage"] == "original")
    assert float(original_row["psnr"]) == pytest.approx(expected_original, rel=1e-5)

    with (output / "fusion_weights.csv").open(encoding="utf-8") as handle:
        weight_rows = list(csv.DictReader(handle))
    assert len(weight_rows) == 6
    assert {row["candidate"] for row in weight_rows} == {
        "original",
        "classical",
        "deep_learning",
    }
    assert all(row["mean"] for row in weight_rows)

    # The captured final image must match normal inference, not a reconstruction.
    normal = pipeline.run(
        Image.open(tmp_path / "low.png").convert("RGB"), mode="hybrid"
    )
    try:
        captured = np.asarray(
            Image.open(output / "stages" / "fixture" / "low.png" / "final.png")
        )
        assert np.array_equal(captured, np.asarray(normal.final_image.convert("RGB")))
    finally:
        normal.release()


def test_attribution_rejects_test_and_training_records(tmp_path) -> None:
    for split in ("test", "train"):
        manifest = _manifest(tmp_path / split, split=split)
        with pytest.raises(ValidationError):
            load_validation_pairs(manifest)


def test_loader_resolves_relative_paths_against_data_root(tmp_path) -> None:
    data_root = tmp_path / "mounted-data"
    low = data_root / "LOL-v1" / "our485" / "low" / "10.png"
    high = data_root / "LOL-v1" / "our485" / "high" / "10.png"
    _image(low, 1)
    _image(high, 2)
    manifest = tmp_path / "validation.json"
    manifest.write_text(
        json.dumps(
            {
                "pool": "validation",
                "records": [
                    {
                        "dataset": "lol_v1",
                        "image_id": "our485/low/10.png",
                        "low": "LOL-v1/our485/low/10.png",
                        "high": "LOL-v1/our485/high/10.png",
                        "split": "validation",
                        "kind": "paired",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    pairs, _ = load_validation_pairs(manifest, data_root=data_root)

    assert len(pairs) == 1
    assert pairs[0].low == low.resolve()
    assert pairs[0].high == high.resolve()


def test_attribution_reports_missing_postprocess_stages(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "missing-stage-experiment"
    pipeline = EnhancementPipeline(
        stages=(
            AnalysisStage(),
            ClassicalStage(),
            DeepLearningStage(),
            FusionStage(),
            QualityAssessmentStage(),
        ),
        model_manager=StubModelManager(transform=brighten),
    )
    config = AttributionConfig(
        validation_manifest=manifest,
        output_dir=output,
        metrics=("psnr",),
        skip_metrics=("ssim", "lpips", "delta_e00"),
        measure_memory=False,
    )

    summary = run_attribution(config, pipeline=pipeline)

    assert summary.failure_count == 0
    with (output / "per_image_metrics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = {row["stage"] for row in rows if row["missing"] == "True"}
    assert {"fusion_edgeguard", "fusion_edgeguard_huelock"} <= missing


def test_attribution_refuses_existing_outputs(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("do not overwrite", encoding="utf-8")
    config = AttributionConfig(
        validation_manifest=manifest,
        output_dir=output,
        metrics=("psnr",),
        skip_metrics=("ssim", "lpips", "delta_e00"),
        measure_memory=False,
    )

    with pytest.raises(FileExistsError):
        run_attribution(
            config,
            pipeline=EnhancementPipeline(
                model_manager=StubModelManager(transform=brighten)
            ),
        )
    assert (output / "keep.txt").read_text(encoding="utf-8") == "do not overwrite"
