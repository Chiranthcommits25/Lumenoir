"""Tests for the GPU-only full-pipeline benchmark.

These tests never require CUDA: they cover argument parsing, image collection,
config merging, the CPU-fallback trap and report rendering. The CUDA paths are
only reachable on an NVIDIA host, where the benchmark either measures or fails
loudly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from training.benchmarks import benchmark_full_pipeline_gpu as bench


def _write_image(path: Path, size: tuple[int, int] = (8, 6), value: int = 10) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (value, value, value)).save(path)
    return path


def _sample_report() -> dict:
    return {
        "benchmark": "lumenoir.full_pipeline.gpu",
        "generated_at": "2026-11-07T00:00:00+00:00",
        "git_commit": "abc1234",
        "gpu": {
            "device": "cuda:0",
            "name": "Tesla T4",
            "compute_capability": "7.5",
            "total_memory_bytes": 16 * 1024**3,
            "total_memory_mb": 15360.0,
            "multi_processor_count": 40,
            "torch_version": "2.15.0",
            "cuda_version": "12.1",
            "cudnn_version": 8900,
            "python": "3.10.13",
            "platform": "Linux-5.15",
        },
        "settings": {
            "device": "cuda:0",
            "precision": "float32",
            "weights": "backend/weights/zero_dce_plus_plus.pth",
            "image_source": "/kaggle/working/images",
            "count": 10,
            "warmups": 3,
            "runs": 1,
            "modes": ["hybrid"],
        },
        "checks": {
            "cuda_required": True,
            "cpu_fallback_allowed": False,
            "resolved_device": "cuda:0",
            "cpu_stage_calls": 0,
            "ops_class": "TorchImageOps",
            "ops_device": "cuda:0",
            "tensor_devices": ["cuda:0"],
            "parameter_count": 10561,
            "parameter_dtypes": ["torch.float32"],
        },
        "images": [
            {"name": "a.png", "path": "/kaggle/working/images/a.png", "width": 600, "height": 400}
        ],
        "modes": {
            "hybrid": {
                "mode": "hybrid",
                "required_stages": ["analysis", "classical_cv", "deep_learning", "fusion"],
                "images": 10,
                "timed_runs": 10,
                "warmup_passes": 3,
                "latency": {
                    "mean_ms": 1200.5,
                    "median_ms": 1198.0,
                    "p90_ms": 1240.0,
                    "min_ms": 1150.0,
                    "max_ms": 1300.0,
                    "total_processing_ms": 12005.0,
                    "loop_wall_ms": 12050.0,
                    "throughput_images_per_second": 0.83,
                },
                "per_stage_ms": {
                    "analysis": {"images": 10, "mean_ms": 120.0, "median_ms": 119.0},
                    "fusion": {"images": 10, "mean_ms": 300.0, "median_ms": 298.0},
                },
                "peak_cuda_allocated_bytes": 100 * 1024**2,
                "peak_cuda_reserved_bytes": 140 * 1024**2,
                "images_with_skipped_stages": 0,
                "runs": [
                    {
                        "image": "a.png",
                        "path": "/kaggle/working/images/a.png",
                        "width": 600,
                        "height": 400,
                        "mode": "hybrid",
                        "status": "completed",
                        "wall_ms": 1200.5,
                        "pipeline_total_ms": 1195.0,
                        "stage_ms": {"analysis": 120.0, "fusion": 300.0},
                        "cuda_peak_allocated_bytes": 100 * 1024**2,
                    }
                ],
            }
        },
    }


def test_parser_exposes_the_required_arguments() -> None:
    args = bench.build_parser().parse_args([])
    assert args.config is None
    assert args.images is None
    assert args.device is None
    assert args.precision is None
    assert args.output_dir is None


@pytest.mark.parametrize("device", ["cpu", "mps", "auto", ""])
def test_non_cuda_devices_are_refused(device: str) -> None:
    with pytest.raises(bench.GpuBenchmarkError):
        bench.require_cuda(device, "float32")


def test_cpu_refusal_explains_the_gpu_only_rule() -> None:
    with pytest.raises(bench.GpuBenchmarkError, match="GPU-only"):
        bench.require_cuda("cpu", "float32")


def test_collect_images_sorts_deduplicates_and_limits(tmp_path: Path) -> None:
    _write_image(tmp_path / "b.png")
    _write_image(tmp_path / "a.png")
    _write_image(tmp_path / "nested" / "c.jpg")
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")

    names = [path.name for path in bench.collect_images(tmp_path)]
    assert names == ["a.png", "b.png", "c.jpg"]
    assert [path.name for path in bench.collect_images(tmp_path, 2)] == ["a.png", "b.png"]


def test_collect_images_from_manifest_resolves_relative_paths(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "low" / "one.png")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"images": [{"low": "low/one.png"}]}), encoding="utf-8")
    assert bench.collect_images(manifest) == [image]


def test_collect_images_without_matches_raises(tmp_path: Path) -> None:
    with pytest.raises(bench.GpuBenchmarkError, match="No benchmark images"):
        bench.collect_images(tmp_path)


def test_settings_from_benchmark_config(tmp_path: Path) -> None:
    _write_image(tmp_path / "imgs" / "a.png")
    config = tmp_path / "bench.json"
    config.write_text(
        json.dumps(
            {
                "images": "imgs",
                "count": 3,
                "modes": ["hybrid"],
                "warmups": 1,
                "runs": 2,
                "precision": "float16",
                "device": "cuda:0",
            }
        ),
        encoding="utf-8",
    )
    args = bench.build_parser().parse_args(["--config", str(config)])
    settings = bench.settings_from_args(args)
    assert settings.images == tmp_path / "imgs"
    assert settings.count == 3
    assert settings.modes == ("hybrid",)
    assert (settings.warmups, settings.runs) == (1, 2)
    assert settings.precision == "float16"
    assert settings.device == "cuda:0"


def test_cli_overrides_the_config_file(tmp_path: Path) -> None:
    config = tmp_path / "bench.json"
    config.write_text(json.dumps({"count": 3, "modes": ["hybrid"]}), encoding="utf-8")
    args = bench.build_parser().parse_args(
        ["--config", str(config), "--count", "1", "--modes", "classical,deep_learning"]
    )
    settings = bench.settings_from_args(args)
    assert settings.count == 1
    assert settings.modes == ("classical", "deep_learning")


def test_unknown_mode_is_rejected() -> None:
    args = bench.build_parser().parse_args(["--modes", "fusion"])
    with pytest.raises(bench.GpuBenchmarkError, match="Unknown mode"):
        bench.settings_from_args(args)


def test_cpu_fallback_trap_fails_loudly_and_restores_the_module() -> None:
    from backend.enhancement import stages

    original = stages.scan
    with pytest.raises(bench.CpuExecutionError, match="scan"):
        with bench.CpuFallbackTrap() as trap:
            stages.scan(np.zeros((4, 4, 3), dtype=np.float32))
    assert trap.calls == ["scan"]
    assert stages.scan is original


def test_stage_totals_keep_pipeline_order() -> None:
    records = [
        {"stage_ms": {"fusion": 10.0, "analysis": 2.0}},
        {"stage_ms": {"fusion": 20.0, "analysis": 4.0}},
    ]
    totals = bench._stage_totals(records)
    assert list(totals) == ["analysis", "fusion"]
    assert totals["analysis"]["mean_ms"] == 3.0
    assert totals["fusion"]["median_ms"] == 15.0


def test_render_markdown_reports_the_measured_values() -> None:
    text = bench.render_markdown(_sample_report())
    assert "Lumenoir full-pipeline GPU benchmark" in text
    assert "Tesla T4" in text
    assert "| hybrid | 10 |" in text
    assert "CPU comparison: not run (GPU-only benchmark)" in text
    assert "| CPU stage implementation calls | 0 |" in text
    assert "100.0" in text  # peak CUDA megabytes
    assert "Do not quote them as another device's numbers." in text


def test_write_outputs_uses_the_precision_suffix(tmp_path: Path) -> None:
    json_path, markdown_path = bench.write_outputs(_sample_report(), tmp_path, "float16")
    assert json_path.name == "full_pipeline_gpu_fp16.json"
    assert markdown_path.name == "full_pipeline_gpu_fp16.md"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["checks"]["cpu_fallback_allowed"] is False
    assert "hybrid" in markdown_path.read_text(encoding="utf-8")
