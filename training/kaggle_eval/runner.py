"""Kaggle evaluation runner.

Runs the existing enhancement pipeline over deterministic dataset manifests and
records image-level metrics, failures, per-stage timings, device information,
aggregates, CSV/JSON exports and before/after visuals.

It never reimplements enhancement or tuning. Datasets are read from the mounted
Kaggle input directory; nothing is downloaded.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from backend import deep_learning  # noqa: F401  (registers zero_dce_plus_plus)
from backend.config import DEFAULT_WEIGHTS_DIR, InferenceSettings
from backend.enhancement.pipeline import EnhancementPipeline
from backend.enhancement.shared import SharedStageCache
from backend.evaluation.aggregate import summarise
from backend.evaluation.harness import evaluate_arrays
from backend.evaluation.images import to_float32_rgb
from backend.evaluation.latency import LatencyBenchmark
from backend.evaluation.report import write_csv, write_json, write_markdown
from backend.inference.errors import InferenceError
from backend.inference.manager import ModelManager, ModelStatus

from .config import (
    DEFAULT_WEIGHTS,
    MODES,
    EvalConfig,
)
from .discovery import (
    DatasetManifest,
    EvalRecord,
    discover,
    write_manifests,
)

MODEL_REQUIRED_MODES = ("deep_learning", "hybrid")


@dataclass(frozen=True)
class MetricRecord:
    dataset: str
    label: str
    kind: str
    split: str
    image_id: str
    mode: str
    success: bool
    metrics: dict[str, float | None] = field(default_factory=dict)
    metric_notes: dict[str, str] = field(default_factory=dict)
    processing_time_ms: float | None = None
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    metric_time_ms: float | None = None
    error: str | None = None
    subgroup: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "label": self.label,
            "kind": self.kind,
            "split": self.split,
            "image_id": self.image_id,
            "subgroup": self.subgroup,
            "mode": self.mode,
            "success": self.success,
            "metrics": self.metrics,
            "metric_notes": self.metric_notes,
            "processing_time_ms": self.processing_time_ms,
            "stage_timings_ms": self.stage_timings_ms,
            "metric_time_ms": self.metric_time_ms,
            "error": self.error,
        }


@dataclass(frozen=True)
class FailureRecord:
    dataset: str
    image_id: str
    mode: str
    code: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "image_id": self.image_id,
            "mode": self.mode,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class RunSummary:
    output_dir: Path
    record_count: int
    failure_count: int
    skipped_modes: tuple[str, ...]
    manifest_paths: tuple[Path, ...]
    record_paths: tuple[Path, ...]
    aggregate_paths: tuple[Path, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "output_dir": str(self.output_dir),
            "record_count": self.record_count,
            "failure_count": self.failure_count,
            "skipped_modes": list(self.skipped_modes),
            "manifest_paths": [str(path) for path in self.manifest_paths],
            "record_paths": [str(path) for path in self.record_paths],
            "aggregate_paths": [str(path) for path in self.aggregate_paths],
            "warnings": list(self.warnings),
        }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
            timeout=10,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _find_weight_files(*roots: Path, max_depth: int = 3) -> list[Path]:
    names = {"zero_dce_plus_plus.pth", "epoch99.pth"}
    found: list[Path] = []
    seen: set[Path] = set()

    for root in roots:
        if root is None or not root.is_dir():
            continue
        root = root.resolve()
        for current, dirs, files in os.walk(root):
            current_path = Path(current)
            depth = len(current_path.relative_to(root).parts)
            dirs[:] = [name for name in dirs if not name.startswith(".")]
            for name in files:
                if name.lower() in names:
                    path = current_path / name
                    if path not in seen:
                        seen.add(path)
                        found.append(path)
            if depth >= max_depth:
                dirs[:] = []
    return sorted(found)


def resolve_weights(config: EvalConfig) -> Path | None:
    if config.weights is not None:
        return config.weights if config.weights.is_file() else None
    if DEFAULT_WEIGHTS.is_file():
        return DEFAULT_WEIGHTS
    candidates = _find_weight_files(
        config.input_root,
        Path("/kaggle/working"),
        DEFAULT_WEIGHTS_DIR,
    )
    return candidates[0] if candidates else None


def _build_manager(
    config: EvalConfig, modes: tuple[str, ...]
) -> tuple[ModelManager | None, ModelStatus | None, Path | None, tuple[str, ...]]:
    if not any(mode in MODEL_REQUIRED_MODES for mode in modes):
        return None, None, None, ()

    weights = resolve_weights(config)
    settings = InferenceSettings(
        model_id=config.model_id,
        weights_path=weights,
        weights_dir=weights.parent if weights else DEFAULT_WEIGHTS_DIR,
        device=config.device,
        load_on_startup=True,
    )
    manager = ModelManager(settings)
    status = manager.load_at_startup()
    if status.ready:
        return manager, status, weights, ()

    message = f"Model is not ready ({status.state.value}): {status.message}"
    if not config.allow_missing_model:
        manager.close()
        raise RuntimeError(message)

    skipped = tuple(mode for mode in modes if mode in MODEL_REQUIRED_MODES)
    return manager, status, weights, skipped


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        upright = ImageOps.exif_transpose(opened)
        image = upright.convert("RGB")
        image.load()
        return image


def _save_visual(low: Image.Image, output: Image.Image, path: Path) -> None:
    if low.size != output.size:
        return
    width, height = low.size
    canvas = Image.new("RGB", (width * 2 + 2, height), (0, 0, 0))
    canvas.paste(low, (0, 0))
    canvas.paste(output, (width + 2, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")
    canvas.close()


def _visual_name(image_id: str, mode: str) -> str:
    stem = Path(image_id).with_suffix("").as_posix().replace("/", "_")
    return f"{stem}_{mode}.png"


def _finite(value: float | None) -> tuple[float | None, str | None]:
    if value is None:
        return None, None
    if not math.isfinite(value):
        return None, "non-finite value"
    return float(value), None


def _evaluate(
    output: np.ndarray,
    low: np.ndarray,
    reference: np.ndarray | None,
    kind: str,
    config: EvalConfig,
) -> tuple[dict[str, float | None], dict[str, str]]:
    keys = (
        config.paired_metrics()
        if kind == "paired"
        else config.unpaired_metrics()
    )
    values = evaluate_arrays(
        output,
        reference=reference,
        original=low,
        keys=keys,
        assume_valid=True,
    )
    metrics: dict[str, float | None] = {}
    notes: dict[str, str] = {}
    metric_time_ms = 0.0
    for key, metric in values.items():
        metric_time_ms += float(metric.metric_time_ms or 0.0)
        value, note = _finite(metric.value)
        metrics[key] = value
        if metric.notes:
            notes[key] = metric.notes
        elif note:
            notes[key] = note
    return metrics, notes, round(metric_time_ms, 3)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_records_csv(records: list[MetricRecord], path: Path) -> None:
    metric_keys = sorted(
        {key for record in records for key in record.metrics}
    )
    fieldnames = [
        "dataset",
        "label",
        "kind",
        "split",
        "image_id",
        "subgroup",
        "mode",
        "success",
        "processing_time_ms",
        "metric_time_ms",
        "error",
        *metric_keys,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "dataset": record.dataset,
                "label": record.label,
                "kind": record.kind,
                "split": record.split,
                "image_id": record.image_id,
                "subgroup": record.subgroup,
                "mode": record.mode,
                "success": record.success,
                "processing_time_ms": record.processing_time_ms,
                "metric_time_ms": record.metric_time_ms,
                "error": record.error,
            }
            for key in metric_keys:
                row[key] = record.metrics.get(key)
            writer.writerow(row)


def _write_failures(failures: list[FailureRecord], output_dir: Path) -> None:
    failure_dir = output_dir / "failures"
    failure_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        failure_dir / "failures.jsonl",
        [failure.to_dict() for failure in failures],
    )
    with (failure_dir / "failures.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["dataset", "image_id", "mode", "code", "message"]
        )
        writer.writeheader()
        for failure in failures:
            writer.writerow(failure.to_dict())


def _metric_keys(records: list[MetricRecord]) -> list[str]:
    return sorted({key for record in records for key in record.metrics})


def _rows_for(
    records: list[MetricRecord], dataset: str, mode: str
) -> list[dict[str, object]]:
    return [
        {"name": record.image_id, **record.metrics}
        for record in records
        if record.success
        and record.dataset == dataset
        and record.mode == mode
    ]


def _summarise_aggregates(
    records: list[MetricRecord],
    datasets: list[DatasetManifest],
    output_dir: Path,
) -> tuple[list[dict[str, object]], list[Path]]:
    rows: list[dict[str, object]] = []
    written: list[Path] = []
    for manifest in datasets:
        for mode in sorted({record.mode for record in records if record.dataset == manifest.key}):
            subset = _rows_for(records, manifest.key, mode)
            if not subset:
                continue
            for metric in _metric_keys(
                [record for record in records if record.dataset == manifest.key]
            ):
                values = [
                    float(row[metric])
                    for row in subset
                    if isinstance(row.get(metric), (int, float))
                ]
                summary = summarise(values)
                if summary is None:
                    continue
                rows.append(
                    {
                        "dataset": manifest.key,
                        "label": manifest.label,
                        "kind": manifest.kind.value,
                        "mode": mode,
                        "metric": metric,
                        "count": summary.count,
                        "mean": summary.mean,
                        "std": summary.std,
                        "median": summary.median,
                        "p10": summary.p10,
                        "p90": summary.p90,
                        "min": summary.minimum,
                        "max": summary.maximum,
                    }
                )

            report_dir = output_dir / "reports" / manifest.key / mode
            subgroup_counts = Counter(
                record.subgroup
                for record in records
                if record.dataset == manifest.key
                and record.mode == mode
                and record.subgroup is not None
            )
            report_metadata: dict[str, object] = {
                "dataset": manifest.key,
                "mode": mode,
                "kind": manifest.kind.value,
                "count": len(subset),
            }
            if subgroup_counts:
                report_metadata["subgroup_counts"] = dict(sorted(subgroup_counts.items()))
            written.extend(
                [
                    write_json(
                        subset,
                        report_dir / "image_metrics.json",
                        metadata=report_metadata,
                    ),
                    write_csv(subset, report_dir / "image_metrics.csv"),
                    write_markdown(
                        subset,
                        report_dir / "image_metrics.md",
                        title=f"{manifest.label} — {mode}",
                        metadata={"dataset": manifest.key, "mode": mode},
                    ),
                ]
            )
    return rows, written


def _write_aggregate_files(
    rows: list[dict[str, object]], output_dir: Path
) -> list[Path]:
    written: list[Path] = []
    json_path = output_dir / "aggregate.json"
    json_path.write_text(json.dumps({"metrics": rows}, indent=2), encoding="utf-8")
    written.append(json_path)

    csv_path = output_dir / "aggregate.csv"
    fieldnames = [
        "dataset",
        "label",
        "kind",
        "mode",
        "metric",
        "count",
        "mean",
        "std",
        "median",
        "p10",
        "p90",
        "min",
        "max",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    written.append(csv_path)

    lines = ["# LUMENOIR Kaggle evaluation aggregate", ""]
    lines.append("| Dataset | Mode | Metric | n | Mean | Std | Median | P10 | P90 |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['mode']} | {row['metric']} | {row['count']} | "
            f"{row['mean']:.4f} | {row['std']:.4f} | {row['median']:.4f} | "
            f"{row['p10']:.4f} | {row['p90']:.4f} |"
        )
    markdown_path = output_dir / "aggregate.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(markdown_path)
    return written


def _latency_rows(records: list[MetricRecord]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
    for record in records:
        if not record.success or not record.stage_timings_ms:
            continue
        key = (record.dataset, record.mode)
        stage_values = grouped.setdefault(key, {})
        for stage, value in record.stage_timings_ms.items():
            stage_values.setdefault(stage, []).append(float(value))

    rows: list[dict[str, object]] = []
    for (dataset, mode), stage_values in sorted(grouped.items()):
        for stage, values in sorted(stage_values.items()):
            summary = summarise(values)
            if summary is None:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "stage": stage,
                    "count": summary.count,
                    "mean_ms": summary.mean,
                    "median_ms": summary.median,
                    "p95_ms": float(np.percentile(values, 95.0)),
                    "min_ms": summary.minimum,
                    "max_ms": summary.maximum,
                }
            )
    return rows


def _latency_total_rows(records: list[MetricRecord]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for record in records:
        if not record.success or record.processing_time_ms is None:
            continue
        grouped.setdefault((record.dataset, record.mode), []).append(
            float(record.processing_time_ms)
        )

    rows: list[dict[str, object]] = []
    for (dataset, mode), values in sorted(grouped.items()):
        summary = summarise(values)
        if summary is None:
            continue
        rows.append(
            {
                "dataset": dataset,
                "mode": mode,
                "count": summary.count,
                "mean_ms": summary.mean,
                "median_ms": summary.median,
                "p95_ms": float(np.percentile(values, 95.0)),
                "min_ms": summary.minimum,
                "max_ms": summary.maximum,
            }
        )
    return rows


def _benchmark_image(
    pipeline: EnhancementPipeline,
    image: Image.Image,
    mode: str,
    config: EvalConfig,
) -> dict[str, object] | None:
    if config.latency_runs <= 0:
        return None

    def timed() -> dict[str, float]:
        result = pipeline.run(image, mode=mode)  # type: ignore[arg-type]
        if not result.completed or result.final_image is None:
            result.release()
            raise RuntimeError(result.message)
        timings = dict(result.timings_ms)
        timings["total"] = float(result.total_ms)
        result.release()
        return timings

    try:
        benchmark = LatencyBenchmark(
            timed,
            warmups=config.latency_warmups,
            runs=config.latency_runs,
            threads=1,
            sample_memory=False,
        ).run()
    except Exception:
        return None

    return {
        "warmups": benchmark.warmups,
        "runs": benchmark.runs,
        "median_ms": benchmark.median_ms,
        "mean_ms": benchmark.mean_ms,
        "p95_ms": benchmark.p95_ms,
        "min_ms": benchmark.min_ms,
        "max_ms": benchmark.max_ms,
        "per_stage_ms": benchmark.per_stage_ms,
        "peak_rss_mb": benchmark.peak_rss_mb,
    }


def _benchmark_rows(
    manifests: list[DatasetManifest],
    pipeline: EnhancementPipeline,
    modes: tuple[str, ...],
    config: EvalConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for manifest in manifests:
        try:
            records = manifest.validated_evaluation_records()
        except ValueError:
            continue
        if not records:
            continue
        image = None
        try:
            image = _load_image(records[0].low)
            for mode in modes:
                if mode == "original":
                    continue
                result = _benchmark_image(pipeline, image, mode, config)
                if result is not None:
                    rows.append(
                        {
                            "dataset": manifest.key,
                            "mode": mode,
                            **result,
                        }
                    )
        except Exception:
            continue
        finally:
            if image is not None:
                image.close()
    return rows


def _write_latency(
    rows: list[dict[str, object]],
    totals: list[dict[str, object]],
    benchmarks: list[dict[str, object]],
    output_dir: Path,
) -> list[Path]:
    written: list[Path] = []
    json_path = output_dir / "latency.json"
    json_path.write_text(
        json.dumps(
            {"stages": rows, "totals": totals, "benchmarks": benchmarks},
            indent=2,
        ),
        encoding="utf-8",
    )
    written.append(json_path)

    csv_path = output_dir / "latency.csv"
    fieldnames = [
        "dataset",
        "mode",
        "stage",
        "count",
        "mean_ms",
        "median_ms",
        "p95_ms",
        "min_ms",
        "max_ms",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    written.append(csv_path)

    totals_csv = output_dir / "latency_totals.csv"
    total_fields = [
        "dataset",
        "mode",
        "count",
        "mean_ms",
        "median_ms",
        "p95_ms",
        "min_ms",
        "max_ms",
    ]
    with totals_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=total_fields)
        writer.writeheader()
        writer.writerows(totals)
    written.append(totals_csv)

    lines = ["# Per-stage latency", ""]
    lines.append("| Dataset | Mode | Stage | n | Mean ms | Median ms | P95 ms |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['mode']} | {row['stage']} | {row['count']} | "
            f"{row['mean_ms']:.1f} | {row['median_ms']:.1f} | {row['p95_ms']:.1f} |"
        )
    lines.extend(["", "## Total processing latency", ""])
    lines.append("| Dataset | Mode | n | Mean ms | Median ms | P95 ms |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in totals:
        lines.append(
            f"| {row['dataset']} | {row['mode']} | {row['count']} | "
            f"{row['mean_ms']:.1f} | {row['median_ms']:.1f} | {row['p95_ms']:.1f} |"
        )
    if benchmarks:
        lines.extend(["", "## Warm-up benchmark", ""])
        lines.append("| Dataset | Mode | Warm-ups | Runs | Median ms | P95 ms |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for row in benchmarks:
            lines.append(
                f"| {row['dataset']} | {row['mode']} | {row['warmups']} | "
                f"{row['runs']} | {row['median_ms']:.1f} | {row['p95_ms']:.1f} |"
            )
    markdown_path = output_dir / "latency.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(markdown_path)
    return written


def _device_report(config: EvalConfig, manager: ModelManager | None) -> dict[str, object]:
    report: dict[str, object] = {
        "configured_device": config.device,
        "resolved_device": (
            manager.status.device
            if manager is not None and manager.status.device is not None
            else ("cpu" if config.device == "auto" else config.device)
        ),
        "classical_stages": "cpu",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import importlib.util

        if importlib.util.find_spec("torch") is not None:
            import torch

            report.update(
                {
                    "torch_version": torch.__version__,
                    "cuda_available": bool(torch.cuda.is_available()),
                    "cuda_device_count": int(torch.cuda.device_count()),
                    "cuda_devices": [
                        torch.cuda.get_device_name(index)
                        for index in range(torch.cuda.device_count())
                    ],
                    "mps_available": bool(
                        getattr(torch.backends, "mps", None)
                        and torch.backends.mps.is_available()
                    ),
                }
            )
        else:
            report["torch_version"] = None
            report["cuda_available"] = False
            report["mps_available"] = False
    except Exception as exc:  # pragma: no cover - environment dependent
        report["torch_error"] = str(exc)
    return report


def _run_record(
    record: EvalRecord,
    label: str,
    *,
    config: EvalConfig,
    pipeline: EnhancementPipeline,
    modes: tuple[str, ...],
    visuals_dir: Path,
    visual_counts: dict[str, int],
) -> tuple[list[MetricRecord], list[FailureRecord]]:
    metric_records: list[MetricRecord] = []
    failures: list[FailureRecord] = []

    try:
        low_image = _load_image(record.low)
    except Exception as exc:
        failures.append(
            FailureRecord(record.dataset, record.image_id, "*", "load_failed", str(exc))
        )
        return metric_records, failures

    reference = None
    if record.kind.value == "paired":
        if record.high is None:
            low_image.close()
            failures.append(
                FailureRecord(
                    record.dataset,
                    record.image_id,
                    "*",
                    "missing_reference",
                    "paired record has no reference image",
                )
            )
            return metric_records, failures
        try:
            reference_image = _load_image(record.high)
        except Exception as exc:
            low_image.close()
            failures.append(
                FailureRecord(
                    record.dataset, record.image_id, "*", "load_reference_failed", str(exc)
                )
            )
            return metric_records, failures
        if reference_image.size != low_image.size:
            low_image.close()
            reference_image.close()
            failures.append(
                FailureRecord(
                    record.dataset,
                    record.image_id,
                    "*",
                    "size_mismatch",
                    f"input {low_image.size} and reference {reference_image.size} differ",
                )
            )
            return metric_records, failures
        reference = to_float32_rgb(reference_image)
        reference_image.close()
    else:
        reference_image = None

    low = to_float32_rgb(low_image)

    cache: SharedStageCache | None = None
    if config.share_stages and sum(mode != "original" for mode in modes) > 1:
        cache = SharedStageCache()

    try:
        for mode in modes:
            output_image = None
            try:
                if mode == "original":
                    output = low.copy()
                    processing_time = 0.0
                    timings: dict[str, float] = {}
                    output_image = low_image.copy()
                else:
                    result = pipeline.run(  # type: ignore[arg-type]
                        low_image, mode=mode, cache=cache
                    )
                    if not result.completed or result.final_image is None:
                        failures.append(
                            FailureRecord(
                                record.dataset,
                                record.image_id,
                                mode,
                                "pipeline_failed",
                                result.message,
                            )
                        )
                        result.release()
                        continue
                    if result.final_image.size != low_image.size:
                        failures.append(
                            FailureRecord(
                                record.dataset,
                                record.image_id,
                                mode,
                                "output_size_mismatch",
                                f"{result.final_image.size} != {low_image.size}",
                            )
                        )
                        result.release()
                        continue
                    processing_time = result.total_ms
                    timings = dict(result.timings_ms)
                    output_image = result.final_image.copy()
                    output = to_float32_rgb(result.final_image)
                    result.release()

                metrics, notes, metric_time_ms = _evaluate(
                    output,
                    low,
                    reference,
                    record.kind.value,
                    config,
                )
                metric_records.append(
                    MetricRecord(
                        dataset=record.dataset,
                        label=label,
                        kind=record.kind.value,
                        split=record.split,
                        image_id=record.image_id,
                        mode=mode,
                        success=True,
                        subgroup=record.subgroup,
                        metrics=metrics,
                        metric_notes=notes,
                        processing_time_ms=processing_time,
                        stage_timings_ms=timings,
                        metric_time_ms=metric_time_ms,
                    )
                )

                if (
                    mode != "original"
                    and config.save_visuals
                    and visual_counts.get(record.dataset, 0) < config.visual_limit
                ):
                    _save_visual(
                        low_image,
                        output_image,
                        visuals_dir / record.dataset / _visual_name(record.image_id, mode),
                    )
                    visual_counts[record.dataset] = visual_counts.get(record.dataset, 0) + 1
            except InferenceError as exc:
                failures.append(
                    FailureRecord(
                        record.dataset,
                        record.image_id,
                        mode,
                        exc.code,
                        exc.message,
                    )
                )
            except Exception as exc:
                failures.append(
                    FailureRecord(
                        record.dataset,
                        record.image_id,
                        mode,
                        "evaluation_failed",
                        str(exc),
                    )
                )
            finally:
                if output_image is not None:
                    output_image.close()
    finally:
        if cache is not None:
            cache.close()
        low_image.close()

    return metric_records, failures


def run_evaluation(config: EvalConfig) -> RunSummary:
    """Execute one deterministic evaluation run and write every report."""
    started = datetime.now(UTC).isoformat(timespec="seconds")
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = output_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    discovery = discover(config)
    manifest_paths = tuple(write_manifests(discovery, output_dir, config=config))

    modes = tuple(mode for mode in config.modes if mode in MODES)
    manager, model_status, weights, skipped_modes = _build_manager(config, modes)
    active_modes = tuple(mode for mode in modes if mode not in skipped_modes)
    pipeline = EnhancementPipeline(model_manager=manager)

    records: list[MetricRecord] = []
    failures: list[FailureRecord] = []
    visual_counts: dict[str, int] = {}
    warnings: list[str] = list(discovery.warnings())

    if skipped_modes:
        message = (
            "Model could not be loaded; these modes were skipped: "
            + ", ".join(skipped_modes)
        )
        warnings.append(message)
        for mode in skipped_modes:
            failures.append(
                FailureRecord(
                    "*",
                    "*",
                    mode,
                    "model_unavailable",
                    message,
                )
            )

    for manifest in discovery.manifests:
        if not manifest.present:
            continue
        try:
            evaluation_records = manifest.validated_evaluation_records()
        except ValueError as exc:
            failures.append(
                FailureRecord(manifest.key, "*", "*", "manifest_invalid", str(exc))
            )
            warnings.append(f"{manifest.key}: manifest validation failed: {exc}")
            continue
        if manifest.records and not evaluation_records:
            warnings.append(
                f"{manifest.key}: discovered {len(manifest.records)} training-only "
                "record(s); no evaluation-split records were evaluated."
            )
            continue
        for record in evaluation_records:
            metric_records, record_failures = _run_record(
                record,
                manifest.label,
                config=config,
                pipeline=pipeline,
                modes=active_modes,
                visuals_dir=visuals_dir,
                visual_counts=visual_counts,
            )
            records.extend(metric_records)
            failures.extend(record_failures)

    record_dir = output_dir / "records"
    record_dir.mkdir(parents=True, exist_ok=True)
    records_jsonl = record_dir / "records.jsonl"
    _write_jsonl(records_jsonl, [record.to_dict() for record in records])
    records_csv = record_dir / "records.csv"
    _write_records_csv(records, records_csv)

    _write_failures(failures, output_dir)

    aggregate_rows, report_paths = _summarise_aggregates(
        records, list(discovery.manifests), output_dir
    )
    aggregate_paths = _write_aggregate_files(aggregate_rows, output_dir)

    latency_paths = _write_latency(
        _latency_rows(records),
        _latency_total_rows(records),
        _benchmark_rows(list(discovery.manifests), pipeline, active_modes, config),
        output_dir,
    )

    device_report = _device_report(config, manager)
    device_path = output_dir / "device.json"
    device_path.write_text(json.dumps(device_report, indent=2), encoding="utf-8")

    finished = datetime.now(UTC).isoformat(timespec="seconds")
    run_config = {
        "config": config.to_dict(),
        "started_at": started,
        "finished_at": finished,
        "git_commit": _git_commit(),
        "device": device_report,
        "model_status": None
        if model_status is None
        else {
            "state": model_status.state.value,
            "model_id": model_status.model_id,
            "device": model_status.device,
            "weights_path": model_status.weights_path,
            "message": model_status.message,
            "ready": model_status.ready,
        },
        "weights": None if weights is None else str(weights),
        "skipped_modes": list(skipped_modes),
        "warnings": warnings,
        "record_count": len(records),
        "failure_count": len(failures),
        "manifests": [manifest.to_dict() for manifest in discovery.manifests],
    }
    config_path = output_dir / "run_config.json"
    config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    if manager is not None:
        manager.close()

    record_paths = (records_jsonl, records_csv)
    aggregate_paths = (*aggregate_paths, *report_paths, *latency_paths, device_path, config_path)

    return RunSummary(
        output_dir=output_dir,
        record_count=len(records),
        failure_count=len(failures),
        skipped_modes=skipped_modes,
        manifest_paths=manifest_paths,
        record_paths=record_paths,
        aggregate_paths=aggregate_paths,
        warnings=tuple(warnings),
    )
