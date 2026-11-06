"""Stage attribution experiment for the existing Lumenoir pipeline.

This module adds no enhancement logic. It runs the production ``hybrid`` mode
with diagnostic capture enabled, saves the actual intermediate images, evaluates
each stage against the same ground truth, and writes stage/dataset/per-image
reports.

The experiment refuses to use test or training records. It requires an explicit
validation manifest and never creates a validation split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from backend import deep_learning  # noqa: F401  (registers Zero-DCE++)
from backend.config import DEFAULT_WEIGHTS_DIR, InferenceSettings
from backend.enhancement.pipeline import EnhancementPipeline
from backend.enhancement.types import ArtifactKind
from backend.evaluation.aggregate import summarise
from backend.evaluation.harness import evaluate_arrays
from backend.evaluation.images import to_float32_rgb
from backend.inference.manager import ModelManager, ModelStatus

DEFAULT_WEIGHTS = DEFAULT_WEIGHTS_DIR / "zero_dce_plus_plus.pth"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "raw"
DEFAULT_METRICS = ("psnr", "ssim", "lpips", "delta_e00")
STAGES: tuple[tuple[str, str], ...] = (
    ("original", "Original"),
    ("raw_zero_dce", "Raw Zero-DCE++"),
    ("classical", "Classical LumenRetinex"),
    ("fusion", "Raw LumenFuse"),
    ("fusion_edgeguard", "Fusion + EdgeGuard"),
    ("fusion_edgeguard_huelock", "Fusion + EdgeGuard + HueLock"),
    ("final", "Final alpha blend"),
)
STAGE_LABELS = dict(STAGES)


class ValidationError(ValueError):
    """The supplied validation manifest is not usable."""


@dataclass(frozen=True)
class ValidationPair:
    dataset: str
    image_id: str
    low: Path
    high: Path


@dataclass(frozen=True)
class AttributionConfig:
    validation_manifest: Path
    output_dir: Path
    data_root: Path = DEFAULT_DATA_ROOT
    weights: Path = DEFAULT_WEIGHTS
    device: str = "cpu"
    model_id: str = "zero_dce_plus_plus"
    max_images: int | None = None
    visual_limit: int = 10
    metrics: tuple[str, ...] = DEFAULT_METRICS
    skip_metrics: tuple[str, ...] = ()
    measure_memory: bool = True


@dataclass
class AttributionSummary:
    output_dir: Path
    image_count: int
    failure_count: int
    stage_count: int
    files: tuple[Path, ...] = field(default_factory=tuple)


def _resolve(path: Path, base: Path, data_root: Path) -> Path:
    if path.is_absolute():
        return path
    candidates = (
        base / path,
        data_root / path,
        PROJECT_ROOT / path,
        Path.cwd() / path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return (base / path).resolve()


def _flatten_manifest(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    pool = payload.get("pool")
    if isinstance(payload.get("datasets"), list):
        records: list[dict[str, Any]] = []
        for dataset in payload["datasets"]:
            dataset_key = dataset.get("dataset") or dataset.get("key") or "validation"
            for item in dataset.get("records", []):
                record = dict(item)
                record.setdefault("dataset", dataset_key)
                records.append(record)
        return records, pool
    records = payload.get("records")
    if isinstance(records, list):
        return [dict(item) for item in records], pool
    raise ValidationError("Manifest has no 'datasets' or 'records' list.")


def load_validation_pairs(
    manifest_path: Path,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> tuple[list[ValidationPair], dict[str, Any]]:
    """Load only validation records. Reject test/training records."""
    if not manifest_path.is_file():
        raise ValidationError(
            f"Validation manifest not found: {manifest_path}. "
            "Provide an existing validation split; no replacement is created."
        )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records, pool = _flatten_manifest(payload)
    pairs: list[ValidationPair] = []
    seen: set[tuple[Path, Path]] = set()
    base = manifest_path.parent

    for index, record in enumerate(records):
        split = record.get("split", pool)
        if split != "validation":
            raise ValidationError(
                f"Record {index} has split {split!r}; the stage attribution experiment "
                "accepts validation records only."
            )
        low_raw = record.get("low")
        high_raw = record.get("high")
        if not low_raw or not high_raw:
            raise ValidationError(f"Record {index} is missing low/high paths.")
        low = _resolve(Path(str(low_raw)), base, data_root).resolve()
        high = _resolve(Path(str(high_raw)), base, data_root).resolve()
        if not low.is_file() or not high.is_file():
            raise ValidationError(f"Record {index} has missing image files: {low} / {high}")
        key = (low, high)
        if key in seen:
            continue
        seen.add(key)

        with Image.open(low) as low_image, Image.open(high) as high_image:
            if low_image.size != high_image.size:
                raise ValidationError(
                    f"Record {index} size mismatch: {low_image.size} vs {high_image.size}"
                )
        dataset = str(record.get("dataset") or "validation")
        image_id = str(record.get("image_id") or record.get("id") or low.name)
        pairs.append(
            ValidationPair(
                dataset=dataset,
                image_id=image_id,
                low=low,
                high=high,
            )
        )

    if not pairs:
        raise ValidationError("Validation manifest contains no usable paired records.")
    pairs.sort(key=lambda item: (item.dataset, item.image_id))
    return pairs, payload


def _safe(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "__", value).strip("_")
    return cleaned[:160] or "image"


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        upright = ImageOps.exif_transpose(opened)
        image = upright.convert("RGB")
        image.load()
        return image


def _artifact_images(result) -> dict[ArtifactKind, Image.Image]:
    return {
        artifact.descriptor.kind: artifact.image
        for artifact in result.artifacts
        if artifact.available and artifact.image is not None
    }


def _stage_images(
    result, original: Image.Image, reference: Image.Image
) -> dict[str, Image.Image | None]:
    artifacts = _artifact_images(result)
    diagnostics = result.diagnostic_images
    return {
        "original": artifacts.get(ArtifactKind.ORIGINAL),
        "raw_zero_dce": artifacts.get(ArtifactKind.DEEP_LEARNING_OUTPUT),
        "classical": artifacts.get(ArtifactKind.CLASSICAL_OUTPUT),
        "fusion": artifacts.get(ArtifactKind.FUSION_OUTPUT),
        "fusion_edgeguard": diagnostics.get("edge_guard"),
        "fusion_edgeguard_huelock": diagnostics.get("hue_lock"),
        "final": result.final_image,
        "ground_truth": reference,
    }


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) if math.isfinite(float(value)) else None


def _metric_values(
    image: Image.Image,
    reference: Image.Image,
    original: Image.Image,
    metrics: tuple[str, ...],
) -> tuple[dict[str, float | None], dict[str, str]]:
    output = to_float32_rgb(image, name="stage")
    reference_array = to_float32_rgb(reference, name="reference")
    original_array = to_float32_rgb(original, name="original")
    values = evaluate_arrays(
        output,
        reference=reference_array,
        original=original_array,
        keys=metrics,
    )
    result: dict[str, float | None] = {}
    notes: dict[str, str] = {}
    for key, metric in values.items():
        result[key] = _finite(metric.value)
        if metric.notes:
            notes[key] = metric.notes
    return result, notes


def _rss_mb() -> float | None:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / (1024**2)
    except Exception:
        return None


def _csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _grid(
    images: dict[str, Image.Image | None],
    labels: dict[str, str],
    path: Path,
    panel: int = 220,
) -> None:
    keys = [key for key, _ in STAGES] + ["ground_truth"]
    header = 18
    canvas = Image.new("RGB", (panel * len(keys), panel + header), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, key in enumerate(keys):
        image = images.get(key)
        x = index * panel
        draw.text((x + 4, 3), labels.get(key, key), fill=(0, 0, 0), font=font)
        if image is None:
            draw.rectangle(
                (x + 1, header, x + panel - 2, header + panel - 2),
                outline=(220, 220, 220),
            )
            continue
        thumb = ImageOps.contain(image.convert("RGB"), (panel - 4, panel - 4))
        canvas.paste(thumb, (x + 2, header + 2))
        thumb.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")
    canvas.close()


def _aggregate_rows(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        if row.get("missing"):
            continue
        for metric in metrics:
            value = row.get(metric)
            if value is not None:
                grouped.setdefault((row["dataset"], row["stage"], metric), []).append(float(value))

    result: list[dict[str, Any]] = []
    for (dataset, stage, metric), values in sorted(grouped.items()):
        summary = summarise(values)
        if summary is None:
            continue
        result.append(
            {
                "dataset": dataset,
                "stage": stage,
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
    return result


def _dataset_comparison(
    rows: list[dict[str, Any]], metrics: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        if row.get("missing"):
            continue
        for metric in metrics:
            value = row.get(metric)
            if value is not None:
                grouped.setdefault((row["dataset"], row["stage"], metric), []).append(float(value))

    by_dataset: dict[tuple[str, str], dict[str, Any]] = {}
    for (dataset, stage, metric), values in sorted(grouped.items()):
        entry = by_dataset.setdefault(
            (dataset, stage), {"dataset": dataset, "stage": stage, "count": len(values)}
        )
        entry[f"{metric}_mean"] = sum(values) / len(values)
    return list(by_dataset.values())


def _latency_rows(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_image: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[float]] = {}
    for record in records:
        dataset = record["dataset"]
        image_id = record["image_id"]
        for stage, duration in record.get("stage_timings", {}).items():
            per_image.append(
                {
                    "dataset": dataset,
                    "image_id": image_id,
                    "stage": stage,
                    "duration_ms": float(duration),
                    "total_ms": record.get("total_ms"),
                }
            )
            grouped.setdefault((dataset, stage), []).append(float(duration))
    aggregate: list[dict[str, Any]] = []
    for (dataset, stage), values in sorted(grouped.items()):
        summary = summarise(values)
        if summary is None:
            continue
        aggregate.append(
            {
                "dataset": dataset,
                "stage": stage,
                "count": summary.count,
                "mean_ms": summary.mean,
                "std_ms": summary.std,
                "median_ms": summary.median,
                "p95_ms": sorted(values)[max(0, round(0.95 * (len(values) - 1)))],
            }
        )
    return per_image, aggregate


def _write_summary(
    path: Path,
    config: AttributionConfig,
    pairs: list[ValidationPair],
    model_status: ModelStatus | None,
    aggregate: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    stage_images: dict[str, int],
    memory_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Stage attribution",
        "",
        f"- Validation manifest: `{config.validation_manifest}`",
        f"- Validation images: {len(pairs)}",
        f"- Metrics: {', '.join(config.metrics)}",
        f"- Model: {model_status.model_id if model_status else 'injected pipeline'}",
        f"- Device: {model_status.device if model_status else config.device}",
        "- Test data was not used. No validation split was created.",
        "",
        "## Stage aggregate",
        "",
        "| Dataset | Stage | Metric | n | Mean | Std | Median |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['dataset']} | {row['stage']} | {row['metric']} | {row['count']} | "
            f"{row['mean']:.4f} | {row['std']:.4f} | {row['median']:.4f} |"
        )
    lines.extend(["", "## Stage captures", ""])
    for stage, _ in STAGES:
        lines.append(f"- `{stage}`: {stage_images.get(stage, 0)} image(s)")
    if failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures[:20]:
            lines.append(f"- {failure['dataset']} / {failure['image_id']}: {failure['error']}")
    if memory_rows:
        values = [row["after_mb"] for row in memory_rows if row.get("after_mb") is not None]
        if values:
            lines.extend(
                [
                    "",
                    "## Memory",
                    "",
                    f"- Maintained process RSS samples: {len(values)}",
                    f"- Maximum observed RSS: {max(values):.1f} MiB",
                ]
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_attribution(
    config: AttributionConfig,
    *,
    pipeline: EnhancementPipeline | None = None,
    model_status: ModelStatus | None = None,
) -> AttributionSummary:
    pairs, _payload = load_validation_pairs(config.validation_manifest, config.data_root)
    if config.max_images is not None:
        if config.max_images <= 0:
            raise ValueError("max_images must be positive.")
        pairs = pairs[: config.max_images]

    output_dir = config.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new experiment directory; previous outputs are never overwritten."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    owns_manager = False
    manager: ModelManager | None = None
    if pipeline is None:
        if not config.weights.is_file():
            raise FileNotFoundError(
                f"Zero-DCE++ checkpoint not found: {config.weights}. "
                "The experiment does not download weights."
            )
        manager = ModelManager(
            InferenceSettings(
                model_id=config.model_id,
                weights_path=config.weights,
                device=config.device,
                load_on_startup=True,
            )
        )
        model_status = manager.load_at_startup()
        if not model_status.ready:
            manager.close()
            raise RuntimeError(model_status.message)
        pipeline = EnhancementPipeline(model_manager=manager)
        owns_manager = True

    metrics = tuple(metric for metric in config.metrics if metric not in config.skip_metrics)
    if not metrics:
        raise ValueError("At least one metric is required.")

    image_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    latency_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    stage_counts: dict[str, int] = {stage: 0 for stage, _ in STAGES}
    visual_counts: dict[str, int] = {}

    try:
        for pair in pairs:
            low_image = None
            reference_image = None
            result = None
            try:
                low_image = _load_image(pair.low)
                reference_image = _load_image(pair.high)
                if low_image.size != reference_image.size:
                    raise ValueError(
                        f"input {low_image.size} != reference {reference_image.size}"
                    )
                rss_before = _rss_mb() if config.measure_memory else None
                result = pipeline.run(
                    low_image, mode="hybrid", capture_diagnostics=True
                )
                rss_after = _rss_mb() if config.measure_memory else None
                if not result.completed or result.final_image is None:
                    raise RuntimeError(result.message)
                if result.final_image.size != low_image.size:
                    raise RuntimeError(
                        f"pipeline output {result.final_image.size} != input {low_image.size}"
                    )

                stage_images = _stage_images(result, low_image, reference_image)
                safe = _safe(pair.image_id)
                image_dir = output_dir / "stages" / _safe(pair.dataset) / safe
                image_dir.mkdir(parents=True, exist_ok=True)
                for stage, _label in STAGES:
                    image = stage_images.get(stage)
                    if image is None:
                        continue
                    image.convert("RGB").save(image_dir / f"{stage}.png", format="PNG")
                    stage_counts[stage] += 1
                reference_image.save(image_dir / "ground_truth.png", format="PNG")

                for stage, _label in STAGES:
                    image = stage_images.get(stage)
                    if image is None:
                        image_rows.append(
                            {
                                "dataset": pair.dataset,
                                "image_id": pair.image_id,
                                "stage": stage,
                                "missing": True,
                                "notes": "stage output was not produced",
                                **{metric: None for metric in metrics},
                            }
                        )
                        continue
                    values, notes = _metric_values(
                        image, reference_image, low_image, metrics
                    )
                    image_rows.append(
                        {
                            "dataset": pair.dataset,
                            "image_id": pair.image_id,
                            "stage": stage,
                            "missing": False,
                            "notes": json.dumps(notes, ensure_ascii=False) if notes else "",
                            **values,
                        }
                    )

                scan = result.diagnostics.get("scan", {})
                features = scan.get("features", {}) if isinstance(scan, dict) else {}
                parameters = scan.get("parameters", {}) if isinstance(scan, dict) else {}
                weights = result.diagnostics.get("fusion_weights")
                initial = result.diagnostics.get("fusion_weights_initial")
                for label, array in (("refined", weights), ("initial", initial)):
                    if array is None:
                        continue
                    for index, name in enumerate(("original", "classical", "deep_learning")):
                        weight_rows.append(
                            {
                                "dataset": pair.dataset,
                                "image_id": pair.image_id,
                                "weight_set": label,
                                "candidate": name,
                                "mean": float(array[..., index].mean()),
                                "median": float(
                                    np.median(array[..., index])
                                ),
                                "dl_strength": parameters.get("dl_strength"),
                                "darkness": features.get("darkness"),
                                "noise_sigma": features.get("noise_sigma"),
                                "nonuniformity": features.get("nonuniformity"),
                                "colour_cast_a": features.get("colour_cast_a"),
                                "colour_cast_b": features.get("colour_cast_b"),
                                "colour_cast_magnitude": features.get("colour_cast_magnitude"),
                                "output_blend_alpha": parameters.get("output_blend_alpha"),
                                "fusion_contrast": parameters.get("fusion_contrast"),
                                "fusion_saturation": parameters.get("fusion_saturation"),
                                "fusion_exposure": parameters.get("fusion_exposure"),
                                "fusion_noise": parameters.get("fusion_noise"),
                                "fusion_tau": parameters.get("fusion_tau"),
                            }
                        )

                if config.measure_memory:
                    memory_rows.append(
                        {
                            "dataset": pair.dataset,
                            "image_id": pair.image_id,
                            "before_mb": rss_before,
                            "after_mb": rss_after,
                        }
                    )
                latency_records.append(
                    {
                        "dataset": pair.dataset,
                        "image_id": pair.image_id,
                        "stage_timings": dict(result.timings_ms),
                        "total_ms": result.total_ms,
                    }
                )

                if (
                    config.visual_limit > 0
                    and visual_counts.get(pair.dataset, 0) < config.visual_limit
                ):
                    _grid(
                        stage_images,
                        dict(STAGES) | {"ground_truth": "Ground Truth"},
                        output_dir / "visuals" / _safe(pair.dataset) / f"{safe}_grid.png",
                    )
                    visual_counts[pair.dataset] = visual_counts.get(pair.dataset, 0) + 1
            except Exception as exc:
                failures.append(
                    {
                        "dataset": pair.dataset,
                        "image_id": pair.image_id,
                        "error": str(exc),
                    }
                )
            finally:
                if result is not None:
                    result.release()
                if low_image is not None:
                    low_image.close()
                if reference_image is not None:
                    reference_image.close()
    finally:
        if owns_manager and manager is not None:
            manager.close()

    metric_fields = list(metrics)
    per_image_fields = [
        "dataset",
        "image_id",
        "stage",
        "missing",
        *metric_fields,
        "notes",
    ]
    _csv(output_dir / "per_image_metrics.csv", image_rows, per_image_fields)

    aggregate = _aggregate_rows(image_rows, metrics)
    _csv(
        output_dir / "stage_aggregate.csv",
        aggregate,
        [
            "dataset",
            "stage",
            "metric",
            "count",
            "mean",
            "std",
            "median",
            "p10",
            "p90",
            "min",
            "max",
        ],
    )
    comparison = _dataset_comparison(image_rows, metrics)
    comparison_fields = ["dataset", "stage", "count", *[f"{m}_mean" for m in metrics]]
    _csv(output_dir / "dataset_comparison.csv", comparison, comparison_fields)

    weight_fields = [
        "dataset",
        "image_id",
        "weight_set",
        "candidate",
        "mean",
        "median",
        "dl_strength",
        "darkness",
        "noise_sigma",
        "nonuniformity",
        "colour_cast_a",
        "colour_cast_b",
        "colour_cast_magnitude",
        "output_blend_alpha",
        "fusion_contrast",
        "fusion_saturation",
        "fusion_exposure",
        "fusion_noise",
        "fusion_tau",
    ]
    _csv(output_dir / "fusion_weights.csv", weight_rows, weight_fields)

    latency_per_image, latency_aggregate = _latency_rows(latency_records)
    _csv(
        output_dir / "stage_latency.csv",
        latency_per_image,
        ["dataset", "image_id", "stage", "duration_ms", "total_ms"],
    )
    _csv(
        output_dir / "stage_latency_aggregate.csv",
        latency_aggregate,
        ["dataset", "stage", "count", "mean_ms", "std_ms", "median_ms", "p95_ms"],
    )

    _csv(
        output_dir / "memory.csv",
        memory_rows,
        ["dataset", "image_id", "before_mb", "after_mb"],
    )
    _csv(
        output_dir / "failures.csv",
        failures,
        ["dataset", "image_id", "error"],
    )
    _write_summary(
        output_dir / "summary.md",
        config,
        pairs,
        model_status,
        aggregate,
        failures,
        stage_counts,
        memory_rows,
    )
    run_config = {
        "validation_manifest": str(config.validation_manifest),
        "data_root": str(config.data_root),
        "output_dir": str(output_dir),
        "metrics": list(metrics),
        "skipped_metrics": list(config.skip_metrics),
        "device": model_status.device if model_status else config.device,
        "model_id": model_status.model_id if model_status else config.model_id,
        "weights": str(config.weights),
        "image_count": len(pairs),
        "failure_count": len(failures),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return AttributionSummary(
        output_dir=output_dir,
        image_count=len(pairs),
        failure_count=len(failures),
        stage_count=len(STAGES),
        files=tuple(output_dir.rglob("*")),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-id", default="zero_dce_plus_plus")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--visual-limit", type=int, default=10)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--skip-metrics", default="")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        pairs, payload = load_validation_pairs(args.validation_manifest, args.data_root)
    except (OSError, ValueError) as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        return 2

    print(f"validation records: {len(pairs)}")
    print(f"datasets: {sorted({pair.dataset for pair in pairs})}")
    print(f"manifest pool: {payload.get('pool')}")
    if args.dry_run:
        return 0

    config = AttributionConfig(
        validation_manifest=args.validation_manifest,
        output_dir=args.output_dir,
        data_root=args.data_root,
        weights=args.weights,
        device=args.device,
        model_id=args.model_id,
        max_images=args.max_images,
        visual_limit=args.visual_limit,
        metrics=tuple(metric.strip() for metric in args.metrics.split(",") if metric.strip()),
        skip_metrics=tuple(
            metric.strip()
            for metric in args.skip_metrics.split(",")
            if metric.strip()
        ),
        measure_memory=not args.no_memory,
    )
    try:
        summary = run_attribution(config)
    except Exception as exc:
        print(f"Attribution failed: {exc}", file=sys.stderr)
        return 1
    print(f"images: {summary.image_count}, failures: {summary.failure_count}")
    print(f"output: {summary.output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
