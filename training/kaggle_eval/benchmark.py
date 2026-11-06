"""A/B benchmark: previous per-mode recomputation vs shared prefix stages.

The benchmark runs the same evaluation harness twice over the same images:

* ``old`` — ``share_stages=False`` reproduces the previous behaviour, where
  every mode recomputes LumenScan, LumenRetinex and Zero-DCE++ independently.
* ``new`` — ``share_stages=True`` computes the mode-independent prefix once per
  image and replays it for the remaining modes.

It reports total wall time, per-mode recorded processing time, execution counts,
peak GPU memory and output equivalence (metrics and saved before/after visuals).

Run it on the target machine (for example a Kaggle T4) with the same checkpoint
and dataset configuration used for evaluation:

    python -m training.kaggle_eval.benchmark --config eval_config.json --images 10

The script never downloads data, tunes parameters or changes enhancement code.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from backend.enhancement import stages as stage_module
from backend.inference.manager import ModelManager

from .config import EvalConfig
from .runner import run_evaluation

_EXECUTION_PATCHES: tuple[tuple[str, Any, str], ...] = (
    ("lumen_scan", stage_module, "scan"),
    ("classical_stage", stage_module, "retinex_enhance"),
    ("lumen_fuse", stage_module, "fuse_candidates"),
    ("edge_guard", stage_module, "edge_guard_denoise"),
    ("hue_lock", stage_module, "hue_lock_correct"),
)


@contextmanager
def _count_executions():
    """Count the algorithm calls made by the harness without altering them."""
    counts: Counter[str] = Counter()
    restore: list[tuple[Any, str, Any]] = []

    def wrap(target: Any, name: str, label: str) -> None:
        original = getattr(target, name)

        def wrapper(*args, **kwargs):
            counts[label] += 1
            return original(*args, **kwargs)

        setattr(target, name, wrapper)
        restore.append((target, name, original))

    for label, module, name in _EXECUTION_PATCHES:
        wrap(module, name, label)
    wrap(ModelManager, "infer", "model_inference")

    try:
        yield counts
    finally:
        for target, name, original in reversed(restore):
            setattr(target, name, original)


def _gpu_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _gpu_reset() -> None:
    if not _gpu_available():
        return
    import torch

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _gpu_peak() -> dict[str, float] | None:
    if not _gpu_available():
        return None
    import torch

    torch.cuda.synchronize()
    return {
        "allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 3),
        "reserved_mb": round(torch.cuda.max_memory_reserved() / (1024**2), 3),
    }


def _load_records(output_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    path = output_dir / "records" / "records.jsonl"
    if not path.is_file():
        return {}
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[(row["dataset"], row["image_id"], row["mode"])] = row
    return rows


def _compare_metrics(
    old: dict[tuple[str, str, str], dict[str, Any]],
    new: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    records_compared = 0
    values_compared = 0
    mismatches = 0
    max_abs = 0.0

    for key, old_row in old.items():
        new_row = new.get(key)
        if new_row is None:
            mismatches += 1
            continue
        records_compared += 1
        for metric, old_value in old_row.get("metrics", {}).items():
            new_value = new_row.get("metrics", {}).get(metric)
            if old_value is None or new_value is None:
                if old_value is not new_value:
                    mismatches += 1
                continue
            old_float = float(old_value)
            new_float = float(new_value)
            if not math.isfinite(old_float) or not math.isfinite(new_float):
                both_nan = math.isnan(old_float) and math.isnan(new_float)
                if not (both_nan or old_float == new_float):
                    mismatches += 1
                continue
            delta = abs(old_float - new_float)
            max_abs = max(max_abs, delta)
            values_compared += 1
            if delta > 1e-9:
                mismatches += 1

    return {
        "records_compared": records_compared,
        "metric_values_compared": values_compared,
        "mismatches": mismatches,
        "max_abs_difference": max_abs,
        "identical": records_compared > 0 and mismatches == 0,
    }


def _visual_paths(root: Path) -> dict[str, Path]:
    visuals = root / "visuals"
    if not visuals.is_dir():
        return {}
    return {
        path.relative_to(visuals).as_posix(): path
        for path in sorted(visuals.rglob("*.png"))
    }


def _enhanced_half(path: Path) -> np.ndarray:
    """Return the right-hand (enhanced) half of a saved before/after visual."""
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    width = (array.shape[1] - 2) // 2
    return array[:, width + 2 :, :]


def _compare_visuals(old_root: Path, new_root: Path) -> dict[str, Any]:
    old = _visual_paths(old_root)
    new = _visual_paths(new_root)
    names = sorted(set(old) & set(new))

    max_mae = 0.0
    min_psnr = math.inf
    min_ssim = math.inf
    mismatched_shapes = 0

    for name in names:
        a = _enhanced_half(old[name])
        b = _enhanced_half(new[name])
        if a.shape != b.shape:
            mismatched_shapes += 1
            continue
        mae = float(np.mean(np.abs(a - b)))
        max_mae = max(max_mae, mae)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            psnr = float(peak_signal_noise_ratio(a, b, data_range=1.0))
        ssim = float(structural_similarity(a, b, channel_axis=-1, data_range=1.0))
        min_psnr = min(min_psnr, psnr)
        min_ssim = min(min_ssim, ssim)

    return {
        "visuals_compared": len(names),
        "shape_mismatches": mismatched_shapes,
        "max_mae": max_mae,
        "min_psnr": None if math.isinf(min_psnr) else min_psnr,
        "min_ssim": None if math.isinf(min_ssim) else min_ssim,
        "identical": (
            len(names) > 0
            and not mismatched_shapes
            and max_mae == 0.0
            and (math.isinf(min_psnr) or min_psnr > 60.0)
            and (math.isinf(min_ssim) or min_ssim > 0.999999)
        ),
    }


def _mode_processing_ms(
    records: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, float]:
    grouped: dict[str, float] = defaultdict(float)
    for row in records.values():
        if row.get("success") and row.get("processing_time_ms") is not None:
            grouped[row["mode"]] += float(row["processing_time_ms"])
    return {mode: round(value, 3) for mode, value in sorted(grouped.items())}


def _run_half(
    config: EvalConfig,
    *,
    share_stages: bool,
    output_dir: Path,
) -> dict[str, Any]:
    run_config = config.replace(
        output_dir=output_dir,
        share_stages=share_stages,
        latency_warmups=0,
        latency_runs=0,
        save_visuals=True,
        visual_limit=max(config.max_images or 1, 10) * max(len(config.modes), 1),
    )

    _gpu_reset()
    with _count_executions() as counts:
        started = time.perf_counter()
        summary = run_evaluation(run_config)
        elapsed = time.perf_counter() - started
    peak = _gpu_peak()

    records = _load_records(output_dir)
    return {
        "output_dir": str(output_dir),
        "total_runtime_s": round(elapsed, 3),
        "records": summary.record_count,
        "failures": summary.failure_count,
        "per_mode_processing_ms": _mode_processing_ms(records),
        "executions": dict(sorted(counts.items())),
        "peak_gpu_memory": peak,
        "_records": records,
    }


def run_benchmark(
    config: EvalConfig, output_dir: Path, *, warmup: bool = True
) -> dict[str, Any]:
    if config.max_images is None:
        raise ValueError("Set max_images (or --images) so both halves use the same images.")

    if warmup:
        # Warm lazy imports, OpenCV and the checkpoint loader before measuring.
        _run_half(
            config.replace(max_images=1),
            share_stages=True,
            output_dir=output_dir / "warmup",
        )

    old = _run_half(config, share_stages=False, output_dir=output_dir / "old")
    new = _run_half(config, share_stages=True, output_dir=output_dir / "new")

    old_records = old.pop("_records")
    new_records = new.pop("_records")

    report = {
        "images": config.max_images,
        "modes": list(config.modes),
        "device": config.device,
        "weights": None if config.weights is None else str(config.weights),
        "old": old,
        "new": new,
        "speedup": (
            round(old["total_runtime_s"] / new["total_runtime_s"], 3)
            if new["total_runtime_s"] > 0.0
            else None
        ),
        "metric_equivalence": _compare_metrics(old_records, new_records),
        "visual_equivalence": _compare_visuals(output_dir / "old", output_dir / "new"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "benchmark.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(report: dict[str, Any]) -> str:
    old = report["old"]
    new = report["new"]
    metrics = report["metric_equivalence"]
    visuals = report["visual_equivalence"]

    lines = [
        "# Shared-stage evaluation benchmark",
        "",
        f"- Images: {report['images']}",
        f"- Modes: {', '.join(report['modes'])}",
        f"- Device: {report['device']}",
        f"- Weights: {report['weights'] or 'not configured'}",
        "",
        "| Metric | Old (per-mode recompute) | New (shared prefix) |",
        "| --- | ---: | ---: |",
        f"| Total runtime (s) | {old['total_runtime_s']:.3f} | {new['total_runtime_s']:.3f} |",
        f"| Speed-up | — | {report['speedup']}x |",
        f"| Records | {old['records']} | {new['records']} |",
        f"| Failures | {old['failures']} | {new['failures']} |",
        f"| Model inference calls | {old['executions'].get('model_inference', 0)} | "
        f"{new['executions'].get('model_inference', 0)} |",
        f"| LumenScan executions | {old['executions'].get('lumen_scan', 0)} | "
        f"{new['executions'].get('lumen_scan', 0)} |",
        f"| Classical stage executions | {old['executions'].get('classical_stage', 0)} | "
        f"{new['executions'].get('classical_stage', 0)} |",
        f"| LumenFuse executions | {old['executions'].get('lumen_fuse', 0)} | "
        f"{new['executions'].get('lumen_fuse', 0)} |",
        f"| Peak GPU allocated (MB) | {_gpu_text(old)} | {_gpu_text(new)} |",
        "",
        "## Per-mode recorded processing time (ms, sum over images)",
        "",
        "| Mode | Old | New |",
        "| --- | ---: | ---: |",
    ]
    for mode in sorted(set(old["per_mode_processing_ms"]) | set(new["per_mode_processing_ms"])):
        lines.append(
            f"| {mode} | {old['per_mode_processing_ms'].get(mode, 0.0):.3f} | "
            f"{new['per_mode_processing_ms'].get(mode, 0.0):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Output equivalence",
            "",
            f"- Metric values compared: {metrics['metric_values_compared']}",
            f"- Metric mismatches: {metrics['mismatches']}",
            f"- Max metric difference: {metrics['max_abs_difference']:.3e}",
            f"- Visuals compared: {visuals['visuals_compared']}",
            f"- Visual max MAE: {visuals['max_mae']:.3e}",
            f"- Visual min PSNR: {visuals['min_psnr']}",
            f"- Visual min SSIM: {visuals['min_ssim']}",
            f"- Identical outputs: {metrics['identical'] and visuals['identical']}",
            "",
            "Per-mode processing time is the existing stage-attribution sum. Shared",
            "stages are measured once and replayed into each mode, so this value",
            "preserves the previous latency semantics; total runtime and execution",
            "counts show the actual work removed.",
            "",
        ]
    )
    return "\n".join(lines)


def _gpu_text(half: dict[str, Any]) -> str:
    peak = half.get("peak_gpu_memory")
    if not peak:
        return "n/a"
    return f"{peak['allocated_mb']:.1f} (reserved {peak['reserved_mb']:.1f})"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--images", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = EvalConfig.from_file(args.config)
        changes: dict[str, Any] = {"max_images": args.images}
        if args.device is not None:
            changes["device"] = args.device
        config = config.replace(**changes)
        output_dir = args.output_dir or (config.output_dir / "shared-benchmark")
        report = run_benchmark(config, output_dir)
    except Exception as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"images={report['images']} old={report['old']['total_runtime_s']:.2f}s "
        f"new={report['new']['total_runtime_s']:.2f}s speedup={report['speedup']}x"
    )
    print(f"report: {output_dir / 'benchmark.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
