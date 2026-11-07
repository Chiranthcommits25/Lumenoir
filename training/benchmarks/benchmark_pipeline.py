"""Benchmark the full Lumenoir pipeline on CPU and accelerator devices.

Example (CPU only):

    python -m training.benchmarks.benchmark_pipeline \
        --devices cpu --sizes 256x256,640x480 --repeat 2

Example (CUDA T4):

    python -m training.benchmarks.benchmark_pipeline \
        --devices cuda --precision float32 --repeat 5

The script reports measured stage durations from the pipeline, end-to-end wall
time, and peak memory. It prints a table and writes JSON; it does not fabricate
values for unavailable devices.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import backend.deep_learning  # noqa: E402,F401  (register the model)
from backend.config import InferenceSettings  # noqa: E402
from backend.enhancement.pipeline import EnhancementPipeline  # noqa: E402
from backend.inference.manager import ModelManager  # noqa: E402
from backend.runtime.config import resolve_runtime  # noqa: E402

DEFAULT_SIZES = "256x256,640x480,1280x960,2304x1728"
DEFAULT_DEVICES = "cpu"
DEFAULT_WEIGHTS = PROJECT_ROOT / "backend" / "weights" / "zero_dce_plus_plus.pth"
MODEL_ID = "zero_dce_plus_plus"


def _parse_sizes(raw: str) -> list[tuple[int, int]]:
    sizes = []
    for item in raw.split(","):
        part = item.strip().lower()
        if not part:
            continue
        try:
            width, height = part.split("x", 1)
            sizes.append((int(width), int(height)))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid size {item!r}; use WIDTHxHEIGHT."
            ) from exc
    if not sizes:
        raise argparse.ArgumentTypeError("At least one size is required.")
    return sizes


def _synthetic_image(width: int, height: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.stack(
        [
            0.03 + 0.16 * xx / width,
            0.025 + 0.12 * yy / height,
            0.02 + 0.09 * (xx + yy) / (width + height),
        ],
        axis=-1,
    )
    texture = rng.normal(0.0, 0.008, base.shape)
    image = np.clip(base + texture, 0.0, 1.0)
    return Image.fromarray(np.round(image * 255.0).astype(np.uint8))


def _cpu_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _peak_rss_bytes() -> int | None:
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if platform.system() == "Darwin" else value * 1024
    except Exception:
        return None


def _accelerator_memory(torch) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        if torch.cuda.is_available():
            result["cuda_max_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated()
            )
            result["cuda_max_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved()
            )
    except Exception:
        pass
    try:
        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None)
        if mps is not None and mps.is_available():
            result["mps_current_allocated_bytes"] = int(
                torch.mps.current_allocated_memory()
            )
            result["mps_driver_allocated_bytes"] = int(
                torch.mps.driver_allocated_memory()
            )
    except Exception:
        pass
    return result


def _reset_peak_memory(runtime) -> None:
    if runtime.is_cuda:
        runtime.reset_peak_memory()
    try:
        import torch

        if runtime.is_mps:
            torch.mps.empty_cache()
    except Exception:
        pass


def _run_once(
    pipeline: EnhancementPipeline,
    image: Image.Image,
    *,
    mode: str,
    runtime,
) -> dict[str, object]:
    gc.collect()
    if runtime.is_cuda or runtime.is_mps:
        runtime.synchronize()
    _reset_peak_memory(runtime)
    started = time.perf_counter()
    result = pipeline.run(image, mode=mode)
    if runtime.is_cuda or runtime.is_mps:
        runtime.synchronize()
    wall_ms = (time.perf_counter() - started) * 1000.0
    record: dict[str, object] = {
        "status": result.status.value,
        "wall_ms": round(wall_ms, 3),
        "pipeline_total_ms": result.total_ms,
        "stage_ms": dict(result.timings_ms),
        "cpu_rss_bytes": _cpu_rss_bytes(),
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    if runtime.backend == "torch":
        record["accelerator_memory"] = _accelerator_memory(runtime.torch)
    result.release()
    return record


def benchmark_device(
    device: str,
    precision: str,
    sizes: list[tuple[int, int]],
    *,
    mode: str,
    warmup: int,
    repeat: int,
    weights: Path,
) -> dict[str, object]:
    runtime = resolve_runtime(device, precision)
    if not weights.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at {weights}; benchmark requires the model."
        )
    manager = ModelManager(
        InferenceSettings(
            model_id=MODEL_ID,
            weights_path=weights,
            device=runtime.device,
            precision=precision,
        )
    )
    load_started = time.perf_counter()
    status = manager.load()
    model_load_ms = (time.perf_counter() - load_started) * 1000.0
    if not status.ready:
        manager.close()
        raise RuntimeError(f"Model is not ready: {status.message}")

    pipeline = EnhancementPipeline(
        model_manager=manager, runtime=runtime
    )
    records = []
    try:
        for width, height in sizes:
            image = _synthetic_image(width, height, seed=width * 1000 + height)
            for _ in range(max(0, warmup)):
                _run_once(pipeline, image, mode=mode, runtime=runtime)
            runs = [
                _run_once(pipeline, image, mode=mode, runtime=runtime)
                for _ in range(max(1, repeat))
            ]
            walls = [float(run["wall_ms"]) for run in runs]
            totals = [float(run["pipeline_total_ms"]) for run in runs]
            stage_names = sorted(
                {name for run in runs for name in run["stage_ms"]}  # type: ignore[union-attr]
            )
            stages = {
                name: {
                    "median_ms": round(
                        statistics.median(
                            float(run["stage_ms"].get(name, 0.0))  # type: ignore[union-attr]
                            for run in runs
                        ),
                        3,
                    ),
                    "min_ms": round(
                        min(
                            float(run["stage_ms"].get(name, 0.0))  # type: ignore[union-attr]
                            for run in runs
                        ),
                        3,
                    ),
                    "max_ms": round(
                        max(
                            float(run["stage_ms"].get(name, 0.0))  # type: ignore[union-attr]
                            for run in runs
                        ),
                        3,
                    ),
                }
                for name in stage_names
            }
            records.append(
                {
                    "device": runtime.device,
                    "backend": runtime.backend,
                    "precision": runtime.precision,
                    "mode": mode,
                    "width": width,
                    "height": height,
                    "megapixels": round(width * height / 1_000_000.0, 3),
                    "runs": runs,
                    "summary": {
                        "wall_median_ms": round(statistics.median(walls), 3),
                        "wall_min_ms": round(min(walls), 3),
                        "wall_max_ms": round(max(walls), 3),
                        "pipeline_total_median_ms": round(
                            statistics.median(totals), 3
                        ),
                        "stages_median_ms": stages,
                        "peak_rss_bytes": max(
                            int(run["peak_rss_bytes"] or 0) for run in runs
                        ),
                        "accelerator_memory": runs[-1].get("accelerator_memory"),
                    },
                }
            )
            image.close()
    finally:
        manager.close()

    return {
        "requested_device": device,
        "resolved_device": runtime.device,
        "backend": runtime.backend,
        "precision": runtime.precision,
        "reason": runtime.reason,
        "model_load_ms": round(model_load_ms, 3),
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default=DEFAULT_DEVICES)
    parser.add_argument("--sizes", default=DEFAULT_SIZES)
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "training"
        / "benchmarks"
        / "results"
        / "pipeline_benchmark.json",
    )
    args = parser.parse_args(argv)
    sizes = _parse_sizes(args.sizes)
    devices = [part.strip() for part in args.devices.split(",") if part.strip()]

    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mode": args.mode,
        "precision": args.precision,
        "sizes": [f"{w}x{h}" for w, h in sizes],
        "warmup": args.warmup,
        "repeat": args.repeat,
        "devices": [],
    }
    for device in devices:
        try:
            result = benchmark_device(
                device,
                args.precision,
                sizes,
                mode=args.mode,
                warmup=args.warmup,
                repeat=args.repeat,
                weights=args.weights,
            )
        except Exception as exc:
            print(f"[{device}] unavailable: {exc}", file=sys.stderr)
            report["devices"].append(
                {"requested_device": device, "error": str(exc)}
            )
            continue
        report["devices"].append(result)
        print(
            f"\n{result['requested_device']} -> {result['resolved_device']} "
            f"({result['backend']}/{result['precision']}) "
            f"model_load={result['model_load_ms']:.1f} ms"
        )
        print(
            f"{'size':>12} {'wall_med':>10} {'pipeline_med':>13} "
            f"{'peak_rss':>12} {'stages (median ms)'}"
        )
        for record in result["records"]:
            summary = record["summary"]
            stages = ", ".join(
                f"{name}={values['median_ms']:.1f}"
                for name, values in summary["stages_median_ms"].items()
            )
            peak = summary["peak_rss_bytes"] / (1024 * 1024)
            print(
                f"{record['width']}x{record['height']:<6} "
                f"{summary['wall_median_ms']:>10.1f} "
                f"{summary['pipeline_total_median_ms']:>13.1f} "
                f"{peak:>9.1f} MB {stages}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
