"""Benchmark LumenFuse: OpenCV CPU vs torch FP32/FP16.

The benchmark uses identical candidates and scan results for every backend.
Cold start is the first call for a configuration in this process; the first
torch accelerator call additionally includes device-context initialization.
Timed CUDA runs are synchronized with CUDA events (via ``synchronize``) so only
kernel-launch time is not reported.

    python -m training.benchmarks.benchmark_lumen_fuse \
        --device auto --output-dir training/reports/lumen_fuse_benchmark

On a CPU-only machine only the CPU backend runs; the script prints that no
accelerator was available. On Kaggle T4, ``--device cuda`` measures the GPU
backend directly.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from backend.classical_cv.scan import scan
from backend.fusion.lumen_fuse import fuse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "datasets" / "raw" / "LOL-v2-Real" / "Test" / "Low"
SIZES: dict[str, tuple[int, int]] = {
    "small": (256, 256),
    "medium": (640, 480),
    "large": (1280, 960),
    "representative": (2304, 1728),
}


@dataclass(frozen=True)
class BackendSpec:
    key: str
    backend: str
    device: str | None
    precision: str


class _GpuSampler(threading.Thread):
    def __init__(self, torch_module, device_index: int = 0) -> None:
        super().__init__(daemon=True)
        self.torch = torch_module
        self.device_index = device_index
        self.values: list[float] = []
        self._stop = threading.Event()

    def run(self) -> None:  # pragma: no cover - timing dependent
        if not hasattr(self.torch.cuda, "utilization"):
            return
        while not self._stop.wait(0.05):
            try:
                value = self.torch.cuda.utilization(self.device_index)
            except Exception:
                return
            if value is not None:
                self.values.append(float(value))

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=1.0)


def _load_base_image() -> np.ndarray:
    if DEFAULT_IMAGE_DIR.is_dir():
        candidates = sorted(DEFAULT_IMAGE_DIR.glob("*.png"))
        if candidates:
            with Image.open(candidates[0]) as image:
                return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    rng = np.random.default_rng(0)
    return rng.random((64, 64, 3)).astype(np.float32)


def _candidates(
    base: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    original = cv2.resize(base, (width, height), interpolation=cv2.INTER_CUBIC)
    original = np.clip(original, 0.0, 1.0).astype(np.float32)
    classical = np.clip(original**0.55 * 1.15, 0.0, 1.0).astype(np.float32)
    deep = np.clip(original**0.7 * 1.05, 0.0, 1.0).astype(np.float32)
    return original, classical, deep


def _sync(device: str | None) -> None:
    if device is None:
        return
    if device.startswith("cuda"):
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    elif device == "mps":
        try:
            import torch

            torch.mps.synchronize()
        except Exception:
            pass


def _reset_peak_memory(device: str | None) -> None:
    if device and device.startswith("cuda"):
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


def _peak_memory_mb(device: str | None) -> float | None:
    if device and device.startswith("cuda"):
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024**2), 3)
    return None


def _mps_memory_mb() -> float | None:
    try:
        import torch

        return round(torch.mps.current_allocated_memory() / (1024**2), 3)
    except Exception:
        return None


def _quality(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    delta = np.abs(reference - candidate)
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            psnr = float(
                peak_signal_noise_ratio(reference, candidate, data_range=1.0)
            )
        ssim = float(
            structural_similarity(
                reference, candidate, channel_axis=-1, data_range=1.0
            )
        )
    except Exception:
        psnr = math.inf if float(delta.max()) == 0.0 else float("nan")
        ssim = 1.0 if float(delta.max()) == 0.0 else float("nan")
    return {
        "mae": float(delta.mean()),
        "max_abs_diff": float(delta.max()),
        "psnr": psnr,
        "ssim": ssim,
    }


def _run_config(
    spec: BackendSpec,
    original: np.ndarray,
    classical: np.ndarray,
    deep: np.ndarray,
    scan_result,
    reference_image: np.ndarray,
    reference_weights: np.ndarray,
    *,
    warmups: int,
    runs: int,
    cold_ms: float | None = None,
) -> dict[str, Any]:
    device = spec.device
    if cold_ms is None:
        try:
            _sync(device)
            started = time.perf_counter()
            fuse(
                original,
                classical,
                deep,
                scan_result=scan_result,
                backend=spec.backend,
                device=device,
                precision=spec.precision,
            )
            _sync(device)
            cold_ms = (time.perf_counter() - started) * 1000.0
        except Exception as exc:
            return {"backend": spec.key, "error": str(exc)}

    warmup_times: list[float] = []
    try:
        for _ in range(warmups):
            _sync(device)
            started = time.perf_counter()
            fuse(
                original,
                classical,
                deep,
                scan_result=scan_result,
                backend=spec.backend,
                device=device,
                precision=spec.precision,
            )
            _sync(device)
            warmup_times.append((time.perf_counter() - started) * 1000.0)
    except Exception as exc:
        return {"backend": spec.key, "error": str(exc)}

    sampler = None
    if device and device.startswith("cuda"):
        import torch

        sampler = _GpuSampler(torch)
        sampler.start()
    mps_peak = 0.0
    _reset_peak_memory(device)
    durations: list[float] = []
    cpu_started = time.process_time()
    wall_started = time.perf_counter()

    for _ in range(runs):
        _sync(device)
        started = time.perf_counter()
        result = fuse(
            original,
            classical,
            deep,
            scan_result=scan_result,
            backend=spec.backend,
            device=device,
            precision=spec.precision,
        )
        _sync(device)
        durations.append((time.perf_counter() - started) * 1000.0)
        if device == "mps":
            mps_peak = max(mps_peak, _mps_memory_mb() or 0.0)

    wall = time.perf_counter() - wall_started
    cpu_time = time.process_time() - cpu_started
    if sampler is not None:
        sampler.stop()

    peak = _peak_memory_mb(device)
    if peak is None and device == "mps":
        peak = mps_peak if mps_peak > 0.0 else None

    ordered = np.asarray(durations, dtype=np.float64)
    quality = _quality(reference_image, result.image)
    weight_quality = _quality(reference_weights, result.weights)

    return {
        "backend": spec.key,
        "device": device or "cpu",
        "precision": spec.precision,
        "cold_start_ms": cold_ms,
        "warmup_mean_ms": statistics.mean(warmup_times) if warmup_times else 0.0,
        "median_ms": float(np.median(ordered)),
        "mean_ms": float(np.mean(ordered)),
        "p90_ms": float(np.percentile(ordered, 90.0)),
        "min_ms": float(np.min(ordered)),
        "max_ms": float(np.max(ordered)),
        "peak_memory_mb": peak,
        "cpu_utilization_percent": (
            round(100.0 * cpu_time / wall, 3) if wall > 0.0 else None
        ),
        "gpu_utilization_percent_mean": (
            round(statistics.mean(sampler.values), 3)
            if sampler is not None and sampler.values
            else None
        ),
        "gpu_utilization_percent_max": (
            round(max(sampler.values), 3)
            if sampler is not None and sampler.values
            else None
        ),
        "image_quality": quality,
        "weights_mae": weight_quality["mae"],
        "weights_max_abs_diff": weight_quality["max_abs_diff"],
    }


def _backend_specs(requested_device: str) -> tuple[list[BackendSpec], str]:
    specs = [BackendSpec("cpu", "cpu", None, "float32")]
    device = requested_device.strip().lower() or "auto"
    if device == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except Exception:
            device = "cpu"

    if device == "cpu":
        return specs, device

    specs.append(BackendSpec(f"torch-fp32-{device}", "torch", device, "float32"))
    specs.append(BackendSpec(f"torch-fp16-{device}", "torch", device, "float16"))
    return specs, device


def _device_label(device: str) -> str:
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            index = 0
            if ":" in device:
                index = int(device.split(":", 1)[1])
            return f"{device} ({torch.cuda.get_device_name(index)})"
        if device == "mps":
            return "mps (Apple Metal)"
    except Exception:
        pass
    return device


def run_benchmark(
    *,
    output_dir: Path,
    device: str,
    warmups: int,
    runs: int,
    sizes: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    base = _load_base_image()
    specs, resolved_device = _backend_specs(device)
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "device": _device_label(resolved_device),
        "configured_device": device,
        "sizes": {name: {"width": w, "height": h} for name, (w, h) in sizes.items()},
        "warmups": warmups,
        "runs": runs,
        "results": {},
    }

    for name, (width, height) in sizes.items():
        original, classical, deep = _candidates(base, width, height)
        scan_result = scan(original)
        started = time.perf_counter()
        reference = fuse(
            original, classical, deep, scan_result=scan_result, backend="cpu"
        )
        cpu_cold_ms = (time.perf_counter() - started) * 1000.0
        rows = []
        for spec in specs:
            rows.append(
                _run_config(
                    spec,
                    original,
                    classical,
                    deep,
                    scan_result,
                    reference.image,
                    reference.weights,
                    warmups=warmups,
                    runs=runs,
                    cold_ms=cpu_cold_ms if spec.backend == "cpu" else None,
                )
            )
        report["results"][name] = rows
        print(
            f"{name} {width}x{height}: "
            + ", ".join(
                f"{row['backend']}={row.get('median_ms', row.get('error'))}"
                for row in rows
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_lumen_fuse.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "benchmark_lumen_fuse.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# LumenFuse backend benchmark",
        "",
        f"- Platform: {report['platform']}",
        f"- Device: {report['device']}",
        f"- Warm-ups: {report['warmups']}, timed runs: {report['runs']}",
        "",
    ]
    for name, rows in report["results"].items():
        size = report["sizes"][name]
        lines.extend(
            [
                f"## {name} ({size['width']}x{size['height']})",
                "",
                "| Backend | Cold ms | Warmup ms | Median ms | Mean ms | P90 ms | "
                "Peak MB | CPU % | GPU % | MAE | Max diff | PSNR | SSIM |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            if "error" in row:
                lines.append(f"| {row['backend']} | error: {row['error']} |")
                continue
            image = row["image_quality"]
            lines.append(
                f"| {row['backend']} | {row['cold_start_ms']:.2f} | "
                f"{row['warmup_mean_ms']:.2f} | {row['median_ms']:.2f} | "
                f"{row['mean_ms']:.2f} | {row['p90_ms']:.2f} | "
                f"{_number(row['peak_memory_mb'])} | "
                f"{_number(row['cpu_utilization_percent'])} | "
                f"{_number(row['gpu_utilization_percent_mean'])} | "
                f"{image['mae']:.3e} | {image['max_abs_diff']:.3e} | "
                f"{_number(image['psnr'])} | {_number(image['ssim'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "CPU is the existing OpenCV backend. Torch timings include the",
            "CPU/GPU input copy and the final output copy; CUDA timings are",
            "synchronized. Cold start for the first torch configuration also",
            "includes CUDA/MPS context initialization.",
            "",
        ]
    )
    return "\n".join(lines)


def _number(value: float | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cuda, cuda:N, mps or cpu (torch backend device).",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--sizes",
        default=",".join(SIZES),
        help="Comma-separated subset of small,medium,large,representative.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected = {
        name: SIZES[name]
        for name in (item.strip() for item in args.sizes.split(","))
        if name in SIZES
    }
    if not selected:
        raise SystemExit("No valid sizes selected.")
    output_dir = args.output_dir or (
        PROJECT_ROOT / "training" / "reports" / "lumen_fuse_benchmark"
    )
    report = run_benchmark(
        output_dir=output_dir,
        device=args.device,
        warmups=max(0, args.warmups),
        runs=max(1, args.runs),
        sizes=selected,
    )
    print(f"report: {output_dir / 'benchmark_lumen_fuse.md'}")
    return 0 if report["results"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
