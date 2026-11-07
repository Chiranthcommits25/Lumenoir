"""GPU-only benchmark for the complete Lumenoir enhancement pipeline.

Runs every implemented stage on NVIDIA CUDA through the existing
``EnhancementPipeline`` — LumenScan, LumenRetinex, Zero-DCE++, LumenFuse,
EdgeGuard, HueLock and the final alpha blend — and reports per-mode latency for
classical, deep-learning and hybrid processing.

CUDA only:

* ``--device`` must be ``cuda`` or ``cuda:N``; ``cpu``, ``mps`` and ``auto``
  are refused instead of silently measuring the host;
* no CPU comparison is produced;
* the NumPy/OpenCV stage entry points are trapped, so a stage that executes its
  CPU implementation fails the run rather than being reported as GPU;
* CUDA is synchronised before and after every timed run.

No enhancement parameter, weight or output is modified, and no quality metric
(LPIPS, NIQE, PSNR, ...) is computed. This benchmark measures latency only.

Example (Kaggle T4):

    python -m training.benchmarks.benchmark_full_pipeline_gpu \\
        --device cuda --precision float32 \\
        --images datasets/raw/LOL-v2-Real/Test/Low \\
        --count 10 --warmups 3 --runs 1 \\
        --output-dir /kaggle/working/full_pipeline_gpu

The script writes ``full_pipeline_gpu_fp32.json`` and
``full_pipeline_gpu_fp32.md`` (``fp16`` for half precision) and refuses to write
a report when CUDA is unavailable.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import deep_learning  # noqa: E402,F401  (registers zero_dce_plus_plus)
from backend.config import InferenceSettings  # noqa: E402
from backend.enhancement.pipeline import (  # noqa: E402
    REQUIRED_STAGES_BY_MODE,
    EnhancementPipeline,
)
from backend.inference.manager import ModelManager  # noqa: E402
from backend.inference.registry import available_models  # noqa: E402
from backend.runtime.config import PipelineRuntime, resolve_runtime  # noqa: E402

MODEL_ID = "zero_dce_plus_plus"
DEFAULT_WEIGHTS = PROJECT_ROOT / "backend" / "weights" / "zero_dce_plus_plus.pth"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "datasets" / "raw" / "LOL-v2-Real" / "Test" / "Low"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "training" / "reports" / "full_pipeline_gpu"
DEFAULT_DEVICE = "cuda"
DEFAULT_PRECISION = "float32"
DEFAULT_COUNT = 10
DEFAULT_WARMUPS = 3
DEFAULT_RUNS = 1

MODES: tuple[str, ...] = ("classical", "deep_learning", "hybrid")
STAGE_ORDER: tuple[str, ...] = (
    "analysis",
    "classical_cv",
    "deep_learning",
    "fusion",
    "denoising",
    "colour_correction",
    "quality_assessment",
)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")

# Imported into ``backend.enhancement.stages`` and called only on the CPU path.
# Patching them here turns a silent CPU fallback into a hard failure.
CPU_STAGE_FUNCTIONS: tuple[str, ...] = (
    "scan",
    "retinex_enhance",
    "fuse_candidates",
    "edge_guard_denoise",
    "hue_lock_correct",
)


class GpuBenchmarkError(RuntimeError):
    """The benchmark refused to run or could not produce a GPU measurement."""


class CpuExecutionError(GpuBenchmarkError):
    """A pipeline stage executed its CPU implementation on a CUDA run."""


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkSettings:
    """Resolved benchmark inputs (config file plus CLI overrides)."""

    images: Path | None = None
    count: int | None = DEFAULT_COUNT
    modes: tuple[str, ...] = MODES
    warmups: int = DEFAULT_WARMUPS
    runs: int = DEFAULT_RUNS
    weights: Path = DEFAULT_WEIGHTS
    output_dir: Path = DEFAULT_OUTPUT_DIR
    device: str = DEFAULT_DEVICE
    precision: str = DEFAULT_PRECISION


def load_config_file(path: Path) -> dict[str, Any]:
    """Read a benchmark config, or reuse a ``training.kaggle_eval`` config.

    A benchmark config is a JSON object using the keys ``images``, ``count``,
    ``modes``, ``warmups``, ``runs``, ``weights``, ``output_dir``, ``device``
    and ``precision``. A Kaggle evaluation config is recognised by its
    ``input_root``/``datasets`` keys and contributes the settings that overlap.
    """
    if not path.is_file():
        raise GpuBenchmarkError(f"Config file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GpuBenchmarkError(f"Config file is not valid JSON: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise GpuBenchmarkError(f"Config file must contain a JSON object: {path}")

    if "datasets" not in payload and "input_root" not in payload:
        return payload

    from training.kaggle_eval.config import EvalConfig

    evaluation = EvalConfig.from_file(path)
    device = evaluation.device if evaluation.device.startswith("cuda") else None
    return {
        "count": evaluation.max_images,
        "modes": [mode for mode in evaluation.modes if mode in MODES],
        "warmups": evaluation.latency_warmups,
        "runs": max(1, evaluation.latency_runs),
        "weights": evaluation.weights,
        "device": device,
        "output_dir": None,
        "images": None,
    }


def _resolve_path(value: Any, base: Path) -> Path | None:
    if value is None:
        return None
    raw = Path(str(value)).expanduser()
    return raw if raw.is_absolute() else (base / raw)


def _parse_modes(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return MODES
    if isinstance(raw, str):
        items = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, (list, tuple)):
        items = [str(part).strip() for part in raw]
    else:
        raise GpuBenchmarkError("Modes must be a comma-separated string or a list.")
    selected = tuple(dict.fromkeys(item for item in items if item))
    if not selected:
        raise GpuBenchmarkError("At least one mode is required.")
    unknown = [item for item in selected if item not in MODES]
    if unknown:
        raise GpuBenchmarkError(
            f"Unknown mode(s) {unknown}; choose from {', '.join(MODES)}."
        )
    return tuple(mode for mode in MODES if mode in selected)


def settings_from_args(args: argparse.Namespace) -> BenchmarkSettings:
    """Merge the optional config file with explicit CLI arguments.

    Explicit CLI values always win; otherwise the config file is used, then the
    documented defaults.
    """
    config: dict[str, Any] = {}
    base = PROJECT_ROOT
    if args.config is not None:
        base = args.config.resolve().parent
        config = load_config_file(args.config)

    def pick(cli_value: Any, key: str, default: Any) -> Any:
        if cli_value is not None:
            return cli_value
        value = config.get(key)
        return default if value is None else value

    images = _resolve_path(
        pick(args.images, "images", None),
        base,
    )
    weights = _resolve_path(pick(args.weights, "weights", DEFAULT_WEIGHTS), base)
    output_dir = _resolve_path(
        pick(args.output_dir, "output_dir", DEFAULT_OUTPUT_DIR), base
    )
    precision = args.precision or str(config.get("precision") or DEFAULT_PRECISION)
    if precision not in ("float32", "float16"):
        raise GpuBenchmarkError(
            f"Unsupported precision {precision!r}; use 'float32' or 'float16'."
        )

    device = args.device or str(config.get("device") or DEFAULT_DEVICE)
    count = pick(args.count, "count", DEFAULT_COUNT)
    warmups = pick(args.warmups, "warmups", DEFAULT_WARMUPS)
    runs = pick(args.runs, "runs", DEFAULT_RUNS)
    modes = _parse_modes(pick(args.modes, "modes", None))

    if count is not None:
        count = int(count)
        if count <= 0:
            raise GpuBenchmarkError("--count must be a positive integer.")
    warmups = int(warmups)
    runs = int(runs)
    if warmups < 0:
        raise GpuBenchmarkError("--warmups cannot be negative.")
    if runs < 1:
        raise GpuBenchmarkError("--runs must be at least 1.")

    return BenchmarkSettings(
        images=images,
        count=count,
        modes=modes,
        warmups=warmups,
        runs=runs,
        weights=weights or DEFAULT_WEIGHTS,
        output_dir=output_dir or DEFAULT_OUTPUT_DIR,
        device=device,
        precision=precision,
    )


# ---------------------------------------------------------------------------
# image sources
# ---------------------------------------------------------------------------


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def _manifest_entries(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("images", "records", "entries", "pairs", "evaluation"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise GpuBenchmarkError(
        "Unsupported manifest layout: expected a list of paths or an object "
        "with an 'images', 'records', 'entries' or 'pairs' list."
    )


def _entry_paths(entry: Any) -> list[str]:
    if isinstance(entry, str):
        return [entry]
    if isinstance(entry, dict):
        for key in ("low", "input", "path", "image", "image_path", "low_path"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return [value]
    return []


def collect_images(source: Path | None, limit: int | None = None) -> list[Path]:
    """Return the images to benchmark, in a deterministic order."""
    resolved = source
    if resolved is None:
        if DEFAULT_IMAGE_DIR.is_dir():
            resolved = DEFAULT_IMAGE_DIR
        else:
            raise GpuBenchmarkError(
                "No --images source supplied and the default LOL-v2 test "
                f"directory does not exist ({DEFAULT_IMAGE_DIR})."
            )
    resolved = resolved.expanduser()
    if not resolved.exists():
        raise GpuBenchmarkError(f"Image source not found: {resolved}")

    manifest_entries = 0
    if resolved.is_dir():
        paths = sorted(
            path for path in resolved.rglob("*") if path.is_file() and _is_image(path)
        )
    elif resolved.suffix.lower() == ".json":
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise GpuBenchmarkError(
                f"Manifest is not valid JSON: {resolved} ({exc})"
            ) from exc
        base = resolved.parent
        if isinstance(payload, dict) and payload.get("input_root"):
            base = Path(str(payload["input_root"]))
        entries = _manifest_entries(payload)
        manifest_entries = len(entries)
        paths = []
        for entry in entries:
            for raw in _entry_paths(entry):
                candidate = Path(raw)
                if not candidate.is_absolute():
                    candidate = base / candidate
                if candidate.is_file() and _is_image(candidate):
                    paths.append(candidate)
        paths = list(dict.fromkeys(paths))
    elif _is_image(resolved):
        paths = [resolved]
    else:
        raise GpuBenchmarkError(
            f"Unsupported image source {resolved}; pass a directory, an image "
            "file or a JSON manifest."
        )

    if not paths:
        detail = (
            f" ({manifest_entries} manifest entries, none matched an existing image)"
            if manifest_entries
            else ""
        )
        raise GpuBenchmarkError(
            f"No benchmark images found in {resolved}{detail}. Supported "
            f"extensions: {', '.join(IMAGE_SUFFIXES)}."
        )
    if limit is not None:
        if limit <= 0:
            raise GpuBenchmarkError("Image count must be positive.")
        paths = paths[:limit]
    return paths


# ---------------------------------------------------------------------------
# CUDA checks
# ---------------------------------------------------------------------------


def _cuda_index(runtime: PipelineRuntime) -> int:
    if ":" in runtime.device:
        return int(runtime.device.split(":", 1)[1])
    return 0


def require_cuda(device: str, precision: str) -> PipelineRuntime:
    """Resolve ``device`` and refuse to continue unless it is real CUDA."""
    requested = (device or "").strip().lower()
    if requested == "gpu":
        requested = "cuda"
    if requested == "cpu":
        raise GpuBenchmarkError(
            "This benchmark is GPU-only: --device cpu is not supported and no "
            "CPU comparison is produced."
        )
    if requested == "mps":
        raise GpuBenchmarkError(
            "Apple MPS is not a supported benchmark target; this benchmark "
            "requires NVIDIA CUDA (--device cuda or cuda:N)."
        )
    if requested in ("", "auto"):
        raise GpuBenchmarkError(
            "Pass --device cuda explicitly. 'auto' can resolve to the CPU "
            "reference pipeline, which this benchmark must never measure."
        )
    if requested != "cuda" and not requested.startswith("cuda:"):
        raise GpuBenchmarkError(
            f"Unsupported device {device!r}; use 'cuda' or 'cuda:N'."
        )

    try:
        runtime = resolve_runtime(requested, precision, allow_cpu_fallback=False)
    except Exception as exc:
        raise GpuBenchmarkError(
            f"Could not initialise {requested} ({precision}): {exc}"
        ) from exc

    if not runtime.is_cuda or runtime.backend != "torch":
        raise GpuBenchmarkError(
            f"Resolved device {runtime.device!r} is not CUDA; refusing to benchmark."
        )

    torch = runtime.torch
    if not torch.cuda.is_available():
        raise GpuBenchmarkError(
            "torch.cuda.is_available() is False; install a CUDA PyTorch build and "
            "run on an NVIDIA GPU."
        )
    index = _cuda_index(runtime)
    if index >= torch.cuda.device_count():
        raise GpuBenchmarkError(
            f"CUDA device index {index} is out of range; this machine exposes "
            f"{torch.cuda.device_count()} device(s)."
        )
    try:
        probe = torch.zeros(1, device=runtime.device, dtype=runtime.dtype)
        runtime.synchronize()
        probe.add_(1)
        runtime.synchronize()
    except Exception as exc:
        raise GpuBenchmarkError(
            f"CUDA device {runtime.device} could not execute a kernel: {exc}"
        ) from exc
    return runtime


def gpu_facts(runtime: PipelineRuntime) -> dict[str, Any]:
    """Record the hardware and runtime actually used for the measurement."""
    torch = runtime.torch
    index = _cuda_index(runtime)
    properties = torch.cuda.get_device_properties(index)
    total = int(properties.total_memory)
    facts: dict[str, Any] = {
        "device": runtime.device,
        "name": str(properties.name),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": total,
        "total_memory_mb": round(total / (1024**2), 1),
        "multi_processor_count": int(properties.multi_processor_count),
        "torch_version": str(torch.__version__),
        "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn_version": (
            None if torch.backends.cudnn.version() is None
            else int(torch.backends.cudnn.version())
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    return facts


def audit_model_device(
    manager: ModelManager, runtime: PipelineRuntime, precision: str
) -> dict[str, Any]:
    """Prove the loaded checkpoint lives on the CUDA device."""
    torch = runtime.torch
    model = manager.model
    if model is None:
        raise GpuBenchmarkError("The model manager reports no loaded model.")
    allocated = int(torch.cuda.memory_allocated(torch.device(runtime.device)))
    check: dict[str, Any] = {
        "model_id": manager.status.model_id,
        "manager_device": manager.status.device,
        "requested_precision": precision,
        "model_precision": getattr(model, "precision", None),
        "cuda_allocated_bytes_after_load": allocated,
        "module_inspected": False,
        "tensor_devices": [],
        "parameter_count": None,
        "parameter_dtypes": [],
    }

    module = None
    for attribute in ("network", "net", "module", "model", "_network", "_net", "_module"):
        candidate = getattr(model, attribute, None)
        if isinstance(candidate, torch.nn.Module):
            module = candidate
            break
    if module is not None:
        parameters = list(module.parameters())
        buffers = list(module.buffers())
        devices = sorted({str(tensor.device) for tensor in (*parameters, *buffers)})
        off_device = [device for device in devices if not device.startswith("cuda")]
        if off_device:
            raise GpuBenchmarkError(
                "The checkpoint is not fully on CUDA; tensors live on "
                f"{devices}. A CPU-resident model must not be benchmarked as GPU."
            )
        check.update(
            module_inspected=True,
            tensor_devices=devices,
            parameter_count=int(sum(item.numel() for item in parameters)),
            parameter_dtypes=sorted({str(item.dtype) for item in parameters}),
        )

    if allocated <= 0:
        raise GpuBenchmarkError(
            "No CUDA memory is allocated after loading the checkpoint; the model "
            "is not running on the GPU."
        )
    return check


def audit_pipeline_ops(pipeline: EnhancementPipeline, runtime: PipelineRuntime) -> dict[str, Any]:
    """Prove the pipeline selected the tensor operations, not the NumPy path."""
    ops = pipeline._execution_ops()
    if ops is None:
        raise GpuBenchmarkError(
            "The pipeline resolved no accelerator operations; it would run on CPU."
        )
    ops_runtime = getattr(ops, "runtime", None)
    device = getattr(ops_runtime, "device", None) or getattr(ops, "device", None)
    if device is None or not str(device).startswith("cuda"):
        raise GpuBenchmarkError(
            f"Pipeline operations target {device!r} instead of CUDA; refusing to "
            "benchmark a CPU execution path."
        )
    return {
        "ops_class": type(ops).__name__,
        "ops_device": str(device),
        "pipeline_backend": runtime.backend,
    }


class CpuFallbackTrap:
    """Fail immediately if a stage runs its NumPy/OpenCV implementation.

    In accelerator mode every stage takes the tensor branch, so the CPU entry
    points imported by ``backend.enhancement.stages`` must never be called. If
    one is, the pipeline silently fell back to the host and the measurement
    would not be a GPU number.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._patched: list[tuple[Any, str, Any]] = []

    def __enter__(self) -> CpuFallbackTrap:
        from backend.enhancement import stages as stages_module

        for name in CPU_STAGE_FUNCTIONS:
            original = getattr(stages_module, name, None)
            if original is None:  # pragma: no cover - defensive
                continue
            setattr(stages_module, name, self._guard(name, original))
            self._patched.append((stages_module, name, original))
        return self

    def _guard(self, name: str, original: Any):
        def guarded(*args: Any, **kwargs: Any):
            self.calls.append(name)
            raise CpuExecutionError(
                f"'{name}' executed its CPU implementation during a CUDA-only "
                "benchmark: the pipeline fell back to the host. No GPU number "
                "was produced."
            )

        guarded.__wrapped__ = original  # type: ignore[attr-defined]
        guarded.__name__ = getattr(original, "__name__", name)
        return guarded

    def __exit__(self, exc_type, exc, traceback) -> bool:
        for module, name, original in self._patched:
            setattr(module, name, original)
        self._patched.clear()
        return False


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as handle:
        return handle.convert("RGB")


def _peak_bytes(torch: Any, device: str) -> tuple[int, int]:
    target = torch.device(device)
    return (
        int(torch.cuda.max_memory_allocated(target)),
        int(torch.cuda.max_memory_reserved(target)),
    )


def _timed_run(
    pipeline: EnhancementPipeline,
    path: Path,
    mode: str,
    runtime: PipelineRuntime,
) -> dict[str, Any]:
    torch = runtime.torch
    image = _load_image(path)
    try:
        runtime.synchronize()
        started = time.perf_counter()
        result = pipeline.run(image, mode=mode)
        runtime.synchronize()
        wall_ms = (time.perf_counter() - started) * 1000.0
        record: dict[str, Any] = {
            "image": path.name,
            "path": str(path),
            "width": int(image.width),
            "height": int(image.height),
            "mode": mode,
            "status": str(result.status.value),
            "wall_ms": round(wall_ms, 3),
            "pipeline_total_ms": round(float(result.total_ms), 3),
            "stage_ms": {
                name: round(float(value), 3)
                for name, value in result.timings_ms.items()
            },
            "cuda_peak_allocated_bytes": _peak_bytes(torch, runtime.device)[0],
        }
        if record["status"] != "completed":
            result.release()
            raise GpuBenchmarkError(
                f"{mode} run on {path.name} returned status "
                f"'{record['status']}': {result.message}"
            )
        required = {stage.value for stage in REQUIRED_STAGES_BY_MODE[mode]}
        record["skipped_stages"] = sorted(required - set(record["stage_ms"]))
        result.release()
        return record
    finally:
        image.close()


def _warmup(
    pipeline: EnhancementPipeline,
    image_path: Path,
    mode: str,
    runtime: PipelineRuntime,
    passes: int,
) -> None:
    for _ in range(passes):
        image = _load_image(image_path)
        try:
            runtime.synchronize()
            result = pipeline.run(image, mode=mode)
            runtime.synchronize()
            if result.status.value != "completed":
                result.release()
                raise GpuBenchmarkError(
                    f"Warm-up run for {mode} failed: {result.message}"
                )
            result.release()
        finally:
            image.close()


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _stage_totals(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    names = [name for name in STAGE_ORDER if any(name in item["stage_ms"] for item in records)]
    extra = sorted(
        {
            name
            for item in records
            for name in item["stage_ms"]
            if name not in STAGE_ORDER
        }
    )
    totals: dict[str, dict[str, float]] = {}
    for name in (*names, *extra):
        samples = [item["stage_ms"][name] for item in records if name in item["stage_ms"]]
        if not samples:
            continue
        totals[name] = {
            "images": len(samples),
            "mean_ms": round(statistics.mean(samples), 3),
            "median_ms": round(statistics.median(samples), 3),
        }
    return totals


def benchmark_mode(
    pipeline: EnhancementPipeline,
    mode: str,
    images: list[Path],
    runtime: PipelineRuntime,
    *,
    warmups: int,
    runs: int,
) -> dict[str, Any]:
    """Measure one mode: warm-up on the first image, then timed passes."""
    torch = runtime.torch
    _warmup(pipeline, images[0], mode, runtime, warmups)
    runtime.reset_peak_memory()
    if int(torch.cuda.memory_allocated(torch.device(runtime.device))) <= 0:
        raise GpuBenchmarkError(
            f"No CUDA memory is in use during the {mode} benchmark; the run is "
            "not executing on the GPU."
        )

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for _ in range(runs):
        for path in images:
            records.append(_timed_run(pipeline, path, mode, runtime))
    loop_seconds = time.perf_counter() - started

    latencies = [float(item["wall_ms"]) for item in records]
    peak_allocated, peak_reserved = _peak_bytes(torch, runtime.device)
    if peak_allocated <= 0:
        raise GpuBenchmarkError(
            f"The {mode} benchmark allocated no CUDA memory; no GPU work ran."
        )
    skipped = sum(1 for item in records if item["skipped_stages"])

    return {
        "mode": mode,
        "required_stages": [stage.value for stage in REQUIRED_STAGES_BY_MODE[mode]],
        "images": len(images),
        "timed_runs": len(records),
        "warmup_passes": warmups,
        "latency": {
            "mean_ms": round(statistics.mean(latencies), 3),
            "median_ms": round(statistics.median(latencies), 3),
            "p90_ms": round(_percentile(latencies, 90.0), 3),
            "min_ms": round(min(latencies), 3),
            "max_ms": round(max(latencies), 3),
            "total_processing_ms": round(sum(latencies), 3),
            "loop_wall_ms": round(loop_seconds * 1000.0, 3),
            "throughput_images_per_second": round(len(records) / loop_seconds, 3)
            if loop_seconds > 0
            else None,
        },
        "per_stage_ms": _stage_totals(records),
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "images_with_skipped_stages": skipped,
        "runs": records,
    }


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip() or None
    except Exception:  # pragma: no cover - git may be absent
        return None


def image_facts(paths: list[Path]) -> list[dict[str, Any]]:
    facts = []
    for path in paths:
        with Image.open(path) as handle:
            facts.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "width": int(handle.width),
                    "height": int(handle.height),
                }
            )
    return facts


def run_benchmark(settings: BenchmarkSettings) -> dict[str, Any]:
    """Measure every requested mode on CUDA and return the raw report."""
    # Importing ``backend.deep_learning`` at module level registers the model;
    # check it so a missing registration fails here rather than inside the
    # model manager after the CUDA audit.
    registered = available_models()
    if MODEL_ID not in registered:
        raise GpuBenchmarkError(
            f"The model registry has no '{MODEL_ID}' implementation "
            f"(registered: {', '.join(registered) or 'none'}). "
            "backend.deep_learning must be imported before the checkpoint is loaded."
        )

    images = collect_images(settings.images, settings.count)
    weights = (settings.weights or DEFAULT_WEIGHTS).expanduser()
    if not weights.is_file():
        raise GpuBenchmarkError(f"Checkpoint not found: {weights}")

    runtime = require_cuda(settings.device, settings.precision)
    facts = gpu_facts(runtime)
    checks: dict[str, Any] = {
        "cuda_required": True,
        "cpu_fallback_allowed": False,
        "resolved_device": runtime.device,
        "precision": runtime.precision,
        "cpu_stage_calls": 0,
    }

    trap = CpuFallbackTrap()
    manager = ModelManager(
        InferenceSettings(
            model_id=MODEL_ID,
            weights_path=weights,
            device=runtime.device,
            precision=runtime.precision,
        )
    )
    modes: dict[str, Any] = {}
    with trap:
        status = manager.load()
        if not status.ready:
            raise GpuBenchmarkError(f"The model could not be loaded: {status.message}")
        checks.update(audit_model_device(manager, runtime, settings.precision))
        pipeline = EnhancementPipeline(model_manager=manager, runtime=runtime)
        checks.update(audit_pipeline_ops(pipeline, runtime))
        try:
            for mode in settings.modes:
                modes[mode] = benchmark_mode(
                    pipeline,
                    mode,
                    images,
                    runtime,
                    warmups=settings.warmups,
                    runs=settings.runs,
                )
        except CpuExecutionError:
            raise
        except Exception as exc:
            if trap.calls:
                raise CpuExecutionError(
                    "A stage fell back to its CPU implementation "
                    f"({', '.join(sorted(set(trap.calls)))}); the run was aborted."
                ) from exc
            raise
        finally:
            manager.close()

    if trap.calls:  # pragma: no cover - the guard raises first
        raise CpuExecutionError(
            f"CPU stage implementations were called: {sorted(set(trap.calls))}"
        )
    checks["cpu_stage_calls"] = len(trap.calls)
    checks["cpu_trap_armed"] = True
    checks["cuda_kernels_executed"] = True

    return {
        "benchmark": "lumenoir.full_pipeline.gpu",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "gpu": facts,
        "settings": {
            "device": runtime.device,
            "precision": runtime.precision,
            "weights": str(weights),
            "image_source": str(settings.images or DEFAULT_IMAGE_DIR),
            "count": len(images),
            "warmups": settings.warmups,
            "runs": settings.runs,
            "modes": list(settings.modes),
        },
        "checks": checks,
        "images": image_facts(images),
        "modes": modes,
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def _megabytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / (1024**2):.1f}"


def _milliseconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def render_markdown(report: dict[str, Any]) -> str:
    """Render the JSON report as a Markdown document."""
    gpu = report["gpu"]
    settings = report["settings"]
    lines: list[str] = [
        "# Lumenoir full-pipeline GPU benchmark",
        "",
        "Complete enhancement pipeline measured on NVIDIA CUDA through the "
        "existing `EnhancementPipeline`: LumenScan, LumenRetinex, Zero-DCE++, "
        "LumenFuse, EdgeGuard, HueLock and the final alpha blend.",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Commit: `{report.get('git_commit') or 'unknown'}`",
        f"- Device: `{settings['device']}` ({gpu['name']})",
        f"- Precision: `{settings['precision']}`",
        f"- Images: {settings['count']} from `{settings['image_source']}`",
        f"- Warm-up passes: {settings['warmups']} (first image), "
        f"timed passes: {settings['runs']}",
        "- CUDA synchronised before and after every timed run",
        "- CPU comparison: not run (GPU-only benchmark)",
        "",
        "## GPU and runtime",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| GPU | {gpu['name']} |",
        f"| Compute capability | {gpu['compute_capability']} |",
        f"| Memory | {gpu['total_memory_mb']:.1f} MB |",
        f"| SMs | {gpu['multi_processor_count']} |",
        f"| PyTorch | {gpu['torch_version']} |",
        f"| CUDA | {gpu['cuda_version']} |",
        f"| cuDNN | {gpu['cudnn_version']} |",
        f"| Python | {gpu['python']} |",
        f"| Platform | {gpu['platform']} |",
        "",
        "## Latency by mode",
        "",
        "| Mode | Timed runs | Mean ms | Median ms | P90 ms | Min ms | Max ms | "
        "Total ms | Throughput img/s | Peak CUDA MB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, result in report["modes"].items():
        latency = result["latency"]
        lines.append(
            f"| {mode} | {result['timed_runs']} | {_milliseconds(latency['mean_ms'])} | "
            f"{_milliseconds(latency['median_ms'])} | {_milliseconds(latency['p90_ms'])} | "
            f"{_milliseconds(latency['min_ms'])} | {_milliseconds(latency['max_ms'])} | "
            f"{_milliseconds(latency['total_processing_ms'])} | "
            f"{latency['throughput_images_per_second']:.2f} | "
            f"{_megabytes(result['peak_cuda_allocated_bytes'])} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Per-stage latency",
            "",
            "Mean and median of the pipeline's own per-stage measurements "
            "(milliseconds). Stages a mode does not require are omitted, and the "
            "bypass gate can legitimately skip enhancement stages on already "
            "well-exposed images.",
            "",
        ]
    )
    for mode, result in report["modes"].items():
        stages = result["per_stage_ms"]
        if not stages:
            continue
        lines.extend(
            [
                f"### {mode}",
                "",
                "| Stage | Images | Mean ms | Median ms |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for name, values in stages.items():
            lines.append(
                f"| {name} | {values['images']} | {_milliseconds(values['mean_ms'])} | "
                f"{_milliseconds(values['median_ms'])} |"
            )
        lines.append("")

    checks = report["checks"]
    lines.extend(
        [
            "## Execution checks",
            "",
            "| Check | Result |",
            "| --- | --- |",
            f"| CUDA required | {'yes' if checks['cuda_required'] else 'no'} |",
            f"| CPU fallback allowed | {'yes' if checks['cpu_fallback_allowed'] else 'no'} |",
            f"| Resolved device | `{checks['resolved_device']}` |",
            f"| Pipeline operations | `{checks['ops_class']}` on `{checks['ops_device']}` |",
            f"| Model tensors on CUDA | {checks['tensor_devices']} |",
            f"| Parameters | {checks['parameter_count']} |",
            f"| Parameter dtypes | {checks['parameter_dtypes']} |",
            f"| CPU stage implementation calls | {checks['cpu_stage_calls']} |",
            "",
            "## Scope and limitations",
            "",
            "- CPU numbers are not produced by this benchmark, and no CPU/GPU "
            "speedup is implied.",
            "- Quality metrics (PSNR, SSIM, LPIPS, NIQE, CIEDE2000) are not "
            "computed; this measures latency and memory only.",
            "- Timings include the host-to-device copy of the input, the whole "
            "stage chain and the device-to-host copy of the final image.",
            "- Results apply to this GPU, driver and PyTorch/CUDA build only. "
            "Do not quote them as another device's numbers.",
            "",
        ]
    )

    lines.extend(
        [
            "## Per-image wall time",
            "",
            "| # | Image | Size | Mode | Wall ms | Status |",
            "| ---: | --- | --- | --- | ---: | --- |",
        ]
    )
    for mode, result in report["modes"].items():
        for index, run in enumerate(result["runs"], start=1):
            lines.append(
                f"| {index} | {run['image']} | {run['width']}x{run['height']} | "
                f"{mode} | {run['wall_ms']:.1f} | {run['status']} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    report: dict[str, Any], output_dir: Path, precision: str
) -> tuple[Path, Path]:
    """Write the JSON results and the Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "fp16" if precision == "float16" else "fp32"
    json_path = output_dir / f"full_pipeline_gpu_{suffix}.json"
    markdown_path = output_dir / f"full_pipeline_gpu_{suffix}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional JSON config. A benchmark config uses images/count/modes/"
            "warmups/runs/weights/output_dir/device/precision; a "
            "training.kaggle_eval config is also accepted."
        ),
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=None,
        help="Image directory, single image or JSON manifest of images.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=f"CUDA device: 'cuda' or 'cuda:N' (default {DEFAULT_DEVICE}).",
    )
    parser.add_argument(
        "--precision",
        default=None,
        choices=("float32", "float16"),
        help=f"Benchmark precision (default {DEFAULT_PRECISION}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Where to write the reports (default {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help=f"Maximum number of images (default {DEFAULT_COUNT}).",
    )
    parser.add_argument(
        "--modes",
        default=None,
        help="Comma-separated subset of classical,deep_learning,hybrid.",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=None,
        help=f"Warm-up passes on the first image (default {DEFAULT_WARMUPS}).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help=f"Timed passes over the image set (default {DEFAULT_RUNS}).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help=f"Zero-DCE++ checkpoint (default {DEFAULT_WEIGHTS}).",
    )
    return parser


def _print_summary(report: dict[str, Any]) -> None:
    print(
        f"device {report['settings']['device']} ({report['gpu']['name']}), "
        f"precision {report['settings']['precision']}, "
        f"{report['settings']['count']} images"
    )
    print(
        f"{'mode':>14} {'runs':>6} {'mean':>9} {'median':>9} {'p90':>9} "
        f"{'img/s':>8} {'peak MB':>9}"
    )
    for mode, result in report["modes"].items():
        latency = result["latency"]
        print(
            f"{mode:>14} {result['timed_runs']:>6} "
            f"{latency['mean_ms']:>9.1f} {latency['median_ms']:>9.1f} "
            f"{latency['p90_ms']:>9.1f} "
            f"{latency['throughput_images_per_second']:>8.2f} "
            f"{_megabytes(result['peak_cuda_allocated_bytes']):>9}"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = settings_from_args(args)
        report = run_benchmark(settings)
        json_path, markdown_path = write_outputs(
            report, settings.output_dir, settings.precision
        )
    except CpuExecutionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except GpuBenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted; no report written", file=sys.stderr)
        return 130

    _print_summary(report)
    print(f"\nJSON: {json_path}\nMarkdown: {markdown_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
