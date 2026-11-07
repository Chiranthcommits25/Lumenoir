"""Unified device and precision selection for the pipeline.

``LUMENOIR_DEVICE`` selects the execution backend:

* ``auto`` (default) — CUDA when available, otherwise CPU. Apple MPS is only
  used when requested explicitly so a Mac development machine does not silently
  change production numerics.
* ``cpu`` — the reference NumPy/OpenCV pipeline. PyTorch is not imported.
* ``cuda`` / ``cuda:N`` — CUDA with an explicit device index.
* ``mps`` — Apple Silicon, for local validation.

``LUMENOIR_PRECISION`` selects ``float32`` (default) or ``float16``. CPU always
uses float32; an explicit device that is unavailable raises instead of falling
back. ``auto`` falls back to CPU with a warning.

The module is intentionally importable without PyTorch. Torch is imported only
when a non-CPU device is resolved or an accelerator dtype is requested.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DEVICE_ENV = "LUMENOIR_DEVICE"
PRECISION_ENV = "LUMENOIR_PRECISION"

DEVICE_ALIASES = {
    "": "auto",
    "default": "auto",
    "gpu": "cuda",
}
PRECISION_ALIASES = {
    "": "float32",
    "fp32": "float32",
    "float": "float32",
    "fp16": "float16",
    "half": "float16",
}
SUPPORTED_DEVICES = ("auto", "cpu", "cuda", "mps")
SUPPORTED_PRECISIONS = ("float32", "float16")


def torch_available() -> bool:
    """Return whether PyTorch is installed without importing it."""
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):  # pragma: no cover - broken environment
        return False


def _normalise_device(value: str | None) -> str:
    raw = (value or "").strip().lower()
    return DEVICE_ALIASES.get(raw, raw)


def _normalise_precision(value: str | None) -> str:
    raw = (value or "").strip().lower()
    canonical = PRECISION_ALIASES.get(raw, raw)
    if canonical not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"Unsupported precision {value!r}; use 'float32' or 'float16'."
        )
    return canonical


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PyTorch is not installed. Install requirements-inference.txt to use "
            "the accelerator pipeline, or set LUMENOIR_DEVICE=cpu."
        ) from exc
    return torch


def _cuda_device(torch: Any, requested: str) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available in this PyTorch build or on "
            "this machine."
        )
    index = 0
    if requested.startswith("cuda:"):
        raw_index = requested.split(":", 1)[1]
        try:
            index = int(raw_index)
        except ValueError as exc:
            raise RuntimeError(f"Invalid CUDA device index '{raw_index}'.") from exc
    count = int(torch.cuda.device_count())
    if index < 0 or index >= count:
        raise RuntimeError(
            f"CUDA device index {index} is out of range; this machine exposes "
            f"{count} CUDA device(s)."
        )
    return f"cuda:{index}"


def _mps_available(torch: Any) -> bool:
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is None:
        return False
    try:
        return bool(mps.is_available())
    except Exception:  # pragma: no cover - platform dependent
        return False


@dataclass
class PipelineRuntime:
    """Resolved execution target for one pipeline instance."""

    device: str = "cpu"
    precision: str = "float32"
    reason: str = ""
    torch_installed: bool = field(default_factory=torch_available)
    _torch: Any | None = field(default=None, repr=False, compare=False)
    _torch_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    @property
    def backend(self) -> str:
        return "cpu" if self.device == "cpu" else "torch"

    @property
    def is_gpu(self) -> bool:
        return self.device != "cpu"

    @property
    def is_cuda(self) -> bool:
        return self.device.startswith("cuda")

    @property
    def is_mps(self) -> bool:
        return self.device == "mps"

    @property
    def torch(self):
        """Import and cache PyTorch. Raises if it is not installed."""
        if self._torch is None:
            with self._torch_lock:
                if self._torch is None:
                    self._torch = _import_torch()
        return self._torch

    @property
    def dtype(self):
        torch = self.torch
        return torch.float16 if self.precision == "float16" else torch.float32

    def inference_mode(self):
        """Return the best no-grad context for this runtime."""
        if self.backend == "cpu":
            return _NullContext()
        return self.torch.inference_mode()

    def synchronize(self) -> None:
        """Wait for queued accelerator work so wall timings are honest."""
        if self.is_cuda:
            self.torch.cuda.synchronize(self.torch.device(self.device))
        elif self.is_mps:
            self.torch.mps.synchronize()

    def empty_cache(self) -> None:
        if self.is_cuda:
            self.torch.cuda.empty_cache()
        elif self.is_mps:
            self.torch.mps.empty_cache()

    def reset_peak_memory(self) -> None:
        if self.is_cuda:
            self.torch.cuda.reset_peak_memory_stats(
                self.torch.device(self.device)
            )

    def as_tensor(self, array, *, dtype=None):
        torch = self.torch
        tensor = torch.from_numpy(array)
        return tensor.to(
            device=self.device, dtype=dtype if dtype is not None else self.dtype
        )

    def to_numpy(self, tensor):
        return (
            tensor.detach()
            .to(device="cpu", dtype=self.torch.float32)
            .contiguous()
            .numpy()
        )

    def describe(self) -> dict[str, object]:
        return {
            "device": self.device,
            "backend": self.backend,
            "precision": self.precision,
            "torch_installed": self.torch_installed,
            "reason": self.reason,
        }


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def resolve_runtime(
    device: str | None = None,
    precision: str | None = None,
    *,
    allow_cpu_fallback: bool | None = None,
) -> PipelineRuntime:
    """Resolve the pipeline device and precision.

    Explicit arguments win over ``LUMENOIR_DEVICE`` / ``LUMENOIR_PRECISION``.
    ``auto`` falls back to CPU; explicit non-CPU devices raise when unavailable.
    """
    requested = _normalise_device(
        device if device is not None else os.getenv(DEVICE_ENV)
    )
    chosen_precision = _normalise_precision(
        precision if precision is not None else os.getenv(PRECISION_ENV)
    )
    if allow_cpu_fallback is None:
        allow_cpu_fallback = requested == "auto"

    if requested not in SUPPORTED_DEVICES and not requested.startswith("cuda:"):
        raise ValueError(
            f"Unsupported device {requested!r}; use auto, cpu, cuda, cuda:N or mps."
        )

    if requested == "cpu":
        runtime = PipelineRuntime(
            device="cpu",
            precision="float32",
            reason="CPU reference pipeline requested.",
        )
        if chosen_precision != "float32":
            logger.warning(
                "LUMENOIR_PRECISION=%s is not supported on the CPU pipeline; "
                "using float32.",
                chosen_precision,
            )
        return runtime

    if not torch_available():
        if allow_cpu_fallback:
            logger.warning(
                "LUMENOIR_DEVICE=%s requested but PyTorch is not installed; "
                "using the CPU pipeline.",
                requested,
            )
            return PipelineRuntime(
                device="cpu",
                precision="float32",
                reason="PyTorch is not installed; auto fell back to CPU.",
                torch_installed=False,
            )
        raise RuntimeError(
            f"Device '{requested}' requires PyTorch, which is not installed."
        )

    torch = _import_torch()
    resolved = requested
    try:
        if requested == "auto":
            if torch.cuda.is_available():
                resolved = _cuda_device(torch, "cuda")
                reason = "auto selected CUDA."
            else:
                resolved = "cpu"
                reason = "auto found no CUDA device; using CPU."
        elif requested.startswith("cuda"):
            resolved = _cuda_device(torch, requested)
            reason = f"explicit device '{requested}' resolved."
        elif requested == "mps":
            if not _mps_available(torch):
                raise RuntimeError(
                    "Apple Silicon MPS was requested but is not available in this "
                    "PyTorch build or on this machine."
                )
            resolved = "mps"
            reason = "explicit Apple Silicon MPS device resolved."
        else:  # pragma: no cover - guarded above
            raise ValueError(f"Unsupported device {requested!r}.")
    except Exception:
        if allow_cpu_fallback:
            logger.warning(
                "LUMENOIR device resolution failed for %s; using the CPU pipeline.",
                requested,
                exc_info=True,
            )
            return PipelineRuntime(
                device="cpu",
                precision="float32",
                reason="Accelerator initialisation failed; fell back to CPU.",
            )
        raise

    if resolved == "cpu":
        chosen_precision = "float32"
    elif resolved == "mps" and chosen_precision == "float16":
        logger.warning(
            "float16 MPS inference is supported but float32 is recommended for "
            "numerical parity; continuing with float16."
        )

    # Fail during construction, not mid-image, if the device cannot allocate.
    try:
        torch.zeros(1, device=resolved, dtype=(
            torch.float16 if chosen_precision == "float16" else torch.float32
        ))
    except Exception as exc:
        if allow_cpu_fallback:
            logger.warning(
                "Could not allocate on %s (%s); using the CPU pipeline.",
                resolved,
                exc,
            )
            return PipelineRuntime(
                device="cpu",
                precision="float32",
                reason="Accelerator allocation failed; fell back to CPU.",
            )
        raise RuntimeError(
            f"Could not initialise device '{resolved}': {exc}"
        ) from exc

    return PipelineRuntime(
        device=resolved,
        precision=chosen_precision,
        reason=reason,
        torch_installed=True,
    )


__all__ = [
    "DEVICE_ENV",
    "PRECISION_ENV",
    "PipelineRuntime",
    "resolve_runtime",
    "torch_available",
]
