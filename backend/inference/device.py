"""Device selection for inference (CPU, CUDA, Apple Silicon MPS).

The requested device stays a string so the model layer and the API status keep
their existing contract. ``auto`` is the only mode that falls back; an explicit
request that is unavailable raises instead of silently running on another
device.
"""

from __future__ import annotations

from .errors import DeviceUnavailableError, TorchNotInstalledError

SUPPORTED_PREFIXES = ("cpu", "cuda", "mps", "auto")


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise TorchNotInstalledError() from exc
    return torch


def mps_available(torch_module=None) -> bool:
    torch = torch_module or _torch()
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is None:
        return False
    try:
        return bool(mps.is_available())
    except Exception:  # pragma: no cover - platform dependent
        return False


def cuda_available(torch_module=None) -> bool:
    torch = torch_module or _torch()
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - platform dependent
        return False


def resolve_device(requested: str | None) -> str:
    """Return a concrete device string: ``cpu``, ``mps``, ``cuda`` or ``cuda:N``.

    Accepted inputs: ``cpu`` (default), ``mps``, ``cuda``, ``cuda:N``, ``auto``.
    ``cpu`` is resolved without importing torch.

    Raises:
        TorchNotInstalledError: a non-CPU device was requested but torch is absent.
        DeviceUnavailableError: the explicit device is not available here.
    """
    choice = (requested or "cpu").strip().lower()
    if choice in {"", "default"}:
        choice = "cpu"

    if choice == "cpu":
        return "cpu"

    if choice not in SUPPORTED_PREFIXES and not choice.startswith("cuda:"):
        raise DeviceUnavailableError(
            choice,
            f"Unsupported inference device '{requested}'. "
            "Use cpu, mps, cuda, cuda:N or auto.",
        )

    torch = _torch()

    if choice == "auto":
        if mps_available(torch):
            return "mps"
        if cuda_available(torch):
            return "cuda:0"
        return "cpu"

    if choice == "mps":
        if not mps_available(torch):
            raise DeviceUnavailableError(
                "mps",
                "Apple Silicon MPS was requested but is not available in this "
                "PyTorch build or on this machine.",
            )
        return "mps"

    if choice == "cuda" or choice.startswith("cuda:"):
        if not cuda_available(torch):
            raise DeviceUnavailableError(
                "cuda",
                "CUDA was requested but is not available in this PyTorch build "
                "or on this machine.",
            )
        index = 0
        if choice.startswith("cuda:"):
            raw_index = choice.split(":", 1)[1]
            try:
                index = int(raw_index)
            except ValueError as exc:
                raise DeviceUnavailableError(
                    choice, f"Invalid CUDA device index '{raw_index}'."
                ) from exc
        count = int(torch.cuda.device_count())
        if index < 0 or index >= count:
            raise DeviceUnavailableError(
                choice,
                f"CUDA device index {index} is out of range; this machine exposes "
                f"{count} CUDA device(s).",
            )
        return f"cuda:{index}"

    raise DeviceUnavailableError(  # pragma: no cover - guarded above
        choice, f"Unsupported inference device '{requested}'."
    )
