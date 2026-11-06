"""Model-agnostic inference layer.

Registered models: ``zero_dce_plus_plus`` (imported through
``backend.deep_learning``). The registry, load-once manager and device resolver
contain no architecture code; adding another model means one registration.

The model modules import PyTorch and NumPy lazily, so importing this package
(and the API) does not require them.
"""

from __future__ import annotations

from .base import EnhancementModel, ModelInfo
from .device import resolve_device
from .errors import (
    CheckpointError,
    DeviceUnavailableError,
    InferenceError,
    InferenceFailedError,
    ModelNotAvailableError,
    TorchNotInstalledError,
    UnknownModelError,
    WeightsNotFoundError,
)
from .manager import InferenceOutcome, ModelManager, ModelState, ModelStatus
from .registry import (
    available_models,
    create_model,
    register_model,
    unregister_model,
)

__all__ = [
    "CheckpointError",
    "DeviceUnavailableError",
    "EnhancementModel",
    "InferenceError",
    "InferenceFailedError",
    "InferenceOutcome",
    "ModelInfo",
    "ModelManager",
    "ModelNotAvailableError",
    "ModelState",
    "ModelStatus",
    "TorchNotInstalledError",
    "UnknownModelError",
    "WeightsNotFoundError",
    "available_models",
    "create_model",
    "register_model",
    "resolve_device",
    "unregister_model",
]
