"""Inference errors.

Every failure the inference layer can raise carries a stable ``code`` and an
HTTP status so the API can return the same error envelope as upload
validation. Nothing here selects or assumes a model implementation.
"""

from __future__ import annotations


class InferenceError(Exception):
    """Base class for inference-layer failures."""

    code: str = "inference_error"
    default_status_code: int = 500

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.status_code = status_code or self.default_status_code


class UnknownModelError(InferenceError):
    """The configured model id has no registered implementation."""

    code = "unknown_model"
    default_status_code = 500


class WeightsNotFoundError(InferenceError):
    """A model was selected but its weights file is not on disk."""

    code = "weights_missing"
    default_status_code = 503


class ModelNotAvailableError(InferenceError):
    """Inference was requested while no model is ready."""

    code = "model_not_available"
    default_status_code = 503


class InferenceFailedError(InferenceError):
    """The model raised, or produced something that is not a usable image."""

    code = "inference_failed"
    default_status_code = 500


class TorchNotInstalledError(InferenceError):
    """Inference was requested but PyTorch is not installed in this build."""

    code = "torch_missing"
    default_status_code = 503

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or "PyTorch is not installed. Install the inference dependencies with: "
            "pip install -r requirements-inference.txt"
        )


class DeviceUnavailableError(InferenceError):
    """An explicitly requested inference device is not available."""

    code = "device_unavailable"
    default_status_code = 503

    def __init__(self, device: str, message: str | None = None) -> None:
        self.device = device
        super().__init__(message or f"Inference device '{device}' is not available.")


class CheckpointError(InferenceError):
    """A checkpoint is missing, unreadable, or does not match the architecture."""

    code = "checkpoint_invalid"
    default_status_code = 500

