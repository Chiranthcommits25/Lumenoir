"""Model manager: load weights once, then serve inference requests.

Responsibilities

* resolve the configured weights path and report why a model is unavailable;
* load the model exactly once (guarded by a lock) and never crash the app when
  weights are missing;
* run inference serially, wrap every failure, and measure its duration.

It does **not** contain model code, preprocessing policy or image resizing:
those belong to the model implementation and the pipeline.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from PIL import Image

from ..config import InferenceSettings
from .base import EnhancementModel
from .device import resolve_device
from .errors import (
    InferenceError,
    InferenceFailedError,
    ModelNotAvailableError,
    WeightsNotFoundError,
)
from .registry import create_model

logger = logging.getLogger(__name__)


class ModelState(StrEnum):
    """Lifecycle state of the configured model."""

    NOT_SELECTED = "not_selected"
    WEIGHTS_MISSING = "weights_missing"
    NOT_LOADED = "not_loaded"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class ModelStatus:
    """Everything the API and UI need to explain model availability."""

    state: ModelState
    model_id: str
    device: str
    weights_path: str | None
    message: str
    load_duration_ms: float | None = None

    @property
    def ready(self) -> bool:
        return self.state is ModelState.READY


@dataclass(frozen=True)
class InferenceOutcome:
    """A model output plus the measured inference duration."""

    image: Image.Image
    duration_ms: float
    model_id: str


@dataclass(frozen=True)
class TensorInferenceOutcome:
    """Device tensors plus the measured tensor inference duration."""

    image: object
    curve_alpha: object | None
    duration_ms: float
    model_id: str


class ModelManager:
    """Owns the single loaded model instance for the process."""

    def __init__(self, settings: InferenceSettings) -> None:
        self.settings = settings
        self._model: EnhancementModel | None = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._status = self._initial_status()

    # -- status ------------------------------------------------------------

    def _initial_status(self) -> ModelStatus:
        settings = self.settings

        if not settings.model_id:
            return ModelStatus(
                state=ModelState.NOT_SELECTED,
                model_id="",
                device=settings.device,
                weights_path=None,
                message=(
                    "No enhancement model is selected. Set "
                    "LUMENOIR_MODEL_ID=zero_dce_plus_plus to enable "
                    "deep-learning and hybrid modes."
                ),
            )

        weights = settings.resolved_weights_path()
        return ModelStatus(
            state=ModelState.NOT_LOADED,
            model_id=settings.model_id,
            device=settings.device,
            weights_path=str(weights) if weights else None,
            message="Model configuration found; weights have not been loaded yet.",
        )

    @property
    def status(self) -> ModelStatus:
        return self._status

    @property
    def model(self) -> EnhancementModel | None:
        return self._model

    # -- loading -----------------------------------------------------------

    def load_at_startup(self) -> ModelStatus:
        """Load the configured model if allowed; never raises.

        Missing weights or an unknown model id are reported through the status
        so the rest of the API keeps working (upload and status must stay
        available even when inference cannot run).
        """
        if not self.settings.load_on_startup:
            return self._status

        return self.load()

    def load(self) -> ModelStatus:
        """Load weights once. Subsequent calls are no-ops."""
        with self._load_lock:
            if self._model is not None and self._status.ready:
                return self._status

            settings = self.settings
            if not settings.model_id:
                self._status = self._initial_status()
                return self._status

            weights = settings.resolved_weights_path()
            if weights is None:
                self._status = ModelStatus(
                    state=ModelState.NOT_SELECTED,
                    model_id=settings.model_id,
                    device=settings.device,
                    weights_path=None,
                    message="No weights path is configured for the selected model.",
                )
                return self._status

            if not weights.is_file():
                self._status = ModelStatus(
                    state=ModelState.WEIGHTS_MISSING,
                    model_id=settings.model_id,
                    device=settings.device,
                    weights_path=str(weights),
                    message=(
                        f"Model weights were not found at '{weights}'. "
                        "Enhancement requests will be rejected until the trained "
                        "checkpoint is provided."
                    ),
                )
                logger.warning("LUMENOIR model weights missing: %s", weights)
                return self._status

            started = time.perf_counter()
            resolved_device = settings.device
            try:
                resolved_device = resolve_device(settings.device)
                model = create_model(
                    settings.model_id, weights, device=resolved_device
                )
                setter = getattr(model, "set_precision", None)
                if callable(setter):
                    setter(settings.precision)
                model.load()
            except InferenceError as exc:
                self._status = self._failure_status(str(weights), exc.message)
                logger.error("LUMENOIR model load failed: %s", exc.message)
                return self._status
            except Exception as exc:
                self._status = self._failure_status(str(weights), str(exc))
                logger.exception("LUMENOIR model load crashed")
                return self._status

            self._model = model
            self._status = ModelStatus(
                state=ModelState.READY,
                model_id=settings.model_id,
                device=resolved_device,
                weights_path=str(weights),
                message="Model loaded once at startup and ready for inference.",
                load_duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            )
            logger.info(
                "LUMENOIR model '%s' loaded on %s in %.1f ms",
                settings.model_id,
                resolved_device,
                self._status.load_duration_ms or 0.0,
            )
            return self._status

    def _failure_status(self, weights_path: str, message: str) -> ModelStatus:
        return ModelStatus(
            state=ModelState.FAILED,
            model_id=self.settings.model_id,
            device=self.settings.device,
            weights_path=weights_path,
            message=f"The model could not be loaded: {message}",
        )

    # -- inference ---------------------------------------------------------

    def ensure_loaded(self) -> ModelStatus:
        """Load on demand for deployments that defer startup work."""
        status = self._status
        if status.ready:
            return status
        if status.state in {ModelState.NOT_SELECTED, ModelState.WEIGHTS_MISSING}:
            # Nothing to load: the configuration itself is incomplete.
            return self.load()
        return self.load()

    def infer(self, image: Image.Image) -> InferenceOutcome:
        """Run the model serially, wrapping failures and measuring duration.

        Raises:
            ModelNotAvailableError: no ready model.
            InferenceFailedError: the model raised or returned an unusable image.
        """
        status = self._status
        if not status.ready or self._model is None:
            raise ModelNotAvailableError(
                status.message or "No enhancement model is available.",
                code=self._unavailable_code(status),
                status_code=503,
            )

        with self._infer_lock:
            started = time.perf_counter()
            try:
                result = self._model.run(image)
            except InferenceError:
                raise
            except MemoryError as exc:  # pragma: no cover - environment dependent
                raise InferenceFailedError(
                    "Inference ran out of memory for this image."
                ) from exc
            except Exception as exc:
                raise InferenceFailedError(
                    f"Inference failed: {exc}"
                ) from exc
            duration_ms = round((time.perf_counter() - started) * 1000.0, 3)

        return InferenceOutcome(
            image=result,
            duration_ms=duration_ms,
            model_id=status.model_id,
        )

    def infer_tensor(self, tensor) -> TensorInferenceOutcome:
        """Run the model on a device tensor, serially and with timing."""
        status = self._status
        if not status.ready or self._model is None:
            raise ModelNotAvailableError(
                status.message or "No enhancement model is available.",
                code=self._unavailable_code(status),
                status_code=503,
            )
        runner = getattr(self._model, "infer_tensor", None)
        if not callable(runner):
            raise InferenceFailedError(
                "The loaded model does not implement tensor inference."
            )

        with self._infer_lock:
            started = time.perf_counter()
            try:
                result = runner(tensor)
            except InferenceError:
                raise
            except MemoryError as exc:  # pragma: no cover - environment dependent
                raise InferenceFailedError(
                    "Inference ran out of memory for this image."
                ) from exc
            except Exception as exc:
                raise InferenceFailedError(f"Inference failed: {exc}") from exc
            duration_ms = round((time.perf_counter() - started) * 1000.0, 3)

        image = getattr(result, "image", result)
        if image is None:
            raise InferenceFailedError("The model returned no tensor output.")
        return TensorInferenceOutcome(
            image=image,
            curve_alpha=getattr(result, "curve_alpha", None),
            duration_ms=duration_ms,
            model_id=status.model_id,
        )

    @staticmethod
    def _unavailable_code(status: ModelStatus) -> str:
        if status.state is ModelState.WEIGHTS_MISSING:
            return WeightsNotFoundError.code
        return ModelNotAvailableError.code

    # -- teardown ----------------------------------------------------------

    def close(self) -> None:
        """Release the model and restore the initial (unloaded) status."""
        with self._load_lock:
            if self._model is not None:
                try:
                    self._model.close()
                except Exception:
                    logger.exception("LUMENOIR model teardown failed")
                self._model = None
            self._status = self._initial_status()

    # -- helpers -----------------------------------------------------------

    def weights_path(self) -> Path | None:
        return self.settings.resolved_weights_path()
