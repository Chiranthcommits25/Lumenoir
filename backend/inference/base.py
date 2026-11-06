"""Model-agnostic inference interface.

This module defines the seam a model plugs into. It contains no architecture,
no model selection and no weights: those live in ``backend/deep_learning/``.
Zero-DCE++ is the registered implementation.

A concrete model subclass only has to provide:

    model_id            unique identifier registered in the registry
    preprocess()        PIL image            -> model input
    infer()             model input          -> raw model output
    postprocess()       raw output + size    -> PIL image
    load()              load weights once (called by the manager)

Loading, serialisation, error handling, size restoration and timing stay in
the manager/pipeline so those behaviours cannot drift between models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image

from .errors import InferenceFailedError


@dataclass(frozen=True)
class ModelInfo:
    """Describes a loaded model. Every value is measured, never assumed."""

    model_id: str
    display_name: str
    device: str
    weights_path: str
    weights_bytes: int | None = None
    parameter_count: int | None = None


class EnhancementModel(ABC):
    """Base class for a low-light enhancement network.

    Subclasses are created by the registry through a factory that receives the
    resolved weights path and device, so the manager can build them without
    knowing their class.
    """

    model_id: ClassVar[str] = ""
    display_name: ClassVar[str] = ""

    def __init__(self, weights_path: Path, *, device: str = "cpu") -> None:
        self.weights_path = weights_path
        self.device = device
        self._loaded = False

    # -- lifecycle ---------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """Load weights into memory once.

        Must be idempotent: the manager calls it exactly once per process but
        tests and alternate entry points may call it again.
        """

    def close(self) -> None:
        """Release model resources. The default only clears the loaded flag."""
        self._loaded = False

    # -- model stages ------------------------------------------------------

    @abstractmethod
    def preprocess(self, image: Image.Image) -> Any:
        """Convert a validated RGB image into the model's input representation."""

    @abstractmethod
    def infer(self, prepared: Any) -> Any:
        """Run the network. Must not perform resizing or colour conversion."""

    @abstractmethod
    def postprocess(self, output: Any, target_size: tuple[int, int]) -> Image.Image:
        """Convert raw network output back into an RGB image of ``target_size``."""

    # -- template method ---------------------------------------------------

    def run(self, image: Image.Image) -> Image.Image:
        """Preprocess, infer and postprocess, validating the final image.

        The returned image is never larger than the input: the caller is
        responsible for restoring the original resolution if it resized first.
        """
        prepared = self.preprocess(image)
        raw = self.infer(prepared)
        restored = self.postprocess(raw, image.size)

        if not isinstance(restored, Image.Image):
            raise InferenceFailedError(
                "The model returned an unexpected output type; expected a PIL image."
            )

        if restored.width <= 0 or restored.height <= 0:
            raise InferenceFailedError("The model returned an empty image.")

        return restored.convert("RGB")

    # -- introspection -----------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._loaded

    def info(self) -> ModelInfo:
        try:
            weights_bytes = self.weights_path.stat().st_size
        except OSError:
            weights_bytes = None

        return ModelInfo(
            model_id=self.model_id or type(self).__name__,
            display_name=self.display_name or self.model_id,
            device=self.device,
            weights_path=str(self.weights_path),
            weights_bytes=weights_bytes,
            parameter_count=getattr(self, "parameter_count", None),
        )
