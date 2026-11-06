"""Model registry.

Registered by default: ``zero_dce_plus_plus`` (imported through
``backend.deep_learning``; the model module imports PyTorch lazily). Adding
another architecture registers itself the same way:

    from backend.inference.registry import register_model

    @register_model("another_model")
    class AnotherModel(EnhancementModel):
        ...

Nothing else in the backend needs to change: the manager resolves the model by
the ``LUMENOIR_MODEL_ID`` setting and this registry supplies the factory.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from .base import EnhancementModel
from .errors import UnknownModelError

ModelT = TypeVar("ModelT", bound=EnhancementModel)
ModelFactory = Callable[[Path, str], EnhancementModel]

_FACTORIES: dict[str, ModelFactory] = {}


def register_model(
    model_id: str,
) -> Callable[[type[ModelT]], type[ModelT]]:
    """Class decorator that registers a model implementation under ``model_id``."""

    def decorator(cls: type[ModelT]) -> type[ModelT]:
        existing = _FACTORIES.get(model_id)
        if existing is not None and existing is not cls:
            raise ValueError(f"A model is already registered as '{model_id}'.")

        def factory(weights_path: Path, device: str) -> EnhancementModel:
            return cls(weights_path, device=device)

        _FACTORIES[model_id] = factory
        return cls

    return decorator


def unregister_model(model_id: str) -> None:
    """Remove a registration (used by tests and alternative deployments)."""
    _FACTORIES.pop(model_id, None)


def available_models() -> tuple[str, ...]:
    """Return the registered model ids, sorted for stable API responses."""
    return tuple(sorted(_FACTORIES))


def create_model(
    model_id: str, weights_path: Path, *, device: str = "cpu"
) -> EnhancementModel:
    """Instantiate a registered model. Does not load its weights."""
    factory = _FACTORIES.get(model_id)
    if factory is None:
        registered = ", ".join(available_models()) or "none"
        raise UnknownModelError(
            f"No model implementation is registered as '{model_id}'. "
            f"Registered models: {registered}."
        )

    return factory(weights_path, device)
