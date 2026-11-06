"""Metric registry.

Reference-based and no-reference metrics live in separate registries so a
report can be split cleanly, and so adding a metric never requires editing the
harness. Registration happens at import time in
``reference_metrics``/``no_reference_metrics``.
"""

from __future__ import annotations

from .types import MetricDefinition

_REFERENCE: dict[str, MetricDefinition] = {}
_NO_REFERENCE: dict[str, MetricDefinition] = {}


def register_metric(
    definition: MetricDefinition,
    *,
    reference: bool | None = None,
) -> MetricDefinition:
    """Register a metric. ``reference`` overrides ``requires_reference``."""
    key = definition.key
    if not key:
        raise ValueError("A metric must have a non-empty key.")
    if key in _REFERENCE or key in _NO_REFERENCE:
        raise ValueError(f"A metric is already registered as '{key}'.")

    if reference is None:
        reference = definition.requires_reference
    target = _REFERENCE if reference else _NO_REFERENCE
    target[key] = definition
    return definition


def unregister_metric(key: str) -> None:
    _REFERENCE.pop(key, None)
    _NO_REFERENCE.pop(key, None)


def get_metric(key: str) -> MetricDefinition:
    try:
        return _REFERENCE[key]
    except KeyError:
        pass
    try:
        return _NO_REFERENCE[key]
    except KeyError as exc:
        raise KeyError(f"Unknown metric '{key}'.") from exc


def reference_metrics() -> tuple[MetricDefinition, ...]:
    return tuple(sorted(_REFERENCE.values(), key=lambda item: item.key))


def no_reference_metrics() -> tuple[MetricDefinition, ...]:
    return tuple(sorted(_NO_REFERENCE.values(), key=lambda item: item.key))


def all_metrics() -> tuple[MetricDefinition, ...]:
    return reference_metrics() + no_reference_metrics()
