"""Evaluation package.

Importing this package stays cheap: only the metric declarations are loaded
eagerly, so the FastAPI app keeps starting without numpy, OpenCV or PyTorch.
Heavy objects are resolved lazily through ``__getattr__``:

    from backend.evaluation import evaluate_pair   # imports the harness
"""

from __future__ import annotations

import importlib

from .metrics import (
    METRIC_SPECS,
    EvaluationReport,
    MetricName,
    MetricResult,
    MetricSpec,
    MetricStatus,
    compute_metrics,
    metric_specs,
)

_LAZY: dict[str, tuple[str, str]] = {
    "ImagePair": ("types", "ImagePair"),
    "MetricDefinition": ("types", "MetricDefinition"),
    "MetricDependencyError": ("types", "MetricDependencyError"),
    "MetricInput": ("types", "MetricInput"),
    "MetricInputError": ("types", "MetricInputError"),
    "MetricValue": ("types", "MetricValue"),
    "evaluate_arrays": ("harness", "evaluate_arrays"),
    "evaluate_pair": ("harness", "evaluate_pair"),
    "evaluate_set": ("harness", "evaluate_set"),
    "evaluate_set_detailed": ("harness", "evaluate_set_detailed"),
    "get_metric": ("registry", "get_metric"),
    "register_metric": ("registry", "register_metric"),
    "all_metrics": ("registry", "all_metrics"),
    "summarise": ("aggregate", "summarise"),
    "wilcoxon": ("aggregate", "wilcoxon"),
    "LatencyBenchmark": ("latency", "LatencyBenchmark"),
    "LatencyResult": ("latency", "LatencyResult"),
}

__all__ = [
    "METRIC_SPECS",
    "EvaluationReport",
    "MetricName",
    "MetricResult",
    "MetricSpec",
    "MetricStatus",
    "compute_metrics",
    "metric_specs",
    *_LAZY,
]


def __getattr__(name: str) -> object:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute = target
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, attribute)
    globals()[name] = value
    return value
