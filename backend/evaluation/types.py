"""Shared evaluation types.

Arrays are float32 RGB in [0, 1]. Reference-based metrics receive
``reference``; artifact metrics that compare against the low-light input receive
``original``; no-reference metrics may use either or neither.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np


class MetricInputError(ValueError):
    """Raised when an image or a metric input is not valid."""


class MetricDependencyError(RuntimeError):
    """Raised when an optional metric library is not installed."""

    def __init__(self, dependency: str, message: str | None = None) -> None:
        self.dependency = dependency
        super().__init__(
            message
            or f"Metric dependency '{dependency}' is not installed. "
            "Install it with: pip install -r requirements-metrics.txt"
        )


@dataclass(frozen=True)
class MetricInput:
    """Everything a metric may need for one image."""

    image: np.ndarray
    reference: np.ndarray | None = None
    original: np.ndarray | None = None
    processing_time_ms: float | None = None
    timings_ms: Mapping[str, float] = field(default_factory=dict)
    peak_rss_mb: float | None = None
    extras: Mapping[str, float] = field(default_factory=dict)

    def require_reference(self) -> np.ndarray:
        if self.reference is None:
            raise MetricInputError("This metric requires a ground-truth reference image.")
        return self.reference

    def require_original(self) -> np.ndarray:
        if self.original is None:
            raise MetricInputError(
                "This metric requires the low-light input image ('original')."
            )
        return self.original


@dataclass(frozen=True)
class MetricValue:
    """One measured value, or an explicit reason it could not be computed."""

    key: str
    value: float | None
    requires_reference: bool
    unit: str | None = None
    higher_is_better: bool | None = None
    notes: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "value": self.value,
            "requires_reference": self.requires_reference,
            "unit": self.unit,
            "higher_is_better": self.higher_is_better,
            "notes": self.notes,
        }


MetricFunction = Callable[[MetricInput], float | None]


@dataclass(frozen=True)
class MetricDefinition:
    """A registered metric. The harness never hard-codes a metric name."""

    key: str
    label: str
    function: MetricFunction
    requires_reference: bool
    unit: str | None = None
    higher_is_better: bool | None = None
    description: str = ""


@dataclass(frozen=True)
class ImagePair:
    """Files for one evaluation item."""

    prediction: str
    reference: str | None = None
    original: str | None = None
    name: str | None = None
