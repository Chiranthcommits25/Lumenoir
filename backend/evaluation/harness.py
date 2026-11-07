"""Evaluation harness.

``evaluate_pair`` computes every registered metric that the supplied inputs
allow. A metric that needs a missing dependency or an absent reference is
reported with ``value=None`` and a note; it never invents a number.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from . import no_reference_metrics as _no_reference_metrics  # noqa: F401
from . import reference_metrics as _reference_metrics  # noqa: F401
from .images import load_float32_rgb, to_float32_rgb
from .registry import all_metrics, get_metric
from .types import (
    ImagePair,
    MetricDefinition,
    MetricDependencyError,
    MetricInput,
    MetricInputError,
    MetricValue,
)


def _definitions(keys: Iterable[str] | None) -> tuple[MetricDefinition, ...]:
    if keys is None:
        return all_metrics()
    return tuple(get_metric(key) for key in keys)


def evaluate_arrays(
    image: np.ndarray,
    *,
    reference: np.ndarray | None = None,
    original: np.ndarray | None = None,
    keys: Iterable[str] | None = None,
    processing_time_ms: float | None = None,
    timings_ms: Mapping[str, float] | None = None,
    peak_rss_mb: float | None = None,
    extras: Mapping[str, float] | None = None,
    assume_valid: bool = False,
) -> dict[str, MetricValue]:
    """Compute metrics for one output array.

    ``assume_valid`` skips the defensive re-conversion when every supplied array
    is already float32 HxWx3 RGB in [0, 1] (for example, straight from
    :func:`~backend.evaluation.images.to_float32_rgb`). It never changes metric
    values; callers that have not validated their arrays must leave it false.
    """
    if assume_valid:
        output = image
        reference_array = reference
        original_array = original
    else:
        output = to_float32_rgb(image, name="output")
        reference_array = (
            to_float32_rgb(reference, name="reference")
            if reference is not None
            else None
        )
        original_array = (
            to_float32_rgb(original, name="original")
            if original is not None
            else None
        )

    metric_input = MetricInput(
        image=output,
        reference=reference_array,
        original=original_array,
        processing_time_ms=processing_time_ms,
        timings_ms=timings_ms or {},
        peak_rss_mb=peak_rss_mb,
        extras=extras or {},
    )

    results: dict[str, MetricValue] = {}
    for definition in _definitions(keys):
        base = dict(
            key=definition.key,
            requires_reference=definition.requires_reference,
            unit=definition.unit,
            higher_is_better=definition.higher_is_better,
        )

        if definition.requires_reference and reference_array is None:
            results[definition.key] = MetricValue(
                value=None,
                notes="A reference image is required.",
                metric_time_ms=0.0,
                **base,
            )
            continue

        started = time.perf_counter()
        try:
            value = definition.function(metric_input)
        except MetricDependencyError as exc:
            results[definition.key] = MetricValue(
                value=None,
                notes=str(exc),
                metric_time_ms=round((time.perf_counter() - started) * 1000.0, 3),
                **base,
            )
        except MetricInputError as exc:
            results[definition.key] = MetricValue(
                value=None,
                notes=f"invalid input: {exc}",
                metric_time_ms=round((time.perf_counter() - started) * 1000.0, 3),
                **base,
            )
        except Exception as exc:  # defensive: one metric must not kill a report
            results[definition.key] = MetricValue(
                value=None,
                notes=f"metric failed: {exc}",
                metric_time_ms=round((time.perf_counter() - started) * 1000.0, 3),
                **base,
            )
        else:
            elapsed = round((time.perf_counter() - started) * 1000.0, 3)
            note = None
            if value is None:
                note = "undefined for this input"
            elif not math.isfinite(float(value)):
                note = "non-finite value"
            results[definition.key] = MetricValue(
                value=None if value is None else float(value),
                notes=note,
                metric_time_ms=elapsed,
                **base,
            )

    return results


def evaluate_pair(
    prediction: str | Path,
    *,
    reference: str | Path | None = None,
    original: str | Path | None = None,
    keys: Iterable[str] | None = None,
) -> dict[str, MetricValue]:
    """Compute metrics for one output file."""
    return evaluate_arrays(
        load_float32_rgb(prediction),
        reference=load_float32_rgb(reference) if reference is not None else None,
        original=load_float32_rgb(original) if original is not None else None,
        keys=keys,
        assume_valid=True,
    )


def evaluate_set(
    pairs: Sequence[ImagePair],
    *,
    keys: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, object]]:
    """Evaluate a list of pairs into flat rows (``name`` + one column per metric)."""
    rows: list[dict[str, object]] = []

    for index, pair in enumerate(pairs):
        if limit is not None and index >= limit:
            break
        values = evaluate_pair(
            pair.prediction,
            reference=pair.reference,
            original=pair.original,
            keys=keys,
        )
        row: dict[str, object] = {"name": pair.name or Path(pair.prediction).name}
        row.update({key: value.value for key, value in values.items()})
        rows.append(row)

    return rows


def evaluate_set_detailed(
    pairs: Sequence[ImagePair],
    *,
    keys: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, object]]:
    """Like :func:`evaluate_set` but keeps notes and requirements per metric."""
    results: list[dict[str, object]] = []

    for index, pair in enumerate(pairs):
        if limit is not None and index >= limit:
            break
        values = evaluate_pair(
            pair.prediction,
            reference=pair.reference,
            original=pair.original,
            keys=keys,
        )
        results.append(
            {
                "name": pair.name or Path(pair.prediction).name,
                "metrics": {key: value.as_dict() for key, value in values.items()},
            }
        )

    return results


def reference_metric_keys() -> tuple[str, ...]:
    return tuple(
        definition.key for definition in all_metrics() if definition.requires_reference
    )


def no_reference_metric_keys() -> tuple[str, ...]:
    return tuple(
        definition.key for definition in all_metrics() if not definition.requires_reference
    )
