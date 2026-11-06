"""Aggregation and statistics for evaluation rows.

Reports mean +/- std, median, P10/P90 and the worst images, as required by
``final_decision.md`` §6.6. The Wilcoxon signed-rank test is optional: it needs
SciPy, and it is only meaningful across the same images.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Summary:
    count: int
    mean: float
    std: float
    median: float
    p10: float
    p90: float
    minimum: float
    maximum: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "p10": self.p10,
            "p90": self.p90,
            "min": self.minimum,
            "max": self.maximum,
        }


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarise(values: Sequence[float]) -> Summary | None:
    """Summarise finite metric values; ``None`` when nothing is available."""
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None

    mean = sum(finite) / len(finite)
    variance = sum((value - mean) ** 2 for value in finite) / len(finite)
    return Summary(
        count=len(finite),
        mean=mean,
        std=math.sqrt(variance),
        median=_percentile(finite, 50.0),
        p10=_percentile(finite, 10.0),
        p90=_percentile(finite, 90.0),
        minimum=min(finite),
        maximum=max(finite),
    )


def worst(
    values: Sequence[float | None],
    *,
    n: int = 5,
    higher_is_better: bool,
) -> list[tuple[int, float]]:
    """Return ``(index, value)`` for the ``n`` worst finite values."""
    if n <= 0:
        raise ValueError("n must be positive.")

    scored = [
        (index, float(value))
        for index, value in enumerate(values)
        if value is not None and math.isfinite(float(value))
    ]
    scored.sort(key=lambda item: item[1], reverse=not higher_is_better)
    return scored[:n]


def wilcoxon(left: Sequence[float], right: Sequence[float]) -> dict[str, float] | None:
    """Paired Wilcoxon signed-rank test; ``None`` if SciPy is unavailable."""
    if len(left) != len(right):
        raise ValueError("Wilcoxon needs paired samples of equal length.")
    pairs = [
        (float(a), float(b))
        for a, b in zip(left, right, strict=True)
        if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    if len(pairs) < 2:
        return None

    try:
        from scipy.stats import wilcoxon as scipy_wilcoxon
    except ImportError:
        return None

    first, second = zip(*pairs, strict=True)
    try:
        result = scipy_wilcoxon(first, second)
    except ValueError:
        return None
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def select_hard_slice(
    rows: Sequence[dict[str, object]],
    *,
    p50_key: str = "p50",
    nu_key: str = "nu",
    p50_quantile: float = 0.2,
    nu_threshold: float = 2.0,
) -> list[dict[str, object]]:
    """Rows in the darkest 20% by p50, plus rows with NU > 2.

    The features are supplied by the caller. LumenScan computes them in a later
    phase; the evaluation framework does not re-implement it here.
    """
    if not 0.0 < p50_quantile <= 1.0:
        raise ValueError("p50_quantile must be in (0, 1].")

    with_p50 = [
        (row, float(row[p50_key]))
        for row in rows
        if row.get(p50_key) is not None
    ]
    if with_p50:
        threshold = _percentile([value for _, value in with_p50], p50_quantile * 100.0)
        darkest = {id(row) for row, value in with_p50 if value <= threshold}
    else:
        darkest = set()

    selected: list[dict[str, object]] = []
    for row in rows:
        is_dark = id(row) in darkest
        non_uniform = row.get(nu_key) is not None and float(row[nu_key]) > nu_threshold
        if is_dark or non_uniform:
            selected.append(row)
    return selected
