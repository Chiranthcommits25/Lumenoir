"""Timing helpers used to measure processing stages.

Kept dependency-free: the backend must report real measured durations, never
estimates. Durations are expressed in milliseconds with sub-millisecond
resolution from ``time.perf_counter``.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Stopwatch:
    """Collect labelled durations for one request or pipeline run."""

    durations_ms: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def measure(self, label: str) -> Iterator[None]:
        """Record the wall-clock duration of the block under ``label``.

        Repeated labels accumulate, so a stage that runs twice is reported as
        the total cost rather than the last attempt only.
        """
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(label, (time.perf_counter() - started) * 1000.0)

    def add(self, label: str, duration_ms: float) -> None:
        self.durations_ms[label] = round(
            self.durations_ms.get(label, 0.0) + duration_ms, 3
        )

    @property
    def total_ms(self) -> float:
        return round(sum(self.durations_ms.values()), 3)
