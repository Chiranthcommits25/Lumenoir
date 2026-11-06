"""Latency and peak-memory benchmarking (``final_decision.md`` §6.7).

Warm-up runs come first, then timed runs with fixed thread counts. Median and
P95 are reported; peak RSS is sampled on a background thread so a short spike is
not missed.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LatencyResult:
    runs: int
    warmups: int
    median_ms: float
    mean_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    per_stage_ms: dict[str, float] = field(default_factory=dict)
    peak_rss_mb: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "runs": self.runs,
            "warmups": self.warmups,
            "median_ms": self.median_ms,
            "mean_ms": self.mean_ms,
            "p95_ms": self.p95_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "per_stage_ms": dict(self.per_stage_ms),
            "peak_rss_mb": self.peak_rss_mb,
        }


def configure_threads(threads: int) -> None:
    """Pin OpenCV and PyTorch to a fixed thread count before timing."""
    if threads <= 0:
        raise ValueError("threads must be positive.")
    try:
        import cv2

        cv2.setNumThreads(threads)
    except ImportError:
        pass
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:
        pass


class _RssSampler(threading.Thread):
    def __init__(self, interval: float) -> None:
        super().__init__(daemon=True, name="lumenoir-rss-sampler")
        self._interval = interval
        self._stop = threading.Event()
        self.peak_bytes = 0

    def run(self) -> None:  # pragma: no cover - timing dependent
        try:
            import psutil
        except ImportError:
            return

        process = psutil.Process()
        while not self._stop.wait(self._interval):
            self.peak_bytes = max(self.peak_bytes, process.memory_info().rss)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=1.0)


class LatencyBenchmark:
    """Time a callable that processes one image at a time."""

    def __init__(
        self,
        function: Callable[..., Any],
        *,
        warmups: int = 10,
        runs: int = 100,
        threads: int = 4,
        sample_memory: bool = True,
        sample_interval: float = 0.02,
    ) -> None:
        if warmups < 0:
            raise ValueError("warmups must be non-negative.")
        if runs <= 0:
            raise ValueError("runs must be positive.")
        self.function = function
        self.warmups = warmups
        self.runs = runs
        self.threads = threads
        self.sample_memory = sample_memory
        self.sample_interval = sample_interval

    def run(self, *args: Any, **kwargs: Any) -> LatencyResult:
        configure_threads(self.threads)

        sampler: _RssSampler | None = None
        if self.sample_memory:
            sampler = _RssSampler(self.sample_interval)
            sampler.start()

        try:
            for _ in range(self.warmups):
                self.function(*args, **kwargs)

            durations: list[float] = []
            per_stage: dict[str, float] = {}
            for _ in range(self.runs):
                started = time.perf_counter()
                result = self.function(*args, **kwargs)
                durations.append((time.perf_counter() - started) * 1000.0)
                if isinstance(result, Mapping):
                    for key, value in result.items():
                        if isinstance(value, (int, float)):
                            per_stage[str(key)] = per_stage.get(str(key), 0.0) + float(value)
        finally:
            if sampler is not None:
                sampler.stop()

        values = np.asarray(durations, dtype=np.float64)
        peak_rss_mb = (
            sampler.peak_bytes / (1024**2) if sampler is not None and sampler.peak_bytes else None
        )

        return LatencyResult(
            runs=self.runs,
            warmups=self.warmups,
            median_ms=float(np.median(values)),
            mean_ms=float(np.mean(values)),
            p95_ms=float(np.percentile(values, 95)),
            min_ms=float(np.min(values)),
            max_ms=float(np.max(values)),
            per_stage_ms={key: value / self.runs for key, value in per_stage.items()},
            peak_rss_mb=peak_rss_mb,
        )
