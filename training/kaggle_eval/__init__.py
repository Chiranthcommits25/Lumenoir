"""Kaggle-compatible dataset discovery and evaluation runner.

The package reuses the existing backend evaluation framework and enhancement
pipeline. It only orchestrates dataset manifests, mode runs, metric records and
reports; no algorithm is reimplemented here.
"""

from __future__ import annotations

from .config import DATASET_SPECS, EvalConfig
from .discovery import DatasetManifest, EvalRecord, discover, write_manifests
from .runner import RunSummary, run_evaluation

__all__ = [
    "DATASET_SPECS",
    "DatasetManifest",
    "EvalConfig",
    "EvalRecord",
    "RunSummary",
    "discover",
    "run_evaluation",
    "write_manifests",
]
