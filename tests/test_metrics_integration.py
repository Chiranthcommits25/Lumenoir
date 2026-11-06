"""Integration tests for LPIPS and NIQE.

These tests build the real models, which download pretrained weights
(LPIPS/AlexNet is about 233 MB) on first use. They are therefore excluded from
the default suite by ``pytest.ini`` and each one has a hard timeout so a slow
network can never stall the suite again.

Run them explicitly:

    .venv/bin/python -m pytest -m integration -o addopts="" tests/test_metrics_integration.py
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from backend.evaluation.harness import evaluate_arrays

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]


def _image(size: int = 64) -> np.ndarray:
    rows = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None]
    cols = np.linspace(0.0, 1.0, size, dtype=np.float32)[None, :]
    channel = ((rows + cols) / 2.0).astype(np.float32)
    return np.stack([channel, channel**1.5, 1.0 - channel], axis=2)


def test_lpips_identity_is_near_zero() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("lpips")
    image = _image()

    values = evaluate_arrays(image, reference=image.copy(), keys=["lpips"])

    assert values["lpips"].value is not None
    assert values["lpips"].value < 0.05


def test_niqe_is_finite() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("pyiqa")

    values = evaluate_arrays(_image(), keys=["niqe"])

    assert values["niqe"].value is not None
    assert math.isfinite(values["niqe"].value)
