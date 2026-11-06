"""Integration tests against the published Zero-DCE++ checkpoint.

Excluded from the default suite by ``pytest.ini``. They use the real weights at
``backend/weights/zero_dce_plus_plus.pth`` and are skipped when that file is
absent. Run explicitly:

    python -m pytest -m integration -o addopts="" tests/test_zero_dce_integration.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.config import InferenceSettings
from backend.deep_learning.zero_dce_plus_plus import (
    EXPECTED_CHECKPOINT_BYTES,
    EXPECTED_PARAMETERS,
    MODEL_ID,
    ZeroDCEPlusPlusModel,
)
from backend.inference.manager import ModelManager

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]

WEIGHTS = Path(__file__).resolve().parents[1] / "backend" / "weights" / (
    f"{MODEL_ID}.pth"
)


def _require_weights() -> Path:
    if not WEIGHTS.is_file():
        pytest.skip(f"Pretrained checkpoint not present at {WEIGHTS}.")
    return WEIGHTS


def _dark_image(width: int = 320, height: int = 240) -> Image.Image:
    rows = np.linspace(0.0, 0.25, height, dtype=np.float32)[:, None]
    cols = np.linspace(0.0, 0.15, width, dtype=np.float32)[None, :]
    luminance = np.clip(rows + cols, 0.0, 0.3)
    rgb = np.stack([luminance, luminance * 0.9, luminance * 0.8], axis=2)
    return Image.fromarray((rgb * 255).round().astype(np.uint8))


def _mean_luminance(image: Image.Image) -> float:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return float(
        (0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]).mean()
    )


def test_checkpoint_is_the_published_zero_dce_plus_plus() -> None:
    path = _require_weights()

    assert path.stat().st_size == EXPECTED_CHECKPOINT_BYTES

    model = ZeroDCEPlusPlusModel(path, device="cpu")
    model.load()

    assert model.parameter_count == EXPECTED_PARAMETERS
    assert model.info().weights_bytes == EXPECTED_CHECKPOINT_BYTES

    model.close()


def test_pretrained_model_enhances_a_dark_image_on_cpu() -> None:
    path = _require_weights()
    model = ZeroDCEPlusPlusModel(path, device="cpu")
    model.load()

    image = _dark_image()
    result = model.run(image)

    assert result.size == image.size
    assert result.mode == "RGB"
    assert _mean_luminance(result) > _mean_luminance(image)

    image.close()
    result.close()
    model.close()


def test_cpu_latency_at_1200x900_is_within_the_target() -> None:
    path = _require_weights()
    model = ZeroDCEPlusPlusModel(path, device="cpu")
    model.load()

    image = _dark_image(1200, 900)
    model.run(image).close()  # warm-up

    durations: list[float] = []
    for _ in range(3):
        started = time.perf_counter()
        output = model.run(image)
        durations.append((time.perf_counter() - started) * 1000.0)
        output.close()

    median_ms = sorted(durations)[len(durations) // 2]
    print(f"Zero-DCE++ CPU 1200x900 median: {median_ms:.1f} ms")

    assert median_ms < 10_000.0

    image.close()
    model.close()


def test_manager_serves_the_real_checkpoint() -> None:
    path = _require_weights()
    manager = ModelManager(
        InferenceSettings(model_id=MODEL_ID, weights_dir=path.parent, device="cpu")
    )

    status = manager.load_at_startup()

    assert status.ready is True
    assert status.device == "cpu"
    assert status.weights_path == str(path)

    image = _dark_image(160, 120)
    outcome = manager.infer(image)

    assert outcome.image.size == image.size

    image.close()
    outcome.image.close()
    manager.close()


def test_mps_inference_when_available() -> None:
    path = _require_weights()
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Apple Silicon MPS is not available.")

    model = ZeroDCEPlusPlusModel(path, device="mps")
    model.load()

    image = _dark_image(160, 120)
    result = model.run(image)

    assert result.size == image.size

    image.close()
    result.close()
    model.close()
