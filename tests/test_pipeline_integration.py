"""End-to-end API integration with the real Zero-DCE++ checkpoint.

Excluded from the default suite by ``pytest.ini``. Run explicitly:

    python -m pytest -m integration -o addopts="" tests/test_pipeline_integration.py
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.config import InferenceSettings, Settings
from backend.deep_learning.zero_dce_plus_plus import MODEL_ID
from backend.main import create_app
from tests.conftest import make_image_bytes

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

WEIGHTS = Path(__file__).resolve().parents[1] / "backend" / "weights" / f"{MODEL_ID}.pth"


def _decode(response) -> Image.Image:
    return Image.open(
        io.BytesIO(base64.b64decode(response.json()["output"]["data"]))
    )


def test_hybrid_api_runs_with_the_pretrained_model() -> None:
    pytest.importorskip("torch")
    if not WEIGHTS.is_file():
        pytest.skip(f"Pretrained checkpoint not present at {WEIGHTS}.")

    settings = Settings(
        inference=InferenceSettings(
            model_id=MODEL_ID, weights_dir=WEIGHTS.parent, device="cpu"
        )
    )
    app = create_app(settings)
    data = make_image_bytes("PNG", (128, 96), colour=(10, 12, 14))

    with TestClient(app) as client:
        status = client.get("/api/enhance/status")
        assert status.status_code == 200
        assert status.json()["model"]["ready"] is True
        assert status.json()["modes"][2]["available"] is True

        response = client.post(
            "/api/enhance",
            files={"file": ("night.png", data, "image/png")},
            data={"mode": "hybrid"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["processing_status"] == "completed"
    assert body["pipeline"]["processing_time_ms"] > 0.0

    stage_names = {stage["name"] for stage in body["pipeline"]["stages"]}
    assert stage_names == {
        "analysis",
        "classical_cv",
        "deep_learning",
        "fusion",
        "denoising",
        "colour_correction",
        "quality_assessment",
    }
    assert all(
        stage["status"] == "implemented" for stage in body["pipeline"]["stages"]
    )
    assert body["pipeline"]["timings_ms"]["deep_learning"] > 0.0

    output = _decode(response)
    assert output.size == (128, 96)
    assert output.mode == "RGB"
    array = np.asarray(output, dtype=np.float32) / 255.0
    assert np.isfinite(array).all()
    assert array.min() >= 0.0
    assert array.max() <= 1.0
    output.close()
