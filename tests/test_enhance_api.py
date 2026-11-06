"""Enhancement API: status, successful runs, model failures and validation."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.config import InferenceSettings, Settings
from backend.enhancement.pipeline import EnhancementPipeline
from backend.main import create_app
from tests.conftest import (
    StubModelManager,
    brighten,
    make_image_bytes,
    make_noisy_image_bytes,
)

ENHANCE_URL = "/api/enhance"


def _post(client: TestClient, data: bytes, filename: str, mode: str = "hybrid"):
    return client.post(
        ENHANCE_URL,
        files={"file": (filename, data, "image/png")},
        data={"mode": mode},
    )


def _decode_output(response) -> Image.Image:
    payload = response.json()["output"]["data"]
    return Image.open(io.BytesIO(base64.b64decode(payload)))


def _app_with_stub(stub: StubModelManager):
    app = create_app()
    app.state.model_manager = stub
    app.state.pipeline = EnhancementPipeline(
        model_manager=stub, settings=app.state.settings
    )
    return app


def _ramp_png(size: int = 128) -> bytes:
    ramp = np.tile(
        np.linspace(0.25, 0.85, size, dtype=np.float32)[None, :, None], (size, 1, 3)
    )
    buffer = io.BytesIO()
    Image.fromarray(np.round(ramp * 255.0).astype(np.uint8)).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


def test_status_reports_implemented_pipeline_and_model_readiness(
    client: TestClient,
) -> None:
    response = client.get("/api/enhance/status")

    assert response.status_code == 200
    body = response.json()

    assert body["ready"] is True
    assert body["model"]["state"] == "not_selected"
    assert body["model"]["ready"] is False
    assert body["inference"]["device"] == "cpu"

    assert len(body["stages"]) == 7
    assert all(stage["status"] == "implemented" for stage in body["stages"])
    assert all(stage["description"] for stage in body["stages"])

    modes = {mode["mode"]: mode for mode in body["modes"]}
    assert modes["classical"]["available"] is True
    assert modes["deep_learning"]["available"] is False
    assert modes["hybrid"]["available"] is False

    assert len(body["artifacts"]) == 8
    assert all(artifact["available"] is False for artifact in body["artifacts"])
    assert all(artifact["reason"] for artifact in body["artifacts"])


def test_classical_enhance_returns_an_image(client: TestClient) -> None:
    data = make_image_bytes("PNG", (320, 240), colour=(12, 14, 18))

    response = _post(client, data, "night.png", mode="classical")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["mode"] == "classical"
    assert body["processing_status"] == "completed"
    assert body["processing_time_ms"] > 0.0

    assert len(body["pipeline"]["stages"]) == 5
    assert all(
        stage["status"] == "implemented" for stage in body["pipeline"]["stages"]
    )
    available = {
        artifact["kind"]
        for artifact in body["pipeline"]["artifacts"]
        if artifact["available"]
    }
    assert available == {
        "original",
        "illumination_map",
        "noise_map",
        "classical_output",
        "final_output",
    }

    assert body["image"]["width"] == 320
    assert body["image"]["statistics"]["histogram_bins"] == 16
    assert body["output"]["encoding"] == "base64"
    assert body["output"]["media_type"] == "image/png"

    output = _decode_output(response)
    assert output.size == (320, 240)
    assert output.mode == "RGB"
    array = np.asarray(output, dtype=np.float32) / 255.0
    assert np.isfinite(array).all()
    assert array.min() >= 0.0
    assert array.max() <= 1.0
    output.close()


def test_hybrid_without_model_returns_503(client: TestClient, png_bytes: bytes) -> None:
    response = _post(client, png_bytes, "night.png", mode="hybrid")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_not_available"


def test_deep_learning_without_model_returns_503(
    client: TestClient, png_bytes: bytes
) -> None:
    response = _post(client, png_bytes, "night.png", mode="deep_learning")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_not_available"


def test_hybrid_with_stub_model_returns_an_image(png_bytes: bytes) -> None:
    stub = StubModelManager(transform=lambda image: brighten(image, 2.0))
    app = _app_with_stub(stub)

    with TestClient(app) as client:
        response = _post(client, png_bytes, "night.png", mode="hybrid")

    assert response.status_code == 200
    body = response.json()
    assert body["processing_status"] == "completed"
    assert stub.calls == 1
    assert len(body["pipeline"]["stages"]) == 7
    available = {
        artifact["kind"]
        for artifact in body["pipeline"]["artifacts"]
        if artifact["available"]
    }
    assert available == {
        "original",
        "illumination_map",
        "noise_map",
        "enhancement_strength_map",
        "classical_output",
        "deep_learning_output",
        "fusion_output",
        "final_output",
    }

    output = _decode_output(response)
    assert output.size == (320, 240)
    assert output.mode == "RGB"
    output.close()


def test_bypass_gate_returns_the_original_pixels() -> None:
    data = _ramp_png()
    stub = StubModelManager(
        transform=lambda _image: (_ for _ in ()).throw(
            AssertionError("the model must not run when the bypass gate fires")
        )
    )
    app = _app_with_stub(stub)

    with TestClient(app) as client:
        response = _post(client, data, "bright.png", mode="hybrid")

    assert response.status_code == 200
    assert stub.calls == 0
    output = _decode_output(response)
    with Image.open(io.BytesIO(data)) as original:
        assert np.array_equal(np.asarray(output), np.asarray(original))
    output.close()


def test_enhance_rejects_unknown_mode(client: TestClient, png_bytes: bytes) -> None:
    response = _post(client, png_bytes, "night.png", mode="retinex_magic")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_mode"


def test_enhance_rejects_corrupted_payload(client: TestClient) -> None:
    response = _post(client, b"definitely not an image", "night.png")

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_format"


def test_enhance_validates_before_running_pipeline(
    client: TestClient, png_bytes: bytes
) -> None:
    response = client.post(
        ENHANCE_URL,
        files={"file": ("night.png", png_bytes, "text/plain")},
        data={"mode": "hybrid"},
    )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


def test_enhance_rejects_oversized_upload() -> None:
    noisy = make_noisy_image_bytes((200, 200))
    assert len(noisy) > 1024, "fixture must exceed the limit under test"

    app = create_app(Settings(max_upload_bytes=1024))
    with TestClient(app) as client:
        response = _post(client, noisy, "night.png")

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_enhance_rejects_oversized_dimensions(client: TestClient) -> None:
    data = make_image_bytes("PNG", (9000, 64))

    response = _post(client, data, "wide.png", mode="classical")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "image_too_large"


def test_enhance_rejects_image_above_the_inference_pixel_limit(
    client: TestClient,
) -> None:
    data = make_image_bytes("PNG", (2200, 2000))

    response = _post(client, data, "large.png", mode="classical")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "image_too_large"


def test_missing_weights_return_503(tmp_path: Path, png_bytes: bytes) -> None:
    settings = Settings(
        inference=InferenceSettings(model_id="not_trained_yet", weights_dir=tmp_path)
    )
    app = create_app(settings)

    with TestClient(app) as client:
        status = client.get("/api/enhance/status")
        assert status.status_code == 200
        assert status.json()["model"]["state"] == "weights_missing"
        assert status.json()["model"]["ready"] is False

        response = _post(client, png_bytes, "night.png", mode="hybrid")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "weights_missing"


def test_health_reports_model_state(client: TestClient) -> None:
    body = client.get("/api/health").json()

    assert body["enhancement_pipeline"] == "implemented"
    assert body["model_state"] == "not_selected"
    assert body["model_ready"] is False


def test_metrics_endpoint_reports_implemented_metrics(client: TestClient) -> None:
    response = client.get("/api/evaluation/metrics")

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "available"
    names = {metric["name"] for metric in body["metrics"]}
    assert {"psnr", "ssim", "lpips", "niqe", "latency", "halo", "loe", "black_crushing"} <= names
    assert len(body["metrics"]) == 12

    psnr = next(metric for metric in body["metrics"] if metric["name"] == "psnr")
    niqe = next(metric for metric in body["metrics"] if metric["name"] == "niqe")
    halo = next(metric for metric in body["metrics"] if metric["name"] == "halo")

    assert psnr["requires_reference"] is True
    assert psnr["requires_dependency"] == "scikit-image"
    assert niqe["requires_reference"] is False
    assert niqe["requires_dependency"] == "pyiqa + torch"
    assert halo["requires_reference"] is False
    assert all(metric["status"] == "implemented" for metric in body["metrics"])


def test_enhance_does_not_keep_state(client: TestClient) -> None:
    data = make_image_bytes("PNG", (96, 96), colour=(12, 14, 18))

    first = _post(client, data, "night.png", mode="classical")
    second = _post(client, data, "night.png", mode="classical")

    assert first.status_code == second.status_code == 200
    assert first.json()["output"]["data"] == second.json()["output"]["data"]
