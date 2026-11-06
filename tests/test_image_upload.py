"""Upload validation: accepted files, rejected files, corrupted input."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import create_app
from tests.conftest import make_image_bytes, make_noisy_image_bytes

UPLOAD_URL = "/api/images/upload"


def _post(client: TestClient, data: bytes, filename: str, content_type: str):
    return client.post(UPLOAD_URL, files={"file": (filename, data, content_type)})


# ---------------------------------------------------------------------------
# Accepted uploads
# ---------------------------------------------------------------------------


def test_png_upload_returns_metadata(client: TestClient, png_bytes: bytes) -> None:
    response = _post(client, png_bytes, "night.png", "image/png")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    image = body["image"]
    assert image["format"] == "PNG"
    assert (image["width"], image["height"]) == (320, 240)
    assert image["size_bytes"] == len(png_bytes)
    assert image["content_type"] == "image/png"
    assert image["colour_mode"] == "RGB"
    assert image["megapixels"] == 0.08
    assert image["aspect_ratio"] == pytest.approx(1.3333)


def test_jpeg_upload_accepts_jpeg_extension(client: TestClient, jpeg_bytes: bytes) -> None:
    response = _post(client, jpeg_bytes, "street.jpeg", "image/jpeg")

    assert response.status_code == 200
    body = response.json()["image"]
    assert body["format"] == "JPEG"
    assert (body["width"], body["height"]) == (640, 480)


# ---------------------------------------------------------------------------
# Rejected uploads
# ---------------------------------------------------------------------------


def test_missing_file_is_rejected(client: TestClient) -> None:
    assert client.post(UPLOAD_URL).status_code == 422


def test_unsupported_extension_rejected(client: TestClient) -> None:
    response = _post(client, b"GIF89a", "animation.gif", "image/gif")

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_extension"


def test_unsupported_content_type_rejected(client: TestClient, png_bytes: bytes) -> None:
    response = _post(client, png_bytes, "night.png", "text/plain")

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


def test_empty_file_rejected(client: TestClient) -> None:
    response = _post(client, b"", "empty.png", "image/png")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "empty_file"


def test_non_image_payload_rejected(client: TestClient) -> None:
    response = _post(client, b"this is definitely not an image", "fake.png", "image/png")

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_format"


def test_extension_format_mismatch_rejected(client: TestClient, png_bytes: bytes) -> None:
    response = _post(client, png_bytes, "actually_a_png.jpg", "image/jpeg")

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "format_mismatch"


def test_truncated_image_rejected(client: TestClient, png_bytes: bytes) -> None:
    response = _post(client, png_bytes[: len(png_bytes) // 2], "truncated.png", "image/png")

    assert response.status_code in {400, 415}
    assert response.json()["error"]["code"] in {"corrupted_image", "unsupported_format"}


def test_too_small_image_rejected(client: TestClient) -> None:
    tiny = make_image_bytes("PNG", (16, 16))
    response = _post(client, tiny, "tiny.png", "image/png")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "image_too_small"


def test_oversized_image_rejected(client: TestClient) -> None:
    too_wide = make_image_bytes("PNG", (9000, 64))
    response = _post(client, too_wide, "wide.png", "image/png")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "image_too_large"


def test_file_over_byte_limit_rejected() -> None:
    noisy = make_noisy_image_bytes((200, 200))
    assert len(noisy) > 1024, "fixture must exceed the limit under test"

    app = create_app(Settings(max_upload_bytes=1024))
    with TestClient(app) as client:
        # Custom app settings must reach every route, not just validation.
        assert client.get("/api/images/limits").json()["max_size_bytes"] == 1024
        response = _post(client, noisy, "night.png", "image/png")

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


# ---------------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------------


def test_error_response_shape_is_consistent(client: TestClient) -> None:
    response = _post(client, b"", "empty.png", "image/png")

    body = response.json()
    assert body["ok"] is False
    assert set(body["error"]) == {"code", "message"}
    assert body["error"]["message"]


def test_validation_endpoint_keeps_no_state(client: TestClient, png_bytes: bytes) -> None:
    """Uploading is stateless: identical requests behave identically."""
    first = _post(client, png_bytes, "night.png", "image/png")
    second = _post(client, png_bytes, "night.png", "image/png")

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
