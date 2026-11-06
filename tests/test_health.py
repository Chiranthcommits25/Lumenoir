"""Health and limits endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_reports_ok(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "LUMENOIR API"
    assert body["enhancement_pipeline"] == "implemented"


def test_upload_limits_match_settings(client: TestClient) -> None:
    response = client.get("/api/images/limits")

    assert response.status_code == 200
    body = response.json()
    assert body["max_size_bytes"] == 10 * 1024 * 1024
    assert body["allowed_extensions"] == [".jpeg", ".jpg", ".png"]
    assert "image/png" in body["allowed_mime_types"]
    assert body["dimensions"]["max_megapixels"] == 40.0
