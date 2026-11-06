"""Basic image statistics (measurement only)."""

from __future__ import annotations

from fastapi.testclient import TestClient
from PIL import Image

from backend.preprocessing.statistics import HISTOGRAM_BINS, compute_statistics


def test_uniform_image_statistics() -> None:
    image = Image.new("RGB", (64, 32), (40, 80, 120))

    stats = compute_statistics(image)

    assert stats.red.mean == 40
    assert stats.green.mean == 80
    assert stats.blue.mean == 120
    assert stats.luminance.min == stats.luminance.max
    assert stats.luminance.stddev == 0
    assert 70 <= stats.luminance.mean <= 75

    image.close()


def test_histogram_covers_every_pixel() -> None:
    image = Image.new("RGB", (100, 50), (0, 0, 0))

    stats = compute_statistics(image)

    assert len(stats.luminance_histogram) == HISTOGRAM_BINS
    assert stats.pixel_count == 100 * 50
    assert stats.luminance_histogram[0] == 100 * 50

    image.close()


def test_statistics_leave_the_input_untouched() -> None:
    image = Image.new("RGB", (16, 16), (200, 200, 200))
    before = image.getpixel((0, 0))

    compute_statistics(image)

    assert image.getpixel((0, 0)) == before
    image.close()


def test_upload_response_includes_statistics(client: TestClient) -> None:
    from tests.conftest import make_image_bytes

    data = make_image_bytes("PNG", (120, 90), colour=(10, 20, 30))
    response = client.post(
        "/api/images/upload", files={"file": ("dark.png", data, "image/png")}
    )

    assert response.status_code == 200
    statistics = response.json()["image"]["statistics"]
    assert statistics["histogram_bins"] == HISTOGRAM_BINS
    assert len(statistics["luminance_histogram"]) == HISTOGRAM_BINS
    assert sum(statistics["luminance_histogram"]) == 120 * 90
    assert statistics["red"]["mean"] == 10
    assert statistics["blue"]["mean"] == 30
