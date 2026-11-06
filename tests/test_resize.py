"""Safe resize planning and inference size validation."""

from __future__ import annotations

import pytest
from PIL import Image

from backend.preprocessing.resize import (
    plan_resize,
    resize_image,
    restore_size,
    validate_inference_size,
)
from backend.preprocessing.validation import ImageValidationError


def test_landscape_shrinks_to_max_edge() -> None:
    plan = plan_resize(4000, 3000, max_edge=1600)

    assert (plan.target_width, plan.target_height) == (1600, 1200)
    assert plan.resized is True
    assert plan.scale == pytest.approx(0.4)


def test_portrait_and_square_preserve_ratio() -> None:
    portrait = plan_resize(1200, 3600, max_edge=1200)
    assert (portrait.target_width, portrait.target_height) == (400, 1200)

    square = plan_resize(9000, 9000, max_edge=1000)
    assert (square.target_width, square.target_height) == (1000, 1000)


def test_small_image_is_never_upscaled() -> None:
    plan = plan_resize(800, 600, max_edge=1600)

    assert (plan.target_width, plan.target_height) == (800, 600)
    assert plan.resized is False
    assert plan.scale == 1.0


def test_upscale_only_when_explicitly_allowed() -> None:
    plan = plan_resize(200, 100, max_edge=400, allow_upscale=True)

    assert (plan.target_width, plan.target_height) == (400, 200)
    assert plan.resized is True


@pytest.mark.parametrize(
    ("width", "height", "max_edge"),
    [(0, 10, 100), (10, 0, 100), (-5, 10, 100), (10, 10, 0)],
)
def test_invalid_arguments_rejected(width: int, height: int, max_edge: int) -> None:
    with pytest.raises(ValueError):
        plan_resize(width, height, max_edge=max_edge)


def test_resize_image_applies_plan_without_mutating_input() -> None:
    image = Image.new("RGB", (2000, 1000), (12, 20, 30))

    resized, plan = resize_image(image, max_edge=500)

    assert resized.size == (500, 250)
    assert plan.resized is True
    assert image.size == (2000, 1000)

    image.close()
    resized.close()


def test_restore_size_returns_original_dimensions() -> None:
    processed = Image.new("RGB", (500, 250))

    restored = restore_size(processed, (2000, 1000))
    assert restored.size == (2000, 1000)

    unchanged = restore_size(processed, (500, 250))
    assert unchanged is processed

    processed.close()
    restored.close()


def test_inference_pixel_limit_rejected() -> None:
    with pytest.raises(ImageValidationError) as excinfo:
        validate_inference_size(4000, 3000, max_pixels=4_000_000)

    assert excinfo.value.code == "image_too_large"
    assert excinfo.value.status_code == 422


def test_zero_dimensions_rejected() -> None:
    with pytest.raises(ImageValidationError) as excinfo:
        validate_inference_size(0, 300, max_pixels=4_000_000)

    assert excinfo.value.code == "invalid_dimensions"


def test_boundary_size_is_accepted() -> None:
    validate_inference_size(2000, 2000, max_pixels=4_000_000)
