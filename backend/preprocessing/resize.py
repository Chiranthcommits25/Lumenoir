"""Safe, aspect-preserving resizing for inference.

Model input sizing is a deployment concern, not a research decision, so it can
be implemented now: networks are trained at a fixed input scale, uploads arrive
at arbitrary sizes, and processing a 40 MP image at full resolution would blow
the CPU/memory budget.

Rules enforced here:

* aspect ratio is always preserved (no letterboxing, no cropping, no stretching);
* images are never upscaled — enlarging a small upload adds cost and no detail;
* the longest edge is capped by configuration (``LUMENOIR_INFERENCE_MAX_EDGE``);
* absurd pixel counts are rejected before allocation;
* the original size is recorded so the caller can restore resolution.

No enhancement, filtering or colour work happens in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from .validation import ImageValidationError

RESAMPLING = Image.Resampling.LANCZOS


@dataclass(frozen=True)
class ResizePlan:
    """The size change applied (or skipped) for one image."""

    source_width: int
    source_height: int
    target_width: int
    target_height: int

    @property
    def resized(self) -> bool:
        return (self.target_width, self.target_height) != (
            self.source_width,
            self.source_height,
        )

    @property
    def scale(self) -> float:
        """Target/source scale factor; always <= 1 for a shrink."""
        if self.source_width == 0:
            return 1.0
        return round(self.target_width / self.source_width, 6)

    @property
    def source_size(self) -> tuple[int, int]:
        return (self.source_width, self.source_height)

    @property
    def target_size(self) -> tuple[int, int]:
        return (self.target_width, self.target_height)


def validate_inference_size(
    width: int, height: int, *, max_pixels: int
) -> None:
    """Reject non-positive or excessively large dimensions before decoding work.

    Raises:
        ImageValidationError: ``invalid_dimensions`` or ``image_too_large``.
    """
    if width <= 0 or height <= 0:
        raise ImageValidationError(
            "invalid_dimensions",
            f"Image dimensions must be positive, got {width}x{height}.",
            422,
        )

    if max_pixels > 0 and width * height > max_pixels:
        megapixels = (width * height) / 1_000_000
        limit = max_pixels / 1_000_000
        raise ImageValidationError(
            "image_too_large",
            f"The image is {megapixels:.1f} MP, above the {limit:.1f} MP "
            "processing limit for this deployment.",
            422,
        )


def plan_resize(
    width: int,
    height: int,
    *,
    max_edge: int,
    allow_upscale: bool = False,
) -> ResizePlan:
    """Compute an aspect-preserving size whose longest edge is at most ``max_edge``.

    ``allow_upscale`` defaults to False, and exists only for callers that have
    an explicit reason (for example a model trained at a fixed input size).
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {width}x{height}.")
    if max_edge <= 0:
        raise ValueError(f"max_edge must be positive, got {max_edge}.")

    longest = max(width, height)
    if longest <= max_edge and not allow_upscale:
        return ResizePlan(width, height, width, height)

    scale = max_edge / longest
    target_width = max(1, round(width * scale))
    target_height = max(1, round(height * scale))

    return ResizePlan(width, height, target_width, target_height)


def resize_image(
    image: Image.Image,
    *,
    max_edge: int,
    allow_upscale: bool = False,
) -> tuple[Image.Image, ResizePlan]:
    """Resize ``image`` according to :func:`plan_resize`.

    Returns the (possibly unchanged) image and the applied plan. The input image
    is not modified.
    """
    plan = plan_resize(
        image.width, image.height, max_edge=max_edge, allow_upscale=allow_upscale
    )
    if not plan.resized:
        return image, plan

    return image.resize(plan.target_size, RESAMPLING), plan


def restore_size(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Resize a processed image back to the original resolution.

    Upscaling a model output restores the delivered resolution; it cannot add
    detail that was never inferred, which is why ``max_edge`` should stay close
    to the model's training scale rather than being set aggressively small.
    """
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"Target size must be positive, got {size}.")

    if (image.width, image.height) == (width, height):
        return image

    return image.resize((width, height), RESAMPLING)
