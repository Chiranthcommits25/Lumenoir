"""Image upload and validation endpoints.

Uploaded bytes are validated in memory and then discarded: nothing is written
to disk and no image is retained between requests.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile

from ..config import Settings
from ..preprocessing.statistics import compute_statistics
from ..preprocessing.validation import validate_and_decode
from .deps import get_app_settings
from .schemas import DimensionLimits, ImageInfoResponse, UploadLimits
from .serializers import image_info

router = APIRouter(prefix="/images", tags=["images"])


@router.get(
    "/limits",
    response_model=UploadLimits,
    summary="Report accepted formats, size and dimension limits",
)
def upload_limits(settings: Settings = Depends(get_app_settings)) -> UploadLimits:
    return UploadLimits(
        max_size_bytes=settings.max_upload_bytes,
        allowed_extensions=sorted(settings.allowed_extensions),
        allowed_mime_types=sorted(settings.allowed_mime_types),
        dimensions=DimensionLimits(
            min_width=settings.min_width,
            min_height=settings.min_height,
            max_width=settings.max_width,
            max_height=settings.max_height,
            max_megapixels=settings.max_megapixels,
        ),
    )


@router.post(
    "/upload",
    response_model=ImageInfoResponse,
    summary="Validate an uploaded image and return its information",
    responses={
        400: {"description": "Missing, empty or corrupted image data."},
        413: {"description": "File exceeds the maximum upload size."},
        415: {"description": "Unsupported file type or format mismatch."},
        422: {"description": "Image dimensions outside the accepted range."},
    },
)
async def upload_image(
    file: UploadFile = File(..., description="JPG, JPEG or PNG image."),
    settings: Settings = Depends(get_app_settings),
) -> ImageInfoResponse:
    # Read at most one byte past the limit so oversized uploads are rejected
    # without buffering the whole request body into memory.
    data = await file.read(settings.max_upload_bytes + 1)
    await file.close()

    validated = validate_and_decode(
        filename=file.filename,
        content_type=file.content_type,
        data=data,
        settings=settings,
    )

    try:
        statistics = compute_statistics(validated.image)
    finally:
        validated.image.close()

    return ImageInfoResponse(image=image_info(validated.metadata, statistics))
