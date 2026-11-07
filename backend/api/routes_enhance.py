"""Enhancement endpoints.

``POST /api/enhance`` validates the upload, runs the implemented pipeline and
returns the enhanced PNG. The pipeline is CPU-heavy, so it runs in the thread
pool; the Zero-DCE++ ModelManager serialises model inference internally.
Inference failures keep their typed error envelope.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from starlette.concurrency import run_in_threadpool

from ..config import Settings
from ..enhancement.pipeline import EnhancementPipeline
from ..enhancement.types import ENHANCEMENT_MODES, Artifact, PipelineResult, ProcessingStatus
from ..inference.manager import ModelManager
from ..preprocessing.resize import validate_inference_size
from ..preprocessing.statistics import compute_statistics
from ..preprocessing.validation import ImageValidationError, validate_and_decode
from .deps import get_app_settings, get_model_manager, get_pipeline
from .schemas import (
    ArtifactSchema,
    InferenceSettingsSchema,
    ModeCapabilitySchema,
    ModelStatusSchema,
    PipelineStatusResponse,
    StageCapabilitySchema,
    StageReportSchema,
)
from .serializers import image_info

router = APIRouter(prefix="/enhance", tags=["enhancement"])


def _model_status(manager: ModelManager) -> ModelStatusSchema:
    status = manager.status
    return ModelStatusSchema(
        state=status.state.value,
        model_id=status.model_id,
        device=status.device,
        weights_path=status.weights_path,
        message=status.message,
        ready=status.ready,
        load_duration_ms=status.load_duration_ms,
    )


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _artifact_schema(artifact: Artifact) -> ArtifactSchema:
    descriptor = artifact.descriptor
    image = artifact.image

    if image is None:
        return ArtifactSchema(
            kind=descriptor.kind.value,
            label=descriptor.label,
            description=descriptor.description,
            available=False,
            reason=artifact.reason,
        )

    return ArtifactSchema(
        kind=descriptor.kind.value,
        label=descriptor.label,
        description=descriptor.description,
        available=True,
        width=image.width,
        height=image.height,
        encoding="base64",
        media_type="image/png",
        data=base64.b64encode(_png_bytes(image)).decode("ascii"),
    )


def _run_payload(result: PipelineResult) -> dict:
    return {
        "mode": result.mode,
        "processing_status": result.status.value,
        "processing_time_ms": result.total_ms,
        "timings_ms": dict(result.timings_ms),
        "stages": [
            StageReportSchema(
                name=report.name.value,
                status=report.status.value,
                duration_ms=report.duration_ms,
                detail=report.detail,
            ).model_dump()
            for report in result.stages
        ],
        "artifacts": [_artifact_schema(a).model_dump() for a in result.artifacts],
    }


@router.get(
    "/status",
    response_model=PipelineStatusResponse,
    summary="Report pipeline, model and mode availability",
)
def enhancement_status(
    settings: Settings = Depends(get_app_settings),
    pipeline: EnhancementPipeline = Depends(get_pipeline),
    manager: ModelManager = Depends(get_model_manager),
) -> PipelineStatusResponse:
    capabilities = pipeline.capabilities()

    return PipelineStatusResponse(
        ready=capabilities.ready,
        stages=[
            StageCapabilitySchema(
                name=stage.name.value,
                status=stage.status.value,
                description=stage.description,
            )
            for stage in capabilities.stages
        ],
        modes=[
            ModeCapabilitySchema(
                mode=mode.mode,
                label=mode.label,
                available=mode.available,
                required_stages=[name.value for name in mode.required_stages],
            )
            for mode in capabilities.modes
        ],
        model=_model_status(manager),
        inference=InferenceSettingsSchema(
            device=settings.inference.device,
            precision=settings.inference.precision,
            max_edge=settings.inference.max_edge,
            max_inference_pixels=settings.inference.max_inference_pixels,
        ),
        artifacts=[
            _artifact_schema(artifact)
            for artifact in pipeline.artifact_capabilities()
        ],
    )


@router.post(
    "",
    summary="Validate an image and run the enhancement pipeline",
    responses={
        200: {"description": "Enhancement completed; output returned as base64 PNG."},
        400: {"description": "Unsupported mode or invalid image data."},
        413: {"description": "File exceeds the maximum upload size."},
        415: {"description": "Unsupported file type or format mismatch."},
        422: {"description": "Image dimensions outside the accepted range."},
        500: {"description": "A pipeline stage or the model failed."},
        503: {"description": "The selected model or device is not available."},
    },
)
async def enhance_image(
    file: UploadFile = File(..., description="JPG, JPEG or PNG image."),
    mode: str = Form("hybrid", description="classical, deep_learning or hybrid"),
    settings: Settings = Depends(get_app_settings),
    pipeline: EnhancementPipeline = Depends(get_pipeline),
    manager: ModelManager = Depends(get_model_manager),
):
    requested_mode = (mode or "hybrid").strip().lower()
    if requested_mode not in ENHANCEMENT_MODES:
        raise ImageValidationError(
            "unsupported_mode",
            "Unsupported enhancement mode "
            f"'{mode}'. Choose one of: {', '.join(ENHANCEMENT_MODES)}.",
            400,
        )

    data = await file.read(settings.max_upload_bytes + 1)
    await file.close()

    validated = validate_and_decode(
        filename=file.filename,
        content_type=file.content_type,
        data=data,
        settings=settings,
    )
    image = validated.image
    result: PipelineResult | None = None

    try:
        validate_inference_size(
            image.width,
            image.height,
            max_pixels=settings.inference.max_inference_pixels,
        )
        statistics = compute_statistics(image)
        result = await run_in_threadpool(
            pipeline.run, image, mode=requested_mode
        )
        payload = _run_payload(result)
        info = image_info(validated.metadata, statistics)

        if not result.completed:
            return JSONResponse(
                status_code=500,
                content={
                    "ok": False,
                    "error": {
                        "code": "enhancement_failed",
                        "message": result.message,
                    },
                    "pipeline": payload,
                    "image": info.model_dump(),
                    "model": _model_status(manager).model_dump(),
                },
            )

        stem = Path(validated.metadata.filename).stem or "image"
        filename = f"lumenoir-{stem}-{requested_mode}.png"

        return {
            "ok": True,
            "mode": requested_mode,
            "processing_status": ProcessingStatus.COMPLETED.value,
            "processing_time_ms": result.total_ms,
            "image": info.model_dump(),
            "pipeline": payload,
            "output": {
                "encoding": "base64",
                "media_type": "image/png",
                "filename": filename,
                "data": base64.b64encode(_png_bytes(result.final_image)).decode("ascii"),
            },
        }
    finally:
        if result is not None:
            result.release()
        image.close()
