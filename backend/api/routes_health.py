"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..config import Settings
from ..enhancement.pipeline import EnhancementPipeline
from ..inference.manager import ModelManager
from .deps import get_app_settings, get_model_manager, get_pipeline
from .schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Service health check")
def health(
    settings: Settings = Depends(get_app_settings),
    manager: ModelManager = Depends(get_model_manager),
    pipeline: EnhancementPipeline = Depends(get_pipeline),
) -> HealthResponse:
    status = manager.status
    ready = pipeline.capabilities().ready
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        enhancement_pipeline="implemented" if ready else "not_implemented",
        model_state=status.state.value,
        model_ready=status.ready,
    )
