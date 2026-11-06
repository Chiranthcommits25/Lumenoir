"""FastAPI application entry point.

Run from the repository root:

    uvicorn backend.main:app --reload

Scope: health, upload validation, pipeline/model/status reporting and the
declared (not computed) evaluation metrics. No enhancement stage is
implemented, so ``POST /api/enhance`` reports exactly which stages are pending
and returns no image.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import deep_learning  # noqa: F401  (registers the available models)
from .api import routes_enhance, routes_evaluation, routes_health, routes_images
from .config import Settings, get_settings
from .enhancement.pipeline import EnhancementPipeline
from .inference.errors import InferenceError
from .inference.manager import ModelManager
from .preprocessing.validation import ImageValidationError

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application. Exposed for tests."""
    settings = settings or get_settings()

    # One manager and one pipeline per app instance. The manager loads weights
    # exactly once in the lifespan hook below; the pipeline is stateless.
    manager = ModelManager(settings.inference)
    pipeline = EnhancementPipeline(model_manager=manager, settings=settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        status = manager.load_at_startup()
        logger.info("LUMENOIR inference state: %s", status.state.value)
        try:
            yield
        finally:
            manager.close()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "LUMENOIR — robust low-light image restoration and enhancement. "
            "The full pipeline is implemented. Classical mode runs without a "
            "model; deep-learning and hybrid modes require a ready Zero-DCE++ "
            "checkpoint."
        ),
        lifespan=lifespan,
    )

    # Routes read settings from the app instance (see ``api/deps.py``), so an
    # app built with custom settings behaves consistently everywhere.
    app.state.settings = settings
    app.state.model_manager = manager
    app.state.pipeline = pipeline

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(ImageValidationError)
    async def handle_image_validation_error(
        _request: Request, exc: ImageValidationError
    ) -> JSONResponse:
        """Return validation failures in one consistent shape for the frontend."""
        return JSONResponse(
            status_code=exc.status_code,
            content={"ok": False, "error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(InferenceError)
    async def handle_inference_error(
        _request: Request, exc: InferenceError
    ) -> JSONResponse:
        """Inference failures use the same envelope as validation failures."""
        return JSONResponse(
            status_code=exc.status_code,
            content={"ok": False, "error": {"code": exc.code, "message": exc.message}},
        )

    app.include_router(routes_health.router, prefix=settings.api_prefix)
    app.include_router(routes_images.router, prefix=settings.api_prefix)
    app.include_router(routes_enhance.router, prefix=settings.api_prefix)
    app.include_router(routes_evaluation.router, prefix=settings.api_prefix)

    return app


app = create_app()
