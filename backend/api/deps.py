"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import Request

from ..config import Settings
from ..enhancement.pipeline import EnhancementPipeline
from ..inference.manager import ModelManager


def get_app_settings(request: Request) -> Settings:
    """Return the settings bound to the running app instance.

    Routes depend on this instead of the global ``get_settings`` so a custom
    ``create_app(Settings(...))`` (used in tests and alternative deployments)
    is honoured by every endpoint.
    """
    return request.app.state.settings


def get_model_manager(request: Request) -> ModelManager:
    """Return the process-wide model manager (loaded once at startup)."""
    return request.app.state.model_manager


def get_pipeline(request: Request) -> EnhancementPipeline:
    """Return the process-wide enhancement pipeline instance."""
    return request.app.state.pipeline
