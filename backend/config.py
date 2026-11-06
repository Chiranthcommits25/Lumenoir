"""Configuration for the LUMENOIR backend.

Only values needed by code that actually exists live here. Algorithm
coefficients live with their modules (``LumenScanConfig``,
``LumenRetinexConfig``, ``LumenFuseConfig``, ``EdgeGuardConfig``,
``HueLockConfig``); this module holds deployment settings.

Model settings are deployment plumbing: where weights live, which device to
use, and how large an input may be. ``model_id`` is empty by default, which
means deep-learning/hybrid modes are disabled until a model is configured.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

try:  # optional: .env support in local development
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is a declared dependency
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "backend" / "weights"

DEFAULT_CORS_ORIGINS: tuple[str, ...] = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return default if raw is None or not raw.strip() else raw.strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass(frozen=True)
class InferenceSettings:
    """Deployment settings for a (yet to be selected) enhancement model.

    ``model_id`` empty means no model is configured, which is the current
    default: the architecture is an open research decision.
    """

    model_id: str = ""
    weights_dir: Path = DEFAULT_WEIGHTS_DIR
    weights_path: Path | None = None
    device: str = "cpu"
    # Longest edge used for inference. Set conservatively so a 40 MP upload
    # does not have to be processed at full resolution on a small CPU box.
    max_edge: int = 1600
    max_inference_pixels: int = 4_000_000
    load_on_startup: bool = True

    def resolved_weights_path(self) -> Path | None:
        """Explicit weights path wins; otherwise ``<weights_dir>/<model_id>.pth``."""
        if self.weights_path is not None:
            return self.weights_path
        if not self.model_id:
            return None
        return self.weights_dir / f"{self.model_id}.pth"


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings for the API."""

    app_name: str = "LUMENOIR API"
    app_version: str = "0.2.0"
    api_prefix: str = "/api"

    # ---- upload validation (foundation) ----
    allowed_mime_types: frozenset[str] = field(
        default_factory=lambda: frozenset({"image/jpeg", "image/png"})
    )
    allowed_extensions: frozenset[str] = field(
        default_factory=lambda: frozenset({".jpg", ".jpeg", ".png"})
    )
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB
    min_width: int = 64
    min_height: int = 64
    max_width: int = 8000
    max_height: int = 8000
    max_megapixels: float = 40.0

    # ---- model / inference plumbing (no architecture selected) ----
    inference: InferenceSettings = field(default_factory=InferenceSettings)

    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings, with environment overrides applied."""
    weights_raw = _env_str("LUMENOIR_MODEL_WEIGHTS", "")
    inference = InferenceSettings(
        model_id=_env_str("LUMENOIR_MODEL_ID", ""),
        weights_dir=Path(_env_str("LUMENOIR_WEIGHTS_DIR", str(DEFAULT_WEIGHTS_DIR))),
        weights_path=Path(weights_raw) if weights_raw else None,
        device=_env_str("LUMENOIR_INFERENCE_DEVICE", "cpu"),
        max_edge=_env_int("LUMENOIR_INFERENCE_MAX_EDGE", 1600),
        max_inference_pixels=_env_int("LUMENOIR_INFERENCE_MAX_PIXELS", 4_000_000),
        load_on_startup=_env_bool("LUMENOIR_LOAD_MODEL_ON_STARTUP", True),
    )

    return Settings(
        max_upload_bytes=_env_int("LUMENOIR_MAX_UPLOAD_BYTES", 10 * 1024 * 1024),
        cors_origins=_env_list("LUMENOIR_CORS_ORIGINS", DEFAULT_CORS_ORIGINS),
        inference=inference,
    )
