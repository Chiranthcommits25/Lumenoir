"""Shared test fixtures and image builders."""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.inference.errors import ModelNotAvailableError
from backend.inference.manager import InferenceOutcome, ModelState, ModelStatus
from backend.main import create_app


class StubModelManager:
    """Test double: no PyTorch and no checkpoint, same manager interface."""

    def __init__(self, *, ready: bool = True, transform=None) -> None:
        self.status = ModelStatus(
            state=ModelState.READY if ready else ModelState.NOT_SELECTED,
            model_id="stub",
            device="cpu",
            weights_path=None,
            message="Stub model ready." if ready else "Stub model not selected.",
        )
        self.transform = transform
        self.calls = 0

    def load_at_startup(self) -> ModelStatus:
        return self.status

    def infer(self, image: Image.Image) -> InferenceOutcome:
        if not self.status.ready:
            raise ModelNotAvailableError("Stub model is not ready.")
        self.calls += 1
        output = self.transform(image.copy()) if self.transform else image.copy()
        return InferenceOutcome(image=output, duration_ms=1.0, model_id="stub")

    def close(self) -> None:
        pass


def brighten(image: Image.Image, factor: float = 2.0) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    encoded = np.clip(array * factor, 0.0, 1.0)
    return Image.fromarray(np.round(encoded * 255.0).astype(np.uint8))


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


def make_image_bytes(
    fmt: str = "PNG",
    size: tuple[int, int] = (320, 240),
    mode: str = "RGB",
    colour: tuple[int, ...] = (12, 14, 18),
) -> bytes:
    """Build an in-memory image of the given format and dimensions."""
    if mode == "RGB":
        colour = colour[:3]
    buffer = io.BytesIO()
    Image.new(mode, size, colour).save(buffer, format=fmt)
    return buffer.getvalue()


def make_noisy_image_bytes(size: tuple[int, int] = (200, 200), fmt: str = "PNG") -> bytes:
    """Build an incompressible image, for size-limit tests."""
    import os

    buffer = io.BytesIO()
    Image.frombytes("RGB", size, os.urandom(size[0] * size[1] * 3)).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture()
def png_bytes() -> bytes:
    return make_image_bytes("PNG", (320, 240))


@pytest.fixture()
def jpeg_bytes() -> bytes:
    return make_image_bytes("JPEG", (640, 480), colour=(20, 24, 30))
