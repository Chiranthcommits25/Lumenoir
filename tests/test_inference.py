"""Inference layer: registry, load-once manager, failure handling."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.config import InferenceSettings
from backend.inference.base import EnhancementModel
from backend.inference.errors import (
    InferenceFailedError,
    ModelNotAvailableError,
    UnknownModelError,
)
from backend.inference.manager import ModelManager, ModelState
from backend.inference.registry import (
    available_models,
    create_model,
    register_model,
    unregister_model,
)
from backend.main import create_app


class _FakeModel(EnhancementModel):
    """Test double: proves the loader/inference plumbing, contains no algorithm."""

    model_id = "test_fake"
    display_name = "Fake model"
    load_calls = 0
    close_calls = 0

    def load(self) -> None:
        type(self).load_calls += 1
        self._loaded = True

    def preprocess(self, image: Image.Image):
        return image

    def infer(self, prepared):
        return prepared

    def postprocess(self, output, target_size: tuple[int, int]) -> Image.Image:
        return output.convert("RGB")

    def close(self) -> None:
        type(self).close_calls += 1


class _UnusableOutputModel(_FakeModel):
    def postprocess(self, output, target_size: tuple[int, int]):
        return None  # type: ignore[return-value]


class _ExplodingLoadModel(_FakeModel):
    def load(self) -> None:
        raise RuntimeError("checkpoint is unreadable")


@pytest.fixture()
def fake_model():
    register_model("test_fake")(_FakeModel)
    _FakeModel.load_calls = 0
    _FakeModel.close_calls = 0
    try:
        yield _FakeModel
    finally:
        unregister_model("test_fake")


def _weights(tmp_path: Path, model_id: str = "test_fake") -> Path:
    path = tmp_path / f"{model_id}.pth"
    path.write_bytes(b"placeholder-checkpoint")
    return path


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_is_empty_by_default() -> None:
    assert "test_fake" not in available_models()

    with pytest.raises(UnknownModelError) as excinfo:
        create_model("definitely_not_registered", Path("weights.pth"))

    assert excinfo.value.code == "unknown_model"


# ---------------------------------------------------------------------------
# No model selected
# ---------------------------------------------------------------------------


def test_manager_without_model_is_not_selected() -> None:
    manager = ModelManager(InferenceSettings())

    assert manager.status.state is ModelState.NOT_SELECTED
    assert manager.status.ready is False
    assert "zero_dce_plus_plus" in manager.status.message

    image = Image.new("RGB", (8, 8))
    with pytest.raises(ModelNotAvailableError) as excinfo:
        manager.infer(image)
    assert excinfo.value.code == "model_not_available"
    assert excinfo.value.status_code == 503
    image.close()


# ---------------------------------------------------------------------------
# Missing weights
# ---------------------------------------------------------------------------


def test_missing_weights_are_reported_not_fatal(tmp_path: Path) -> None:
    manager = ModelManager(
        InferenceSettings(model_id="test_fake", weights_dir=tmp_path)
    )

    status = manager.load_at_startup()

    assert status.state is ModelState.WEIGHTS_MISSING
    assert status.ready is False
    assert status.weights_path is not None
    assert str(tmp_path) in status.weights_path
    assert "not found" in status.message

    image = Image.new("RGB", (8, 8))
    with pytest.raises(ModelNotAvailableError) as excinfo:
        manager.infer(image)
    assert excinfo.value.code == "weights_missing"
    image.close()


# ---------------------------------------------------------------------------
# Load once, infer, teardown
# ---------------------------------------------------------------------------


def test_model_loads_once_and_serves_inference(tmp_path: Path, fake_model) -> None:
    _weights(tmp_path)
    manager = ModelManager(
        InferenceSettings(model_id="test_fake", weights_dir=tmp_path)
    )

    status = manager.load_at_startup()
    assert status.state is ModelState.READY
    assert status.ready is True
    assert fake_model.load_calls == 1

    # Repeated calls must not reload the checkpoint.
    manager.ensure_loaded()
    manager.load()
    assert fake_model.load_calls == 1

    image = Image.new("RGB", (32, 24), (9, 9, 9))
    outcome = manager.infer(image)

    assert outcome.image.size == (32, 24)
    assert outcome.model_id == "test_fake"
    assert outcome.duration_ms >= 0

    image.close()
    outcome.image.close()

    manager.close()
    assert fake_model.close_calls == 1
    assert manager.status.state is ModelState.NOT_LOADED


def test_load_failure_is_recorded(tmp_path: Path) -> None:
    register_model("test_exploding")(_ExplodingLoadModel)
    _weights(tmp_path, "test_exploding")
    try:
        manager = ModelManager(
            InferenceSettings(model_id="test_exploding", weights_dir=tmp_path)
        )

        status = manager.load_at_startup()

        assert status.state is ModelState.FAILED
        assert "could not be loaded" in status.message
        assert manager.status.ready is False
    finally:
        unregister_model("test_exploding")


def test_unusable_model_output_raises_inference_error(tmp_path: Path) -> None:
    register_model("test_unusable")(_UnusableOutputModel)
    _weights(tmp_path, "test_unusable")
    try:
        manager = ModelManager(
            InferenceSettings(model_id="test_unusable", weights_dir=tmp_path)
        )
        assert manager.load_at_startup().ready is True

        image = Image.new("RGB", (16, 16))
        with pytest.raises(InferenceFailedError):
            manager.infer(image)
        image.close()
    finally:
        unregister_model("test_unusable")


def test_inference_errors_use_the_shared_envelope() -> None:
    app = create_app()

    @app.get("/api/_test/boom")
    def boom() -> None:
        raise InferenceFailedError("Model exploded.")

    client = TestClient(app)
    response = client.get("/api/_test/boom")

    assert response.status_code == 500
    assert response.json() == {
        "ok": False,
        "error": {"code": "inference_failed", "message": "Model exploded."},
    }
