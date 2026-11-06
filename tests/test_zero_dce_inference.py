"""Zero-DCE++ unit tests: architecture, checkpoint validation, devices, manager.

No network and no published checkpoint are used. Checkpoints are random
initialisations of the same architecture written to ``tmp_path``. The real
pretrained checkpoint is covered by ``tests/test_zero_dce_integration.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from backend.config import InferenceSettings
from backend.deep_learning.zero_dce_plus_plus import (
    EXPECTED_PARAMETERS,
    EXPECTED_TENSORS,
    MODEL_ID,
    ZeroDCEPlusPlusModel,
)
from backend.inference import device as device_module
from backend.inference.errors import (
    CheckpointError,
    DeviceUnavailableError,
    TorchNotInstalledError,
)
from backend.inference.manager import ModelManager, ModelState

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def torch_module():
    return pytest.importorskip("torch")


@pytest.fixture(scope="module")
def net(torch_module):
    from backend.deep_learning.zero_dce_net import ZeroDCEPlusPlusNet

    return ZeroDCEPlusPlusNet(scale_factor=12)


def _save_checkpoint(net, torch, path: Path) -> Path:
    torch.save(net.state_dict(), path)
    return path


def _image(size: tuple[int, int] = (130, 100)) -> Image.Image:
    image = Image.new("RGB", size, (24, 30, 36))
    for x in range(size[0]):
        for y in range(size[1] // 4):
            image.putpixel((x, y), (200, 190, 170))
    return image


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


def test_architecture_matches_zero_dce_plus_plus(net) -> None:
    assert len(net.state_dict()) == EXPECTED_TENSORS
    assert sum(p.numel() for p in net.parameters()) == EXPECTED_PARAMETERS


def test_curve_alpha_convention(torch_module, net) -> None:
    tensor = torch_module.rand(1, 3, 48, 60)
    with torch_module.inference_mode():
        enhanced, raw_alpha = net(tensor)

    curve_alpha = -raw_alpha
    manual = tensor
    for _ in range(8):
        manual = manual + curve_alpha * manual * (1.0 - manual)

    assert torch_module.allclose(enhanced, manual, atol=1e-6)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


def test_wrapper_loads_a_strict_checkpoint(torch_module, net, tmp_path: Path) -> None:
    path = _save_checkpoint(net, torch_module, tmp_path / "model.pth")
    model = ZeroDCEPlusPlusModel(path, device="cpu", scale_factor=12)

    model.load()

    assert model.loaded is True
    assert model.parameter_count == EXPECTED_PARAMETERS
    assert model.info().weights_bytes == path.stat().st_size

    image = _image()
    result = model.run(image)

    assert result.size == image.size
    assert result.mode == "RGB"

    image.close()
    result.close()
    model.close()
    assert model.loaded is False


@pytest.mark.parametrize("size", [(100, 130), (64, 64), (5, 7), (13, 12)])
def test_padding_and_cropping_restore_the_exact_size(
    torch_module, net, tmp_path: Path, size: tuple[int, int]
) -> None:
    path = _save_checkpoint(net, torch_module, tmp_path / "model.pth")
    model = ZeroDCEPlusPlusModel(path, device="cpu")
    model.load()

    image = Image.new("RGB", size, (12, 13, 14))
    result = model.run(image)

    assert result.size == size

    image.close()
    result.close()
    model.close()


def test_alpha_is_exposed_in_both_conventions(torch_module, net, tmp_path: Path) -> None:
    path = _save_checkpoint(net, torch_module, tmp_path / "model.pth")
    model = ZeroDCEPlusPlusModel(path, device="cpu")
    model.load()

    image = _image((48, 48))
    prepared = model.preprocess(image)
    raw = model.infer(prepared)

    assert tuple(raw.raw_alpha.shape) == tuple(raw.curve_alpha.shape)
    assert torch_module.allclose(raw.curve_alpha, -raw.raw_alpha)

    image.close()
    model.close()


def test_checkpoint_with_wrong_parameters_is_rejected(torch_module, tmp_path: Path) -> None:
    path = tmp_path / "wrong.pth"
    torch_module.save({"e_conv1.weight": torch_module.zeros(3, 3, 3, 3)}, path)

    model = ZeroDCEPlusPlusModel(path, device="cpu")

    with pytest.raises(CheckpointError, match="parameters"):
        model.load()


def test_original_zero_dce_sized_checkpoint_gets_a_hint(torch_module, tmp_path: Path) -> None:
    path = tmp_path / "big.pth"
    torch_module.save({"e_conv1.weight": torch_module.zeros(80_000)}, path)

    model = ZeroDCEPlusPlusModel(path, device="cpu")

    with pytest.raises(CheckpointError, match="original Zero-DCE"):
        model.load()


def test_missing_checkpoint_is_rejected(tmp_path: Path) -> None:
    model = ZeroDCEPlusPlusModel(tmp_path / "absent.pth", device="cpu")

    with pytest.raises(CheckpointError, match="does not exist"):
        model.load()


def test_scale_factor_can_be_set_explicitly(net, torch_module, tmp_path: Path) -> None:
    path = _save_checkpoint(net, torch_module, tmp_path / "model.pth")

    model = ZeroDCEPlusPlusModel(path, device="cpu", scale_factor=3)
    assert model.scale_factor == 3

    model.load()
    image = Image.new("RGB", (64, 64), (10, 11, 12))
    result = model.run(image)
    assert result.size == (64, 64)

    image.close()
    result.close()
    model.close()

    with pytest.raises(ValueError):
        ZeroDCEPlusPlusModel(path, device="cpu", scale_factor=99)


def test_scale_factor_is_read_from_the_environment(
    monkeypatch, net, torch_module, tmp_path: Path
) -> None:
    path = _save_checkpoint(net, torch_module, tmp_path / "model.pth")
    monkeypatch.setenv("LUMENOIR_ZERODCE_SCALE_FACTOR", "4")

    assert ZeroDCEPlusPlusModel(path, device="cpu").scale_factor == 4

    monkeypatch.setenv("LUMENOIR_ZERODCE_SCALE_FACTOR", "99")
    with pytest.raises(ValueError):
        ZeroDCEPlusPlusModel(path, device="cpu")


# ---------------------------------------------------------------------------
# Manager integration
# ---------------------------------------------------------------------------


def test_manager_registers_loads_and_infers(torch_module, net, tmp_path: Path) -> None:
    _save_checkpoint(net, torch_module, tmp_path / f"{MODEL_ID}.pth")
    manager = ModelManager(
        InferenceSettings(model_id=MODEL_ID, weights_dir=tmp_path, device="cpu")
    )

    status = manager.load_at_startup()

    assert status.state is ModelState.READY
    assert status.device == "cpu"
    assert status.ready is True

    image = Image.new("RGB", (96, 80), (20, 22, 24))
    outcome = manager.infer(image)

    assert outcome.image.size == image.size
    assert outcome.model_id == MODEL_ID

    image.close()
    outcome.image.close()
    manager.close()
    assert manager.status.state is ModelState.NOT_LOADED


def test_manager_records_device_failure(monkeypatch, torch_module, net, tmp_path: Path) -> None:
    _save_checkpoint(net, torch_module, tmp_path / f"{MODEL_ID}.pth")
    manager = ModelManager(
        InferenceSettings(model_id=MODEL_ID, weights_dir=tmp_path, device="mps")
    )

    def fail(_requested: str) -> str:
        raise DeviceUnavailableError("mps", "MPS is not available in this test.")

    monkeypatch.setattr("backend.inference.manager.resolve_device", fail)

    status = manager.load_at_startup()

    assert status.state is ModelState.FAILED
    assert "MPS is not available" in status.message
    assert manager.status.ready is False


# ---------------------------------------------------------------------------
# Device resolver
# ---------------------------------------------------------------------------


def test_cpu_resolution_does_not_require_a_device() -> None:
    assert device_module.resolve_device(None) == "cpu"
    assert device_module.resolve_device("cpu") == "cpu"
    assert device_module.resolve_device(" default ") == "cpu"


def test_auto_prefers_mps(monkeypatch) -> None:
    monkeypatch.setattr(device_module, "mps_available", lambda _torch=None: True)
    monkeypatch.setattr(device_module, "cuda_available", lambda _torch=None: True)

    assert device_module.resolve_device("auto") == "mps"


def test_auto_falls_back_to_cuda_then_cpu(monkeypatch) -> None:
    monkeypatch.setattr(device_module, "mps_available", lambda _torch=None: False)
    monkeypatch.setattr(device_module, "cuda_available", lambda _torch=None: True)
    assert device_module.resolve_device("auto") == "cuda:0"

    monkeypatch.setattr(device_module, "cuda_available", lambda _torch=None: False)
    assert device_module.resolve_device("auto") == "cpu"


def test_explicit_unavailable_mps_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(device_module, "mps_available", lambda _torch=None: False)

    with pytest.raises(DeviceUnavailableError) as excinfo:
        device_module.resolve_device("mps")

    assert excinfo.value.code == "device_unavailable"


def test_cuda_index_is_validated(monkeypatch, torch_module) -> None:
    monkeypatch.setattr(device_module, "cuda_available", lambda _torch=None: True)
    monkeypatch.setattr(torch_module.cuda, "device_count", lambda: 1)

    assert device_module.resolve_device("cuda") == "cuda:0"
    assert device_module.resolve_device("cuda:0") == "cuda:0"

    with pytest.raises(DeviceUnavailableError):
        device_module.resolve_device("cuda:5")
    with pytest.raises(DeviceUnavailableError):
        device_module.resolve_device("cuda:not-a-number")


def test_unsupported_device_is_rejected() -> None:
    with pytest.raises(DeviceUnavailableError):
        device_module.resolve_device("tpu")


def test_torch_missing_error_is_typed() -> None:
    error = TorchNotInstalledError()

    assert error.code == "torch_missing"
    assert error.status_code == 503
    assert "requirements-inference.txt" in error.message


# ---------------------------------------------------------------------------
# Import safety
# ---------------------------------------------------------------------------


def test_api_import_does_not_load_torch() -> None:
    code = (
        "import sys; import backend.main; "
        "assert 'torch' not in sys.modules, 'torch was imported'"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
