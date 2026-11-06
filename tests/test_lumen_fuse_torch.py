"""LumenFuse torch backend: parity, fallback and numerical validation."""

from __future__ import annotations

import gc
import warnings

import numpy as np
import pytest
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from backend.classical_cv.scan import scan
from backend.enhancement.pipeline import EnhancementPipeline
from backend.fusion.lumen_fuse import (
    CANDIDATE_COUNT,
    LumenFuseConfig,
    compute_weights,
    fuse,
)

torch = pytest.importorskip("torch")

from backend.fusion.torch_backend import TorchFusionOps  # noqa: E402
from tests.conftest import StubModelManager, brighten  # noqa: E402

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")
_backends = getattr(torch, "backends", None)
_mps = getattr(_backends, "mps", None)
if _mps is not None and _mps.is_available():
    DEVICES.append("mps")


def _textured(
    level: float,
    *,
    amp: float = 0.05,
    size: int = 96,
    seed: int = 0,
    tint: tuple[float, float, float] = (1.0, 0.92, 0.85),
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    texture = (rng.random((size, size, 1)) - 0.5) * 2.0 * amp
    base = np.asarray(tint, dtype=np.float32) * level
    return np.clip(base + texture, 0.0, 1.0).astype(np.float32)


def _cases() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(5)
    noisy = np.clip(
        _textured(0.05, amp=0.01) + rng.normal(0.0, 0.05, (96, 96, 3)), 0.0, 1.0
    ).astype(np.float32)
    black = np.zeros((64, 64, 3), dtype=np.float32)
    highlight = np.full((96, 96, 3), 0.4, dtype=np.float32)
    highlight[32:64, 32:64] = 0.95
    return {
        "textured": (_textured(0.1), _textured(0.4, seed=1), _textured(0.3, seed=2)),
        "odd_101x77": (
            _textured(0.1, size=101)[:77],
            _textured(0.4, size=101, seed=1)[:77],
            _textured(0.3, size=101, seed=2)[:77],
        ),
        "noisy_dark": (noisy, _textured(0.6, seed=1), _textured(0.2, seed=2)),
        "all_black": (black, black.copy(), black.copy()),
        "highlight": (highlight, _textured(0.6, seed=1), _textured(0.55, seed=2)),
        "colour_cast": (
            _textured(0.15, tint=(0.9, 0.75, 0.55)),
            _textured(0.35, seed=1, tint=(0.8, 0.65, 0.45)),
            _textured(0.30, seed=2, tint=(0.7, 0.6, 0.5)),
        ),
    }


def _quality(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float, float, float]:
    delta = np.abs(reference - candidate)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        psnr = float(peak_signal_noise_ratio(reference, candidate, data_range=1.0))
    return (
        float(np.mean(delta)),
        float(np.max(delta)),
        psnr,
        float(
            structural_similarity(
                reference, candidate, channel_axis=-1, data_range=1.0
            )
        ),
    )


def _assert_close(
    reference,
    candidate,
    *,
    max_diff_limit: float = 1e-3,
    ssim_limit: float = 0.999,
) -> None:
    assert candidate.shape == reference.shape
    assert candidate.dtype == np.float32
    assert np.isfinite(candidate).all()
    assert float(candidate.min()) >= 0.0
    assert float(candidate.max()) <= 1.0

    mae, max_diff, psnr, ssim = _quality(reference, candidate)
    assert mae <= 1e-3, f"MAE {mae:.3e}"
    assert max_diff <= max_diff_limit, f"max diff {max_diff:.3e}"
    assert psnr >= 60.0, f"PSNR {psnr:.2f}"
    assert ssim >= ssim_limit, f"SSIM {ssim:.8f}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", sorted(_cases()))
def test_torch_backend_matches_cpu(device: str, case: str) -> None:
    original, classical, deep = _cases()[case]
    scan_result = scan(original)

    reference = fuse(
        original, classical, deep, scan_result=scan_result, backend="cpu"
    )
    candidate = fuse(
        original,
        classical,
        deep,
        scan_result=scan_result,
        backend="torch",
        device=device,
        precision="float32",
    )

    _assert_close(reference.image, candidate.image)
    _assert_close(reference.weights, candidate.weights)
    assert np.isfinite(candidate.initial_weights).all()
    assert float(
        np.mean(np.abs(reference.initial_weights - candidate.initial_weights))
    ) <= 1e-3
    assert float(
        np.max(np.abs(reference.initial_weights - candidate.initial_weights))
    ) <= 2e-3
    assert candidate.levels == reference.levels
    assert candidate.sigma == pytest.approx(reference.sigma)
    assert candidate.parameters == reference.parameters


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size", [(32, 48), (47, 63), (80, 100), (17, 19)])
def test_torch_backend_handles_different_dimensions(device: str, size: tuple[int, int]) -> None:
    height, width = size
    rng = np.random.default_rng(sum(size))
    original = np.clip(
        rng.random((height, width, 3)).astype(np.float32) * 0.4, 0.0, 1.0
    )
    classical = np.clip(original * 1.8, 0.0, 1.0).astype(np.float32)
    deep = np.clip(original * 1.5, 0.0, 1.0).astype(np.float32)
    scan_result = scan(original)

    reference = fuse(
        original, classical, deep, scan_result=scan_result, backend="cpu"
    )
    candidate = fuse(
        original,
        classical,
        deep,
        scan_result=scan_result,
        backend="torch",
        device=device,
    )

    _assert_close(reference.image, candidate.image)
    assert candidate.image.shape == (height, width, 3)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "config",
    [
        LumenFuseConfig(contrast_blur_sigma=1.5),
        LumenFuseConfig(laplacian_kernel=1),
        LumenFuseConfig(contrast_blur_sigma=0.8, laplacian_kernel=5),
    ],
)
def test_torch_backend_custom_kernels(device: str, config: LumenFuseConfig) -> None:
    rng = np.random.default_rng(0)
    original = np.clip(rng.random((77, 101, 3)).astype(np.float32) * 0.4, 0.0, 1.0)
    classical = np.clip(original * 1.8, 0.0, 1.0).astype(np.float32)
    deep = np.clip(original * 1.5, 0.0, 1.0).astype(np.float32)
    scan_result = scan(original)

    reference = fuse(
        original, classical, deep, scan_result=scan_result, config=config, backend="cpu"
    )
    candidate = fuse(
        original,
        classical,
        deep,
        scan_result=scan_result,
        config=config,
        backend="torch",
        device=device,
    )

    _assert_close(reference.image, candidate.image)


@pytest.mark.parametrize("device", DEVICES)
def test_torch_compute_weights_matches_cpu(device: str) -> None:
    original, classical, deep = _cases()["textured"]
    reference = compute_weights(original, classical, deep, backend="cpu")
    candidate = compute_weights(
        original, classical, deep, backend="torch", device=device
    )

    _assert_close(reference.refined, candidate.refined)
    assert np.isfinite(candidate.initial).all()
    assert float(np.mean(np.abs(reference.initial - candidate.initial))) <= 1e-3
    assert candidate.refined.shape[2] == CANDIDATE_COUNT


def test_default_backend_is_cpu() -> None:
    original, classical, deep = _cases()["textured"]
    default = fuse(original, classical, deep)
    explicit = fuse(original, classical, deep, backend="cpu")
    assert np.array_equal(default.image, explicit.image)
    assert np.array_equal(default.weights, explicit.weights)


def test_auto_uses_cpu_when_cuda_is_unavailable(monkeypatch) -> None:
    original, classical, deep = _cases()["textured"]
    monkeypatch.setattr(
        TorchFusionOps, "cuda_available", classmethod(lambda cls: False)
    )

    automatic = fuse(original, classical, deep, backend="auto")
    cpu = fuse(original, classical, deep, backend="cpu")

    assert np.array_equal(automatic.image, cpu.image)
    assert np.array_equal(automatic.weights, cpu.weights)


def test_auto_falls_back_when_gpu_initialization_fails(monkeypatch) -> None:
    original, classical, deep = _cases()["textured"]
    monkeypatch.setattr(
        TorchFusionOps, "cuda_available", classmethod(lambda cls: True)
    )

    def fail(self, *, device=None, precision="float32"):
        raise RuntimeError("synthetic GPU initialization failure")

    monkeypatch.setattr(TorchFusionOps, "__init__", fail)

    automatic = fuse(original, classical, deep, backend="auto")
    cpu = fuse(original, classical, deep, backend="cpu")

    assert np.array_equal(automatic.image, cpu.image)


def test_auto_with_explicit_cpu_device_uses_the_cpu_path(monkeypatch) -> None:
    original, classical, deep = _cases()["textured"]

    def fail(self, *, device=None, precision="float32"):
        raise AssertionError("auto with device='cpu' must not construct torch")

    monkeypatch.setattr(TorchFusionOps, "__init__", fail)
    automatic = fuse(original, classical, deep, backend="auto", device="cpu")
    cpu = fuse(original, classical, deep, backend="cpu")
    assert np.array_equal(automatic.image, cpu.image)


def test_invalid_backend_and_precision_are_rejected() -> None:
    original, classical, deep = _cases()["textured"]
    with pytest.raises(ValueError, match="backend"):
        fuse(original, classical, deep, backend="quantum")
    with pytest.raises(ValueError, match="precision"):
        fuse(original, classical, deep, backend="torch", precision="bfloat16")


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available here")
def test_explicit_cuda_request_fails_without_cuda() -> None:
    original, classical, deep = _cases()["textured"]
    with pytest.raises(RuntimeError, match="CUDA"):
        fuse(original, classical, deep, backend="torch", device="cuda")


@pytest.mark.parametrize("device", [d for d in DEVICES if d != "cpu"])
def test_fp16_runs_and_stays_close(device: str) -> None:
    original, classical, deep = _cases()["textured"]
    reference = fuse(original, classical, deep, backend="cpu")
    candidate = fuse(
        original,
        classical,
        deep,
        backend="torch",
        device=device,
        precision="float16",
    )

    assert candidate.image.shape == reference.image.shape
    assert np.isfinite(candidate.image).all()
    assert float(candidate.image.min()) >= 0.0
    assert float(candidate.image.max()) <= 1.0
    mae, _, psnr, ssim = _quality(reference.image, candidate.image)
    assert mae <= 5e-3, f"FP16 MAE {mae:.3e}"
    assert psnr >= 45.0, f"FP16 PSNR {psnr:.2f}"
    assert ssim >= 0.99, f"FP16 SSIM {ssim:.6f}"


def test_pipeline_environment_selects_the_torch_backend(monkeypatch) -> None:
    called = {"torch": False}
    original = TorchFusionOps.refine

    def wrapper(self, *args, **kwargs):
        called["torch"] = True
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TorchFusionOps, "refine", wrapper)
    monkeypatch.setenv("LUMENOIR_FUSE_BACKEND", "torch")
    monkeypatch.setenv("LUMENOIR_FUSE_DEVICE", "cpu")
    monkeypatch.setenv("LUMENOIR_FUSE_PRECISION", "float32")

    image = _textured(0.08, size=64)
    pil_image = Image.fromarray(np.round(image * 255.0).astype(np.uint8))
    pipeline = EnhancementPipeline(
        model_manager=StubModelManager(transform=brighten)
    )
    result = pipeline.run(pil_image, mode="hybrid")
    try:
        assert called["torch"] is True
        assert result.completed is True
        assert result.final_image is not None
    finally:
        result.release()
        pil_image.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_memory_is_released_between_calls() -> None:
    original, classical, deep = _cases()["textured"]
    scan_result = scan(original)

    fuse(
        original,
        classical,
        deep,
        scan_result=scan_result,
        backend="torch",
        device="cuda",
    )
    torch.cuda.synchronize()
    gc.collect()
    baseline = torch.cuda.memory_allocated()

    for _ in range(5):
        fuse(
            original,
            classical,
            deep,
            scan_result=scan_result,
            backend="torch",
            device="cuda",
        )
    torch.cuda.synchronize()
    gc.collect()

    growth = torch.cuda.memory_allocated() - baseline
    assert growth < 16 * 1024 * 1024
