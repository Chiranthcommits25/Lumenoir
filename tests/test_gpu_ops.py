"""Accelerator parity for the torch image operations and classical stages."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from backend.classical_cv.edge_guard import denoise as edge_guard_cpu  # noqa: E402
from backend.classical_cv.hue_lock import correct as hue_lock_cpu  # noqa: E402
from backend.classical_cv.retinex import enhance as retinex_cpu  # noqa: E402
from backend.classical_cv.scan import scan  # noqa: E402
from backend.classical_cv.torch_edge_guard import denoise_tensor  # noqa: E402
from backend.classical_cv.torch_hue_lock import correct_tensor  # noqa: E402
from backend.classical_cv.torch_retinex import enhance_tensor  # noqa: E402
from backend.classical_cv.torch_scan import scan_tensor  # noqa: E402
from backend.fusion.lumen_fuse import fuse, fuse_tensors  # noqa: E402
from backend.fusion.torch_backend import TorchFusionOps  # noqa: E402
from backend.runtime.torch_ops import TorchImageOps  # noqa: E402

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")
_backends = getattr(torch, "backends", None)
_mps = getattr(_backends, "mps", None)
if _mps is not None and _mps.is_available():
    DEVICES.append("mps")


@pytest.fixture(params=DEVICES)
def ops(request):
    return TorchImageOps(device=request.param)


def _low_light(height: int = 72, width: int = 96, seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.stack(
        [
            0.04 + 0.18 * xx / width,
            0.03 + 0.12 * yy / height,
            0.02 + 0.08 * (xx + yy) / (width + height),
        ],
        axis=-1,
    )
    image = image + rng.normal(0.0, 0.006, image.shape)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def test_rgb_to_lab_matches_opencv(ops) -> None:
    import cv2

    rng = np.random.default_rng(0)
    rgb = rng.random((61, 73, 3), dtype=np.float32)
    expected = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    actual = ops.to_numpy(ops.rgb_to_lab(ops.ensure(rgb)))
    assert np.abs(actual - expected).max() <= 0.02
    assert np.abs(actual - expected).mean() <= 0.002


def test_clahe_matches_opencv_levels(ops) -> None:
    import cv2

    rng = np.random.default_rng(1)
    source = rng.integers(0, 65536, (90, 121), dtype=np.uint16)
    reference = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(source)
    actual = ops.to_numpy(
        ops.clahe_uint16(ops.ensure(source.astype(np.float32)), 2.0, (8, 8))
    ).astype(np.uint16)
    delta = np.abs(actual.astype(np.int32) - reference.astype(np.int32))
    assert delta.max() <= 1
    assert delta.mean() <= 0.05


def test_area_resize_matches_opencv(ops) -> None:
    import cv2

    rng = np.random.default_rng(2)
    for height, width in ((64, 96), (63, 95)):
        rgb = rng.random((height, width, 3), dtype=np.float32)
        reference = cv2.resize(rgb, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
        actual = ops.to_numpy(ops.resize(ops.ensure(rgb), (width // 2, height // 2), mode="area"))
        assert np.abs(actual - reference).max() <= 2e-5


def test_filters_match_opencv(ops) -> None:
    import cv2

    rng = np.random.default_rng(3)
    image = rng.random((58, 79), dtype=np.float32)
    tensor = ops.ensure(image)
    assert np.abs(
        ops.to_numpy(ops.box_filter(tensor, 4))
        - cv2.boxFilter(
            image,
            -1,
            (9, 9),
            normalize=True,
            borderType=cv2.BORDER_REFLECT,
        )
    ).max() <= 5e-6
    assert np.abs(
        ops.to_numpy(ops.gaussian_blur(tensor, 1.5))
        - cv2.GaussianBlur(image, (0, 0), sigmaX=1.5)
    ).max() <= 5e-6
    guide = rng.random((58, 79), dtype=np.float32)
    reference = (
        cv2.ximgproc.guidedFilter(guide, image, 4, 1e-2)
        if hasattr(cv2, "ximgproc")
        else None
    )
    actual = ops.to_numpy(ops.guided_filter(ops.ensure(guide), tensor, 4, 1e-2))
    if reference is not None:
        assert np.abs(actual - reference).max() <= 2e-5


def test_noise_matches_cpu(ops) -> None:
    from backend.classical_cv.noise import noise_sigma

    rng = np.random.default_rng(5)
    image = rng.random((80, 101), dtype=np.float32)
    expected = noise_sigma(image)
    actual = ops.noise_sigma(ops.ensure(image))
    assert abs(actual - expected) <= 1e-6


def test_scan_matches_cpu(ops) -> None:
    image = _low_light()
    expected = scan(image)
    actual = scan_tensor(ops.ensure(image), ops=ops)
    ef, af = expected.features, actual.features
    for name in (
        "mean",
        "p5",
        "p50",
        "p95",
        "dynamic_range",
        "nonuniformity",
        "noise_sigma",
        "highlight_fraction",
        "black_fraction",
    ):
        assert abs(getattr(af, name) - getattr(ef, name)) <= 1e-5, name
    assert af.darkness == ef.darkness
    assert actual.parameters.bypass_gate == expected.parameters.bypass_gate
    assert abs(actual.parameters.gamma - expected.parameters.gamma) <= 1e-6
    assert ops.to_numpy(actual.illumination).shape == expected.illumination.shape


def test_retinex_edge_and_hue_match_cpu(ops) -> None:
    image = _low_light(seed=9)
    tensor = ops.ensure(image)
    scan_cpu = scan(image)
    scan_gpu = scan_tensor(tensor, ops=ops)

    classical_cpu = retinex_cpu(image, scan_result=scan_cpu)
    classical_gpu = enhance_tensor(tensor, scan_result=scan_gpu, ops=ops)
    classical = ops.to_numpy(classical_gpu.image)
    delta = np.abs(classical - classical_cpu.image)
    assert delta.mean() <= 0.002
    assert delta.max() <= 0.03

    edge_cpu = edge_guard_cpu(classical_cpu.image, image, scan_result=scan_cpu)
    edge_gpu = denoise_tensor(
        classical_gpu.image, tensor, scan_result=scan_gpu, ops=ops
    )
    edge = ops.to_numpy(edge_gpu.image)
    delta = np.abs(edge - edge_cpu.image)
    assert delta.mean() <= 0.004
    assert delta.max() <= 0.04

    hue_cpu = hue_lock_cpu(edge_cpu.image, scan_result=scan_cpu)
    hue_gpu = correct_tensor(edge_gpu.image, scan_result=scan_gpu, ops=ops)
    hue = ops.to_numpy(hue_gpu.image)
    delta = np.abs(hue - hue_cpu.image)
    assert delta.mean() <= 0.004
    assert delta.max() <= 0.04


def test_fuse_tensors_matches_cpu(ops) -> None:
    original = _low_light(seed=11)
    classical = np.clip(original * 2.2, 0.0, 1.0).astype(np.float32)
    deep = np.clip(original * 1.6 + 0.04, 0.0, 1.0).astype(np.float32)
    scan_result = scan(original)
    expected = fuse(original, classical, deep, scan_result=scan_result, backend="cpu")
    gpu_ops = TorchFusionOps(device=str(ops.device), precision="float32")
    result = fuse_tensors(
        gpu_ops.ensure(original),
        gpu_ops.ensure(classical),
        gpu_ops.ensure(deep),
        noise_map=gpu_ops.ensure(scan_result.noise_map),
        parameters=scan_result.parameters,
        ops=gpu_ops,
        sigma=float(scan_result.features.noise_sigma),
    )
    actual = gpu_ops.to_numpy(result.image)
    assert actual.shape == expected.image.shape
    assert np.abs(actual - expected.image).mean() <= 1e-4
    assert np.abs(actual - expected.image).max() <= 5e-4


def test_pipeline_accelerator_matches_cpu(ops) -> None:
    if str(ops.device) == "cpu":
        pytest.skip("The pipeline only selects torch for an accelerator device.")
    from backend.config import InferenceSettings
    from backend.enhancement.pipeline import EnhancementPipeline
    from backend.inference.manager import ModelManager
    from backend.runtime.config import resolve_runtime

    image = Image.fromarray((_low_light(seed=12) * 255).round().astype(np.uint8))
    settings = InferenceSettings(model_id="", device="cpu")
    cpu = EnhancementPipeline(
        model_manager=ModelManager(settings), runtime=resolve_runtime("cpu")
    ).run(image, mode="classical")
    assert cpu.final_image is not None
    accelerated = EnhancementPipeline(
        model_manager=ModelManager(settings),
        runtime=resolve_runtime(str(ops.device)),
    ).run(image, mode="classical")
    assert accelerated.final_image is not None
    reference = np.asarray(cpu.final_image, dtype=np.float32) / 255.0
    actual = np.asarray(accelerated.final_image, dtype=np.float32) / 255.0
    delta = np.abs(actual - reference)
    assert delta.mean() <= 0.005
    assert delta.max() <= 0.06
    cpu.release()
    accelerated.release()


def test_gpu_metrics_match_skimage(ops, monkeypatch) -> None:
    if str(ops.device) == "cpu":
        pytest.skip("GPU metrics are only selected for an accelerator device.")
    from backend.evaluation import gpu_metrics
    from backend.evaluation.harness import evaluate_arrays

    monkeypatch.setenv("LUMENOIR_EVAL_DEVICE", str(ops.device))
    monkeypatch.setattr(gpu_metrics, "_STATE", {})
    rng = np.random.default_rng(8)
    reference = rng.random((48, 64, 3), dtype=np.float32)
    image = np.clip(reference + rng.normal(0.0, 0.02, reference.shape), 0.0, 1.0)
    cpu = evaluate_arrays(
        image,
        reference=reference,
        keys=["psnr", "ssim"],
        assume_valid=True,
    )
    monkeypatch.setenv("LUMENOIR_EVAL_DEVICE", "cpu")
    monkeypatch.setattr(gpu_metrics, "_STATE", {})
    baseline = evaluate_arrays(
        image,
        reference=reference,
        keys=["psnr", "ssim"],
        assume_valid=True,
    )
    assert cpu["psnr"].value is not None
    assert cpu["ssim"].value is not None
    assert abs(cpu["psnr"].value - baseline["psnr"].value) <= 1e-4
    assert abs(cpu["ssim"].value - baseline["ssim"].value) <= 1e-5
