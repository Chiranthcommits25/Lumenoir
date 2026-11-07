# GPU-first pipeline

The accelerator path runs the whole enhancement pipeline on one device: LumenScan,
LumenRetinex, Zero-DCE++, LumenFuse, EdgeGuard, HueLock and the final alpha blend.
Only pixel-artifact encoding and the JSON API cross back to the host. The CPU
NumPy/OpenCV implementation remains the reference and the fallback.

## Execution model

| Stage | CPU reference | Accelerator implementation |
| --- | --- | --- |
| Analysis | `classical_cv/scan.py` | `classical_cv/torch_scan.py` |
| Classical CV | `classical_cv/retinex.py` | `classical_cv/torch_retinex.py` |
| Deep learning | Zero-DCE++ `run(PIL)` | Zero-DCE++ `infer_tensor(tensor)` |
| Fusion | `fusion/lumen_fuse.py` (`_NumpyOps`) | `fusion/lumen_fuse.py` (`fuse_tensors`) with `TorchFusionOps` |
| Denoising | `classical_cv/edge_guard.py` | `classical_cv/torch_edge_guard.py` |
| Colour correction | `classical_cv/hue_lock.py` | `classical_cv/torch_hue_lock.py` |
| Quality assessment | NumPy alpha blend | Tensor alpha blend |

The shared primitives live in `backend/runtime/torch_ops.py`
(`TorchImageOps`). `TorchFusionOps` subclasses it for the fusion protocol.
Parameter rules and dataclasses are reused from the CPU modules, so the two
paths cannot drift on coefficients.

Device tensors are passed through `StageContext` fields (`rgb_tensor`,
`classical_tensor`, `deep_learning_tensor`, `fused_tensor`, `enhanced_tensor`,
`final_tensor`). `SharedStageCache` replays these fields together with the NumPy
ones, so repeated-mode evaluation reuses analysis/classical/deep-learning work
within one image. A pipeline-wide lock serialises accelerator runs, and stage
timings synchronise CUDA/MPS before stopping the clock.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `LUMENOIR_DEVICE` | `auto` | `auto`, `cpu`, `mps`, `cuda`, `cuda:N`. `auto` selects CUDA when available, otherwise the CPU reference pipeline. MPS is explicit. |
| `LUMENOIR_PRECISION` | `float32` | `float32` or `float16`; CPU is always float32. |
| `LUMENOIR_INFERENCE_DEVICE` | value of `LUMENOIR_DEVICE` | Device used to load Zero-DCE++ and run tensor inference. |
| `LUMENOIR_INFERENCE_PRECISION` | value of `LUMENOIR_PRECISION` | Zero-DCE++ inference precision. |
| `LUMENOIR_EVAL_DEVICE` | `cpu` | `cuda`, `cuda:N`, `mps` or `auto` selects GPU PSNR/SSIM; `cpu` selects skimage. |
| `LUMENOIR_FUSE_BACKEND` | `auto` | Legacy CPU-only fusion dispatcher; ignored when the unified pipeline is already on tensor devices. |
| `LUMENOIR_FUSE_DEVICE`, `LUMENOIR_FUSE_PRECISION` | unset | Legacy overrides for `fuse()` callers. |

Explicit unavailable devices raise; `auto` falls back to the CPU pipeline with a
warning. On a CUDA host the GPU-first configuration is:

```bash
LUMENOIR_DEVICE=cuda LUMENOIR_PRECISION=float32 \
LUMENOIR_INFERENCE_DEVICE=cuda \
LUMENOIR_MODEL_ID=zero_dce_plus_plus \
python -m uvicorn backend.main:app
```

For local Apple Silicon experiments set both `LUMENOIR_DEVICE=mps` and
`LUMENOIR_INFERENCE_DEVICE=mps`; `auto` deliberately keeps the CPU reference on
Macs.

## Numerical parity

The torch operations mirror the OpenCV conventions where it matters:

- RGB→LAB uses the OpenCV 5 33³ trilinear LUT path (max 0.016, mean 0.0005 LAB
  units versus `cv2.cvtColor` on random images).
- CLAHE uses 65536-bin histograms, the OpenCV clip limit and tile interpolation
  (≤1 uint16 level difference on the same input, including version-specific
  padding rules).
- INTER_AREA downscaling uses exact fractional overlap weights (≤5e-6).
- Box, Gaussian and guided filters match OpenCV (≤5e-6); Immerkaer noise is
  exact to float32.
- Bilateral chroma filtering follows the textbook kernel and can differ from
  OpenCV's implementation by about 1% on random noise; this is documented and
  accepted for the chroma denoiser.

Measured parity on **Apple Silicon MPS** (torch 2.15.0, Python 3.14.3, macOS
arm64), CPU reference versus accelerator. These are MPS measurements; there are
no CUDA measurements in this repository.

LumenFuse, FP32, four synthetic cases:

| Case | MAE | Max diff | PSNR | SSIM |
| --- | ---: | ---: | ---: | ---: |
| textured | 3.7e-08 | 2.7e-07 | 146.05 dB | 1.00000000 |
| odd 101×77 | 1.23e-06 | 7.9e-05 | 103.67 dB | 0.99999982 |
| colour_cast | 1.8e-08 | 1.0e-07 | 152.47 dB | 1.00000000 |
| noisy_dark | 2.4e-08 | 1.8e-07 | 149.43 dB | 1.00000000 |

LumenFuse, FP16 on MPS: MAE 7.5e-05–1.4e-04, max diff ≤7.9e-04, PSNR
75.0–80.5 dB, SSIM ≥0.999989.

Per-stage, FP32 on MPS (`noisy_dark`, 96×96):

| Stage | MAE | Max diff | PSNR | SSIM |
| --- | ---: | ---: | ---: | ---: |
| LumenRetinex | 8.4e-05 | 2.9e-02 | 66.05 dB | 0.99999714 |
| EdgeGuard | 3.3e-03 | 2.8e-02 | 47.28 dB | 0.99946982 |
| HueLock | 3.3e-03 | 2.8e-02 | 47.28 dB | 0.99946982 |

LumenScan features agree to float32 round-off (largest absolute delta 4.4e-09 on
`noise_sigma`, 1.9e-09 on `p50`), so the rule set selects identical
coefficients on both paths.

Full pipeline, real checkpoint, 320×240, CPU versus MPS (two seeds each):

| Mode | MAE | Max diff | PSNR | SSIM |
| --- | ---: | ---: | ---: | ---: |
| classical | 9.5e-04 | 1.6e-02 | 54.21 dB | 0.999849 |
| deep_learning | 8.6e-04 | 1.2e-02 | 54.65 dB | 0.997078 |
| hybrid | 9.4e-04 | 1.2e-02 | 54.29 dB | 0.999795 |

The dominant residual is the chroma bilateral approximation in EdgeGuard plus
quantisation differences between the PIL and tensor inference paths.
Deep-learning mode shows the lowest SSIM because the CPU path rounds the model
output through PIL uint8 while the tensor path keeps float precision. No
algorithmic parameter is changed.

## Benchmarks

Run the full CPU/accelerator comparison with the model loaded once per device:

```bash
python -m training.benchmarks.benchmark_pipeline \
    --devices cpu,mps --sizes 256x256,640x480,1280x960,2304x1728 \
    --warmup 1 --repeat 3 \
    --output training/benchmarks/results/pipeline_benchmark.json
```

On a CUDA T4 use `--devices cuda --precision float32` (and optionally
`float16`). The report records wall time, pipeline total, per-stage medians,
CPU RSS and CUDA/MPS memory. Timings are synchronised.

Local observations: Apple Silicon (M-series), torch 2.15.0, Python 3.14.3,
synthetic images, real checkpoint, hybrid mode, one warm-up, median of two runs,
model loaded once per device. Pipeline total, **not** T4 predictions:

| Device | 640×480 | 1280×960 | 2304×1728 |
| --- | ---: | ---: | ---: |
| CPU float32 | 159.9 ms | 600.9 ms | 1943.6 ms |
| MPS float32 | 346.9 ms | 837.1 ms | 3226.9 ms |

Per-stage medians at 1280×960:

| Stage | CPU | MPS |
| --- | ---: | ---: |
| analysis | 52.5 ms | 109.0 ms |
| classical_cv | 113.1 ms | 149.4 ms |
| deep_learning | 43.7 ms | 43.4 ms |
| fusion | 273.1 ms | 200.6 ms |
| denoising | 83.1 ms | 95.6 ms |
| colour_correction | 25.3 ms | 220.6 ms |
| quality_assessment | 10.1 ms | 18.4 ms |

On this hardware the accelerator path is **slower end-to-end** than the CPU
reference: only fusion and Zero-DCE++ improve, while analysis (quantile, CLAHE),
EdgeGuard and HueLock lose more to kernel-launch overhead than they gain from
the GPU. Peak RSS was 391 MB / 837 MB / 1386 MB on CPU and 1457 MB on MPS for
the three sizes.

A LumenFuse-only T4 benchmark was reported in an earlier session at 2304×1728
(CPU 2068.88 ms, CUDA FP32 529.19 ms, CUDA FP16 418.76 ms). It is not
reproduced here and no JSON for it is stored in the repository; treat it as
unverified until it is re-run with the commands below.

## Tests

Default tests run the CPU reference path. The GPU parity tests are unit tests
that use CPU torch and any available MPS/CUDA device:

```bash
.venv/bin/python -m pytest -q tests/test_gpu_ops.py tests/test_lumen_fuse_torch.py
```

They cover LAB/CLAHE/filter/resize parity, all GPU classical stages, tensor
fusion, GPU PSNR/SSIM, and a full pipeline parity check when an accelerator is
available.

Last run in this repository (Apple Silicon, MPS available, no CUDA):
`test_gpu_ops.py` 18 passed / 2 skipped, `test_lumen_fuse_torch.py` 36 passed /
1 skipped, whole suite 394 passed / 3 skipped / 8 deselected. The skips are the
CUDA-only memory test and the accelerator-only branches running on the CPU
parameterisation.

## Known limitations

- **No CUDA validation in this tree.** Every number on this page was measured on
  Apple MPS or the CPU reference. The CUDA paths (`cuda`, `cuda:N`) share the same
  ops but have never run on NVIDIA hardware.
- **MPS is slower than the CPU reference end-to-end** at the measured sizes. It
  exists for local validation, not for production speed.
- **`auto` never selects MPS.** On macOS `LUMENOIR_DEVICE=auto` resolves to the
  CPU reference so a developer machine cannot silently change numerics.
- **FP16 is looser than FP32** (MAE ~1e-4, PSNR 75–80 dB on MPS) and stays
  opt-in; FP32 is the default on every device.
- **Chroma bilateral** uses the textbook kernel rather than OpenCV's
  implementation and can differ by ~1% on noise-dominated chroma; this is the
  main contributor to the EdgeGuard residual above.
- **CLAHE histogram memory** is 65536 int64 bins per tile, so a large tile grid
  costs `tiles_x * tiles_y * 65536 * 8` bytes on the device.
- **`torch.quantile` on CUDA** is limited to 16M elements; larger images fall
  back to per-quantile `kthvalue`, which is correct but slower.
- **uint16 CLAHE round-trip**: the accelerator CLAHE returns a `torch.uint16`
  tensor, a limited-support dtype. It is verified on CPU and MPS only.
- **`LUMENOIR_FUSE_BACKEND` / `_DEVICE` / `_PRECISION`** remain for standalone
  `fuse()` callers; the unified pipeline ignores them.

## Kaggle T4 validation

Run this in a Kaggle notebook with `Accelerator: GPU T4 x2` and the repository
available at `/kaggle/working/LUMENOIR` (uploaded as a dataset or cloned).

```bash
# 1. Confirm real CUDA, not a CPU fallback
!nvidia-smi
!python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 2. Install the development stack
%cd /kaggle/working/LUMENOIR
!pip install -q -r requirements-dev.txt

# 3. Accelerator parity tests (CUDA is picked up automatically by the fixtures)
!PYTHONPATH=/kaggle/working/LUMENOIR python -m pytest -q tests/test_gpu_ops.py tests/test_lumen_fuse_torch.py

# 4. Full-suite regression on the GPU host
!PYTHONPATH=/kaggle/working/LUMENOIR python -m pytest -q

# 5. FP32 and FP16 pipeline benchmarks
!PYTHONPATH=/kaggle/working/LUMENOIR python -m training.benchmarks.benchmark_pipeline \
    --devices cpu,cuda --precision float32 \
    --sizes 640x480,1280x960,2304x1728 --warmup 2 --repeat 5 \
    --weights backend/weights/zero_dce_plus_plus.pth \
    --output /kaggle/working/pipeline_benchmark_fp32.json

!PYTHONPATH=/kaggle/working/LUMENOIR python -m training.benchmarks.benchmark_pipeline \
    --devices cuda --precision float16 \
    --sizes 640x480,1280x960,2304x1728 --warmup 2 --repeat 5 \
    --weights backend/weights/zero_dce_plus_plus.pth \
    --output /kaggle/working/pipeline_benchmark_fp16.json
```

Watch for these CUDA-specific failure modes while running steps 3 and 5:

- a device-mismatch `RuntimeError` inside `TorchImageOps`, meaning a torch
  constant was created on CPU while the image lives on CUDA;
- `torch.quantile` rejecting an oversized input (the >16M-element `kthvalue`
  fallback should engage instead);
- a `uint16` conversion error in `clahe_uint16`;
- dtype errors from float16 kernels in `TorchFusionOps` or the CLAHE LUT;
- `torch.cuda.OutOfMemoryError` at 2304×1728 with float16 fusion; lower
  `--repeat`, lower the size, or set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Record the JSON results, clearly labelled as CUDA T4, before quoting any T4
number.
