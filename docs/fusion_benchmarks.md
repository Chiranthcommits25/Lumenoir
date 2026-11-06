# LumenFuse backend benchmarks

This document records measured LumenFuse and full-pipeline results. **No Kaggle
Tesla T4 measurement was performed in this environment.** The local machine is
an Apple Silicon Mac using PyTorch MPS, so the numbers below are useful for
correctness and relative behaviour only; they are not T4 results and must not be
reported as such.

The historical Kaggle reference is:

| Mode / stage | Historical T4, 1728×2304 |
| --- | ---: |
| Classical pipeline | 2.72 s |
| Deep-learning pipeline | 1.78 s |
| Hybrid pipeline | 8.04 s |
| LumenFuse stage | 5.16 s |

Those numbers come from different hardware, image dimensions and benchmark
conditions. They are context, not a comparison target for the local table.

## Independent LumenFuse benchmark

Reproduce with:

```bash
python -m training.benchmarks.benchmark_lumen_fuse \
  --device auto --warmups 2 --runs 10
```

The CNNs and candidates are identical across backends; `scan()` is excluded from
the timings. Torch timings include the CPU→device input copy and the
device→CPU output copy. CUDA timings are synchronized before and after each
call. Cold start is the first call for a configuration; the first torch
accelerator call also includes device-context initialization.

### Representative 2304×1728 image

| Backend | Cold ms | Warm-up ms | Median ms | Mean ms | P90 ms | CPU % | GPU % | Peak MB | MAE | Max diff | PSNR | SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU (OpenCV) | 1013.49 | 864.77 | 840.03 | 845.18 | 863.63 | 109.8 | n/a | n/a | 0 | 0 | inf | 1.000 |
| torch FP32 (MPS) | 1060.24 | 636.16 | 627.90 | 631.97 | 647.81 | 15.6 | n/a | n/a | 8.49e-08 | 1.10e-06 | 137.06 | 1.000 |
| torch FP16 (MPS) | 411.29 | 394.15 | 484.15 | 501.87 | 566.58 | 24.6 | n/a | n/a | 5.57e-04 | 2.72e-03 | 62.13 | 1.000 |

On this machine the local median improves from 840 ms (CPU) to 628 ms (FP32 MPS)
and 484 ms (FP16 MPS) for the representative size. This is a local MPS result,
not a T4 prediction. MPS peak memory is not exposed by the benchmark; CUDA will
report `max_memory_allocated`.

### Size summary (median ms)

| Size | CPU | torch FP32 | torch FP16 |
| --- | ---: | ---: | ---: |
| 256×256 | 14.25 | 31.33 | 29.14 |
| 640×480 | 65.24 | 50.16 | 46.69 |
| 1280×960 | 260.78 | 202.92 | 148.55 |
| 2304×1728 | 840.03 | 627.90 | 484.15 |

The torch backend is slower on the smallest image because transfer and launch
overhead dominate. This is why `auto` only selects CUDA, not MPS/CPU, and why
the CPU fallback remains the default on non-CUDA systems.

### Numerical equivalence

Across textured, dark, noisy, colour-cast, all-black, highlighted and odd-sized
inputs (17×19 through 640×640):

- CPU vs torch FP32: image MAE ≤ 1.3e-6, max absolute difference ≤ 7.9e-5,
  PSNR ≥ 103 dB, SSIM = 1.0.
- Refined weights: MAE ≤ 5e-7, max absolute difference ≤ 7e-6.
- The sharp highlight edge matches to image MAE ≈ 6e-8 and max difference
  ≈ 7e-7; the diagnostic pre-refinement weights differ by at most ≈ 1e-3 at
  individual pixels because the weight exponents amplify tiny contrast
  differences in flat regions.
- FP16 on MPS: MAE ≤ 5.6e-4, max difference ≤ 2.8e-3, PSNR ≥ 62 dB,
  SSIM = 1.0. FP16 remains opt-in.

No smoothing or post-processing is applied to hide differences.

## Full pipeline benchmark

The existing Kaggle evaluation harness was run on the same 10 LOL-v2 Real test
images at their native 600×400 resolution, with the CPU Zero-DCE++ checkpoint,
LPIPS/NIQE skipped, latency benchmark disabled and shared-stage caching enabled.
Two runs differed only in the LumenFuse backend:

```bash
# CPU fusion
LUMENOIR_FUSE_BACKEND=cpu python -m training.kaggle_eval.benchmark \
  --config /kaggle/working/eval_config.json --images 10

# Torch fusion (use cuda on Kaggle)
LUMENOIR_FUSE_BACKEND=torch LUMENOIR_FUSE_DEVICE=mps \
  python -m training.kaggle_eval.benchmark \
  --config /kaggle/working/eval_config.json --images 10
```

| Measurement | CPU fusion | Torch FP32 MPS fusion |
| --- | ---: | ---: |
| Harness total, old sharing | 7.25 s | 7.26 s |
| Harness total, shared stages | 6.62 s | 6.54 s |
| Classical mode, 10 images | 652.5 ms | 653.0 ms |
| Deep-learning mode, 10 images | 509.2 ms | 508.0 ms |
| Hybrid mode, 10 images | 1264.8 ms | 1193.3 ms |
| LumenFuse stage, mean/image | 52.83 ms | 45.23 ms |

At 600×400 the harness is dominated by metric computation, image I/O and report
writing, so the fusion backend changes the total by only a few percent. The
fusion stage itself is also small at this resolution. The representative-size
independent benchmark above is the relevant measurement for the GPU backend.

### Full-pipeline output equivalence

Comparing the 30 saved hybrid visuals from the CPU and MPS runs:

- max MAE: 2.97e-05
- min PSNR: 69.33 dB
- min SSIM: 0.99999
- metric-value differences: PSNR ≤ 0.0016 dB, SSIM ≤ 1.5e-5, CIEDE2000 ≤ 0.002

These are within the stated acceptance targets and show no visible colour,
halo or edge degradation.

## Kaggle T4 validation commands

Run the independent benchmark on the T4:

```bash
LUMENOIR_FUSE_BACKEND=torch LUMENOIR_FUSE_DEVICE=cuda \
python -m training.benchmarks.benchmark_lumen_fuse \
  --device cuda --warmups 5 --runs 20 \
  --output-dir /kaggle/working/lumenoir-fuse-benchmark
```

Then repeat the full evaluation harness with CPU and CUDA fusion:

```bash
LUMENOIR_FUSE_BACKEND=cpu \
python -m training.kaggle_eval.benchmark \
  --config /kaggle/working/eval_config.json --images 10

LUMENOIR_FUSE_BACKEND=torch LUMENOIR_FUSE_DEVICE=cuda \
python -m training.kaggle_eval.benchmark \
  --config /kaggle/working/eval_config.json --images 10
```

Use `LUMENOIR_FUSE_PRECISION=float16` for the FP16 variant. Only claim a T4
speed-up after these commands have run on the T4 with the same checkpoint,
images and configuration.

## Known limitations

- Local results use Apple MPS, not CUDA/T4.
- MPS peak memory is not reported; CUDA uses `max_memory_allocated`.
- The torch path pays input/output transfer overhead each call. It is not
  beneficial for very small images.
- FP16 quality was measured on a limited set; production defaults to FP32.
- The full-pipeline local benchmark is 600×400, not the historical 1728×2304.
- The algorithm, exponents, weights and quality mechanisms are unchanged.
