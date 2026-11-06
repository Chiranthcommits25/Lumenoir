# SICE evaluation protocol

## Verified facts

- The local `SICE_Paired` test images are **5997 × 3999** pixels
  (23.98 MP), RGB JPEG.
- The current evaluator does **not** resize images. It loads the supplied
  image as RGB float32 in `[0, 1]` and passes it directly to the metric
  implementations.
- The evaluation runner processes images one at a time. It does not batch
  images in memory.
- A combined LOL-v1 + SICE original-mode smoke run timed out after 300 seconds.
  No SICE metric result is claimed.
- SICE has **302 train samples** and **58 test samples** in the local
  Hugging Face layout. Only the 58 test samples enter evaluation.

## Memory estimates per SICE image

| Representation | Approximate size |
| --- | ---: |
| uint8 RGB | 68.6 MiB |
| float32 RGB | 274.5 MiB |
| float64 RGB | 548.9 MiB |
| float32 LAB | 274.5 MiB |
| float64 LAB | 548.9 MiB |

PSNR, SSIM, CIEDE2000, LAB conversion, edge maps and intermediate copies can
hold more than one full-resolution array at once. Plan for **at least 2 GB of
free memory per worker** and do not run SICE concurrently with other large
images. The 58 test images must be processed sequentially.

## Recommended protocols

### A. Native-resolution primary protocol (recommended for paired fidelity)

Evaluate all 58 SICE test pairs at their original 5997 × 3999 resolution:

- PSNR: native resolution.
- SSIM: native resolution.
- CIEDE2000: native resolution.
- LPIPS: the current implementation passes the full tensor to the model. If it
  runs out of memory, do not silently resize; use Protocol B and label it.
- No metric is computed on a resized image under this protocol.
- Run SICE in a separate process/batch from LOL/NPE/LIME/MEF/VV.
- Set `--latency-runs 0` for the SICE batch to avoid repeated full-resolution
  pipeline runs.

### B. Explicit standardized-resolution protocol

If native LPIPS or NIQE cannot run within the available GPU/CPU memory:

- Resize **both** low-light input and ground truth with the same documented
  interpolation and aspect-ratio-preserving rule (for example, longest edge
  1200 pixels).
- Record the resolution in the report and label the results `SICE@1200`.
- Do not compare `SICE@1200` LPIPS/NIQE with native-resolution results from
  other datasets or methods.
- This protocol is not silently applied by the current runner. It requires an
  explicit pre-resize step or a future documented configuration option.

## Do not

- Do not resize only the prediction or only the reference.
- Do not mix native and standardized SICE values in one table.
- Do not use training SICE samples as test samples.
- Do not batch all 58 full-resolution images into one process.

## Command

```bash
python -m training.kaggle_eval \
  --input-root /kaggle/input/datasets/chiranth25x/lumenoir-low-light-datasets \
  --additional-root /kaggle/working \
  --output-dir /kaggle/working/lumenoir-evaluation-sice \
  --datasets sice_p1 \
  --modes original,classical \
  --skip-metrics lpips,niqe \
  --latency-runs 0 \
  --device cpu
```

This command evaluates only SICE test records. It is still expected to be slow
because the metrics run at native resolution.
