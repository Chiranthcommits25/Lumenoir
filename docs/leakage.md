# Leakage check

The existing leakage implementation uses:

1. 64-bit pHash on every training/validation/test image.
2. A Hamming-distance threshold of 8.
3. A 64 × 64 grayscale SSIM confirmation; SSIM > 0.9 means leakage.

Test images are never dropped by the rule; a leaking training/validation image
is the candidate for exclusion. The evaluation runner now discovers train and
test records separately and evaluates only test records, but it does **not**
silently apply the pHash/SSIM exclusion. That requires approval because it
changes the evaluation set.

## Command

```bash
# 1. Write deterministic manifests (read/write only to the output directory)
python -m training.kaggle_eval --discover-only \
  --input-root datasets/raw \
  --additional-root datasets/raw/additional_test_datasets/data \
  --output-dir /tmp/lumenoir-final-manifest

# 2. Read those manifests and check pHash/SSIM leakage; no splits are written
python -m training.kaggle_eval.leakage_check \
  --manifest /tmp/lumenoir-final-manifest/manifests/manifest.json \
  --output /tmp/lumenoir-final-leakage.json
```

On Kaggle, point `--manifest` at
`/kaggle/working/lumenoir-evaluation/manifests/manifest.json` or the equivalent
evaluation output.

## Actual local result

Executed on the local manifests:

```text
checked=2649 duplicates=118 excluded=118 path_overlap=0
```

- **Images checked:** 2,649 train/validation/test paired images.
- **Duplicate pairs found:** 118.
- **Images flagged for exclusion:** 118 training/validation-side candidates.
- **Exact path overlap:** 0.
- **Report:** `/tmp/lumenoir-final-leakage.json`
- **Manifest:** `/tmp/lumenoir-final-manifest/manifests/manifest.json`
- **No split files were generated or overwritten.**

The matches were concentrated in LOL-v1 train versus LOL-v1 eval15 and
LOL-v2-Real test, including cross-dataset matches. This is a **material
leakage finding** and must be resolved/approved before reporting a clean
held-out result. The runner does not remove these test images automatically.

## What the report contains

- `images_checked`
- `training_validation_images`
- `test_images`
- `duplicate_pairs_found`
- `images_excluded` (candidates, not applied)
- `train_test_overlap` (exact path overlap)
- `matches` with kept test ID, dropped train/validation ID, Hamming distance and
  SSIM
- `hash_failures`
- manifest and report paths

This command reads images and writes only the requested JSON report.
