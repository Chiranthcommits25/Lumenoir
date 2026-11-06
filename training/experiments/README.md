# Stage attribution experiment

This experiment measures the actual intermediate outputs of the existing
`hybrid` pipeline. It adds no enhancement logic, changes no weights or
thresholds, and never uses test data.

## Validation data requirement

The repository includes a materialized paired validation split at
`training/splits/validation.json` (391 records across LOL-v1, LOL-v2 Real and
LOL-v2 Synthetic). The experiment still refuses to run without an explicit
manifest, never creates a replacement split, and never uses the 215 test images.

Manifest paths are dataset-relative. Resolve them against `--data-root`
(default: `datasets/raw`) so the same manifest works locally and on Kaggle.
A validation manifest may use either form:

```json
{
  "pool": "validation",
  "data_root": "datasets/raw",
  "records": [
    {
      "dataset": "lol_v1",
      "image_id": "our485/low/10.png",
      "low": "LOL-v1/our485/low/10.png",
      "high": "LOL-v1/our485/high/10.png",
      "split": "validation",
      "kind": "paired"
    }
  ]
}
```

or a discovery-style manifest:

```json
{
  "datasets": [
    {
      "dataset": "lol_v2_real",
      "records": [
        {
          "image_id": "validation/0001.png",
          "low": "/absolute/path/low.png",
          "high": "/absolute/path/high.png",
          "split": "validation"
        }
      ]
    }
  ]
}
```

Every record must be `split: validation` (or come from a `pool: validation`
split file). Test and training records are rejected.

## Command

```bash
python -m training.experiments.stage_attribution \
  --validation-manifest /kaggle/working/LUMENOIR/training/splits/validation.json \
  --data-root /kaggle/input/datasets/chiranth25x/lumenoir-low-light-datasets \
  --output-dir /kaggle/working/lumenoir-stage-attribution-YYYYMMDD \
  --weights /kaggle/working/zero_dce_plus_plus.pth \
  --device cuda \
  --metrics psnr,ssim,lpips,delta_e00
```

Dry-run validation only:

```bash
python -m training.experiments.stage_attribution \
  --validation-manifest training/splits/validation.json \
  --data-root datasets/raw \
  --output-dir /tmp/unused \
  --dry-run
```

## Captured stages

The production pipeline is run with `capture_diagnostics=True`:

1. `original`
2. `raw_zero_dce` — `DEEP_LEARNING_OUTPUT` before EdgeGuard/HueLock
3. `classical`
4. `fusion` — raw LumenFuse output
5. `fusion_edgeguard` — actual EdgeGuard output
6. `fusion_edgeguard_huelock` — actual HueLock output
7. `final` — actual final alpha-blended output
8. `ground_truth`

Diagnostic capture only copies existing stage results; it does not recompute or
approximate them.

## Outputs

```text
stages/<dataset>/<image>/<stage>.png
visuals/<dataset>/<image>_grid.png
per_image_metrics.csv
stage_aggregate.csv
dataset_comparison.csv
fusion_weights.csv
stage_latency.csv
stage_latency_aggregate.csv
memory.csv
failures.csv
summary.md
run_config.json
```

The experiment refuses to write into a non-empty output directory. Use a new
directory for every run.

## Metrics

- PSNR
- SSIM
- LPIPS (optional dependency)
- CIEDE2000 (`delta_e00`)

All stages use the same low-light input, ground truth, dimensions and metric
implementation.

## Restrictions

- No fusion-weight redesign.
- No threshold/parameter tuning.
- No EdgeGuard/HueLock changes.
- No model changes or retraining.
- No test-data use.
- No validation-split creation.
