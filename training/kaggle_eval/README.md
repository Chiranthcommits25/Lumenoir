# Kaggle evaluation

This package evaluates the already-implemented LUMENOIR pipeline on datasets
mounted under `/kaggle/input` and the extracted additional datasets under
`/kaggle/working/lumenoir_additional/extracted`. It does **not** reimplement
enhancement, tune parameters, fine-tune the model or download datasets.

## Quick start

1. Attach the paired datasets under `/kaggle/input` and extract the additional
   unpaired datasets under `/kaggle/working/lumenoir_additional/extracted/`.
2. Attach the LUMENOIR repository (or clone it into `/kaggle/working/LUMENOIR`)
   and the Zero-DCE++ checkpoint (`zero_dce_plus_plus.pth` or `Epoch99.pth`).
3. Install the evaluation dependencies if they are not already present:

   ```bash
   pip install -r requirements-eval.txt
   pip install -r requirements-metrics.txt   # optional: LPIPS, NIQE, torch
   ```

4. Inspect the discovered layout before running:

   ```bash
   python -m training.kaggle_eval --discover-only \
     --input-root /kaggle/input \
     --additional-root /kaggle/working/lumenoir_additional/extracted \
     --output-dir /kaggle/working/lumenoir-evaluation
   ```

5. Copy `example_config.json`, edit the roots/pairs/weights, and run:

   ```bash
   python -m training.kaggle_eval \
     --config /kaggle/working/eval_config.json
   ```

The example notebook (`kaggle_notebook.ipynb`) performs the same steps.

## Dataset discovery and explicit configuration

Auto-discovery searches `/kaggle/input` first, then every configured additional
root. It proposes roots and low/high pair directories; it is a convenience, not
a guarantee:

- Roots are found under each search root and one nested level.
- LOL-v1 prefers `eval15`/test; LOL-v2 prefers `Test`; training splits
  (`our485`, `Train`) are excluded by default.
- Pairs are matched by filename stem. A missing reference is reported and the
  image is excluded; paired metrics are never computed without a reference.
- SICE Part 1 is paired only when valid low/high directories exist. It is
  Zero-DCE++ pre-training data, so it is not a held-out test set.
- Unpaired datasets are read recursively, filtered by image extension.
- JPG, JPEG, PNG and BMP are supported. Other known image formats (for example
  WebP, GIF or HEIC) are not silently dropped: discovery records a warning.

### Additional unpaired datasets

The extracted datasets are handled as four logical unpaired datasets; the
seven source folders are retained as subgroup information where applicable:

| Logical dataset | Source path | Images | Format |
| --- | --- | ---: | --- |
| `lime` | `/kaggle/working/lumenoir_additional/extracted/LIME/data/LIME/` | 10 | BMP |
| `mef` | `/kaggle/working/lumenoir_additional/extracted/MEF/data/MEF/` | 17 | PNG |
| `npe` | `/kaggle/working/lumenoir_additional/extracted/NPE/data/NPE/` | 8 | mixed |
| `npe` (subgroup `npe_ex1`) | `/kaggle/working/lumenoir_additional/extracted/NPE-ex1/data/NPE-ex1/` | 29 | mixed |
| `npe` (subgroup `npe_ex2`) | `/kaggle/working/lumenoir_additional/extracted/NPE-ex2/data/NPE-ex2/` | 18 | mixed |
| `npe` (subgroup `npe_ex3`) | `/kaggle/working/lumenoir_additional/extracted/NPE-ex3/data/NPE-ex3/` | 30 | mixed |
| `vv` | `/kaggle/working/lumenoir_additional/extracted/VV/data/VV/` | 24 | mixed |

NPE total: **85** images. NPE and its three extensions produce one `npe`
manifest and one set of dataset-level aggregates; every record retains its
`subgroup` (`npe`, `npe_ex1`, `npe_ex2`, `npe_ex3`), and the manifest contains
per-subgroup counts. These four logical datasets are unpaired, so PSNR, SSIM
and LPIPS are never calculated for them.

If auto-discovery is ambiguous, set the dataset explicitly. The JSON structure
is:

```json
{
  "additional_roots": ["/kaggle/working/lumenoir_additional/extracted"],
  "datasets": {
    "lol_v1": {
      "root": "/kaggle/input/lol-v1",
      "pairs": [{"low": "eval15/low", "high": "eval15/high", "split": "test"}]
    },
    "npe": {"root": "/kaggle/working/lumenoir_additional/extracted"},
    "exdark": {"root": "/kaggle/input/exdark", "images": [""]}
  }
}
```

Pointing `npe.root` at the extracted parent groups `NPE`, `NPE-ex1`,
`NPE-ex2` and `NPE-ex3`. Pointing it at one subgroup evaluates only that
subgroup while keeping the `npe` dataset key.

`allow_training_split` is intentionally false: explicit training paths are
skipped unless you deliberately set it to true (and the report records that the
results are not held out).

## Evaluation modes

| Mode | Description |
| --- | --- |
| `original` | The low-light input, evaluated directly (baseline). |
| `classical` | LumenScan + LumenRetinex + EdgeGuard + HueLock + alpha blend. |
| `deep_learning` | LumenScan + Zero-DCE++ + EdgeGuard + HueLock + alpha blend. |
| `hybrid` | Full pipeline. |

All modes use the existing `EnhancementPipeline`; no enhancement code is
duplicated. `deep_learning` and `hybrid` require a ready checkpoint and PyTorch.
If the model is unavailable and `allow_missing_model` is true, those modes are
skipped and logged in `failures/` and `run_config.json`.

## Metrics

Paired datasets (LOL-v1, LOL-v2 Real/Synthetic, SICE P1 with valid pairs):

- PSNR, SSIM, LPIPS.

Unpaired datasets (ExDark, DICM, NPE including NPE-ex1/2/3, LIME, MEF, VV):

- NIQE, LOE, relative noise, edge preservation, texture retention, hue shift,
  new clipping, highlight push, black-level crushing, halo rate.

Definitions and assumptions:

- **PSNR/SSIM** use the evaluation framework (`skimage`, `data_range=1.0`).
- **LPIPS** uses `lpips` with AlexNet; its pretrained weights may download on
  first use. Use `skip_metrics: ["lpips"]` if the environment is offline.
- **NIQE** uses `pyiqa`; compare relative values from this same code, not
  MATLAB numbers.
- **LOE** is the NPE-paper lightness-order error on 50×50 max-RGB maps: the
  fraction of valid pixel pairs whose relative lightness order changes. It
  requires the low-light input and is therefore reported for unpaired sets.
- **Noise** is `σ̂_flat / mean(Y_out[flat])` (relative noise) from the Immerkaer
  estimator.
- **Edge preservation** is the Pearson correlation of gradient magnitudes on the
  strongest 10% of edges; unpaired comparisons use the input.
- **Colour distortion** for unpaired sets is the mean wrapped hue difference on
  pixels with `C* > 10`; paired sets can additionally use CIEDE2000 in the
  framework.
- **New clipping** is `frac(out max-channel ≥ 0.98) − frac(in max-channel ≥ 0.98)`.
- **Black-level crushing** is `frac(out max-channel ≤ 0.02) − frac(in max-channel ≤ 0.02)`.
- **Halo rate** is the gradient-reversal rate on the strongest 5% of input
  edges.

Missing optional dependencies or a failed metric produce a `null` value with a
note in the per-image JSONL record, never a fabricated number.

## Outputs

Everything is written under the configured output directory, default
`/kaggle/working/lumenoir-evaluation`:

```
manifests/
  manifest.json                 # all datasets, sorted and deterministic
  <dataset>.json
  discovery_report.json         # image counts, format counts, subgroup counts
records/
  records.jsonl                 # image-level metric records + subgroup + notes + timings
  records.csv
reports/<dataset>/<mode>/
  image_metrics.json|.csv|.md   # per-image rows, summary, worst five
aggregate.json|.csv|.md         # dataset/mode/metric aggregates
latency.json|.csv|.md           # per-stage latency
failures/
  failures.jsonl|.csv
visuals/<dataset>/<image>_<mode>.png
device.json
run_config.json                 # config, seed, git commit, device, model status
```

Every manifest entry contains absolute image paths, `record_count`,
`image_count`, `format_counts`, `subgroup_counts` and the source `roots`.
Visuals are side-by-side low-light/input and output images, capped by
`visual_limit` per dataset.

## Reproducibility

- Manifest records are sorted by relative path; reruns are deterministic.
- `max_images` sampling uses `random.Random(seed)` and is recorded in the
  manifest warnings.
- `seed`, config, additional roots and git commit are in `run_config.json`.
- Per-image records and per-stage timings are retained, so aggregates can be
  recomputed.
- No parameter is tuned on evaluation data.
- Counts are never inferred from filenames: manifests report the discovered
  image count, per-format counts and NPE subgroup counts. Unsupported image
  formats, empty directories and missing NPE subgroups are warned about rather
  than silently ignored.

## Assumptions and limitations

- No dataset is downloaded. Paths must already exist under `/kaggle/input`,
  `/kaggle/working/lumenoir_additional/extracted`, or be supplied explicitly.
- Unpaired datasets (including LIME, MEF, NPE/NPE-ex1/2/3 and VV) have no ground
  truth, so PSNR/SSIM/LPIPS are not reported.
- SICE Part 1 is not held out for Zero-DCE++ (the model was pre-trained on it);
  paired SICE results must be labelled accordingly.
- LOL training splits are excluded. Explicit training paths require
  `allow_training_split` and are flagged.
- Paired images must have identical dimensions; mismatches are logged as
  failures rather than resized silently.
- LPIPS/NIQE require optional packages and model files; on a network-isolated
  Kaggle run, skip them or pre-cache the model files.
- Metrics are only as comparable as their library versions; keep the same
  environment across methods.
- No quality improvement claim should be made until the reports are populated
  with real datasets.
- GPU speeds up Zero-DCE++ only; LumenScan, LumenRetinex, LumenFuse, EdgeGuard
  and HueLock are CPU NumPy/OpenCV work.
