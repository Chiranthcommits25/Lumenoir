# API reference

Base prefix: `/api`. Interactive docs: `http://localhost:8000/docs`.

Upload validation, the full enhancement pipeline, status reporting and metric
computation are implemented. `POST /api/enhance` returns an enhanced PNG.
Classical mode works without a model; deep-learning and hybrid modes return
`503` until a Zero-DCE++ checkpoint is loaded. Metric computation is available
through the evaluation harness; the metrics endpoint reports the registry.

## `GET /api/health`

```json
{
  "status": "ok",
  "service": "LUMENOIR API",
  "version": "0.2.0",
  "enhancement_pipeline": "implemented",
  "model_state": "not_selected",
  "model_ready": false
}
```

## `GET /api/images/limits`

Reports the upload constraints so the UI can display real limits.

```json
{
  "max_size_bytes": 10485760,
  "allowed_extensions": [".jpeg", ".jpg", ".png"],
  "allowed_mime_types": ["image/jpeg", "image/png"],
  "dimensions": {
    "min_width": 64,
    "min_height": 64,
    "max_width": 8000,
    "max_height": 8000,
    "max_megapixels": 40.0
  }
}
```

## `POST /api/images/upload`

`multipart/form-data` with a single `file` field (JPG, JPEG or PNG).
Validated in memory and discarded; nothing is persisted.

Success — `200`:

```json
{
  "ok": true,
  "message": "Image validated successfully.",
  "image": {
    "filename": "night.png",
    "content_type": "image/png",
    "format": "PNG",
    "width": 320,
    "height": 240,
    "size_bytes": 4096,
    "colour_mode": "RGB",
    "megapixels": 0.08,
    "aspect_ratio": 1.3333,
    "statistics": {
      "luminance": { "mean": 12.4, "min": 9, "max": 18, "stddev": 2.1 },
      "red": { "mean": 12.0, "min": 9, "max": 18, "stddev": 2.0 },
      "green": { "mean": 14.0, "min": 11, "max": 20, "stddev": 2.2 },
      "blue": { "mean": 18.0, "min": 14, "max": 24, "stddev": 2.3 },
      "luminance_histogram": [76800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      "histogram_bins": 16
    }
  }
}
```

`statistics` are descriptive measurements (0-255 scale) plus a 16-bin luminance
histogram. They are **not** an illumination or quality assessment.

## `GET /api/enhance/status`

Reports pipeline, model and mode availability.

```json
{
  "ready": true,
  "stages": [
    {
      "name": "analysis",
      "status": "implemented",
      "description": "Measure illumination, dynamic range, colour cast and noise with LumenScan."
    }
  ],
  "modes": [
    {
      "mode": "classical",
      "label": "Classical CV",
      "available": true,
      "required_stages": ["analysis", "classical_cv", "denoising", "colour_correction", "quality_assessment"]
    },
    {
      "mode": "hybrid",
      "label": "Hybrid",
      "available": false,
      "required_stages": ["analysis", "classical_cv", "deep_learning", "fusion", "denoising", "colour_correction", "quality_assessment"]
    }
  ],
  "model": {
    "state": "not_selected",
    "model_id": "",
    "device": "cpu",
    "weights_path": null,
    "message": "No enhancement model is selected…",
    "ready": false,
    "load_duration_ms": null
  },
  "inference": { "device": "cpu", "max_edge": 1600, "max_inference_pixels": 4000000 },
  "artifacts": [
    {
      "kind": "illumination_map",
      "label": "Illumination map",
      "description": "Estimated spatial illumination from LumenScan.",
      "available": false,
      "reason": "This output was not produced for this image.",
      "width": null,
      "height": null,
      "encoding": null,
      "media_type": null,
      "data": null
    }
  ]
}
```

All seven stages are `implemented`. Mode availability also depends on the
model: `classical` is ready without one; `deep_learning` and `hybrid` require
`model.ready = true`. Before a request, artifacts are unavailable with a
reason; during a request, artifacts produced by that run carry base64 PNG data.

Stages: `analysis`, `classical_cv`, `deep_learning`, `fusion`, `denoising`,
`colour_correction`, `quality_assessment`.

Model states: `not_selected`, `weights_missing`, `not_loaded`, `ready`,
`failed`.

## `POST /api/enhance`

`multipart/form-data`: `file` (JPG/JPEG/PNG) and optional `mode`
(`classical`, `deep_learning`, `hybrid`; default `hybrid`).

The pipeline runs as: LumenScan → LumenRetinex ‖ Zero-DCE++ → LumenFuse →
EdgeGuard → HueLock → adaptive α blend. The do-no-harm bypass returns the
original image and skips the model and enhancement stages. After upload
validation, the endpoint enforces `LUMENOIR_INFERENCE_MAX_PIXELS` (default
4 MP) and returns `422 image_too_large` above it.

Success — `200`:

```json
{
  "ok": true,
  "mode": "hybrid",
  "processing_status": "completed",
  "processing_time_ms": 812.4,
  "image": { "...": "input image information + statistics" },
  "pipeline": {
    "mode": "hybrid",
    "processing_status": "completed",
    "processing_time_ms": 812.4,
    "timings_ms": {
      "analysis": 16.1,
      "classical_cv": 24.4,
      "deep_learning": 9.2,
      "fusion": 123.3,
      "denoising": 19.7,
      "colour_correction": 6.9,
      "quality_assessment": 2.2
    },
    "stages": [
      { "name": "analysis", "status": "implemented", "duration_ms": 16.1, "detail": null }
    ],
    "artifacts": [
      {
        "kind": "final_output",
        "label": "Final enhanced image",
        "available": true,
        "width": 320,
        "height": 240,
        "encoding": "base64",
        "media_type": "image/png",
        "data": "<base64 PNG>"
      }
    ]
  },
  "output": {
    "encoding": "base64",
    "media_type": "image/png",
    "filename": "lumenoir-night-hybrid.png",
    "data": "<base64 PNG>"
  }
}
```

The output has the exact input dimensions and is RGB. Available artifacts carry
real `data`, `width`, `height` and `media_type`; artifacts not produced for the
request carry a `reason`. Results are in-memory only — the frontend converts
them to a download.

Mode behaviour:

| Mode | Model required | Stages |
| --- | --- | --- |
| `classical` | no | analysis, classical_cv, denoising, colour_correction, quality_assessment |
| `deep_learning` | yes | analysis, deep_learning, denoising, colour_correction, quality_assessment |
| `hybrid` | yes | all seven stages |

Without a ready model, `deep_learning` and `hybrid` return `503` with the typed
error envelope (`model_not_available`, `weights_missing`, `torch_missing` or
`device_unavailable`). A stage exception returns `500`; invalid uploads keep
their existing `4xx` codes.

Model example — `503`:

```json
{
  "ok": false,
  "error": {
    "code": "model_not_available",
    "message": "No enhancement model is selected. Set LUMENOIR_MODEL_ID=zero_dce_plus_plus to enable deep-learning and hybrid modes."
  }
}
```

## `GET /api/evaluation/metrics`

Reports the implemented metrics, whether each needs a ground-truth reference
and which dependency it uses:

```json
{
  "status": "available",
  "metrics": [
    {
      "name": "psnr",
      "label": "PSNR",
      "status": "implemented",
      "requires_reference": true,
      "requires_dependency": "scikit-image",
      "description": "Full-reference fidelity against the ground truth, with a GT-mean-aligned variant reported alongside the raw value."
    }
  ]
}
```

Metrics: `psnr`, `ssim`, `lpips`, `niqe`, `latency`, `noise_suppression`,
`edge_preservation`, `colour_distortion`, `overexposure`, `halo`, `loe`,
`black_crushing`.

The endpoint only reports the registry; it does not compute numbers. Use
`python -m backend.evaluation.cli` for reports. Metric definitions, the harness
and the integration-test split are documented in
[`evaluation.md`](evaluation.md).

## Error envelope

Every rejection uses the same shape:

```json
{ "ok": false, "error": { "code": "file_too_large", "message": "…" } }
```

### Upload / request error codes

| HTTP | `code` | Cause |
| --- | --- | --- |
| 400 | `missing_filename` | No filename supplied |
| 400 | `empty_file` | Zero bytes |
| 400 | `corrupted_image` | Truncated or unreadable image data |
| 400 | `unsupported_mode` | Mode is not classical / deep_learning / hybrid |
| 413 | `file_too_large` | Body larger than the byte limit |
| 415 | `unsupported_extension` | Extension not in `.jpg`, `.jpeg`, `.png` |
| 415 | `unsupported_media_type` | Declared content type not JPEG/PNG |
| 415 | `unsupported_format` | Payload is not a decodable JPEG/PNG image |
| 415 | `format_mismatch` | Extension disagrees with the detected format |
| 422 | `image_too_small` | Below the minimum dimensions |
| 422 | `image_too_large` | Above the maximum dimensions, megapixels or inference pixel limit |
| 422 | `invalid_dimensions` | Non-positive width or height |

### Inference / pipeline error codes

| HTTP | `code` | Cause |
| --- | --- | --- |
| 500 | `unknown_model` | Configured model id has no registered implementation |
| 500 | `checkpoint_invalid` | Checkpoint missing, unreadable, or not the Zero-DCE++ architecture |
| 500 | `inference_failed` | The model raised or returned an unusable image |
| 500 | `enhancement_failed` | A pipeline stage failed |
| 503 | `weights_missing` | Model selected, checkpoint absent |
| 503 | `model_not_available` | Inference attempted with no ready model |
| 503 | `torch_missing` | Model selected but PyTorch is not installed |
| 503 | `device_unavailable` | Explicit device (mps/cuda) is unavailable or the index is invalid |
| 422 | *(FastAPI)* | `file` field missing from the request |
