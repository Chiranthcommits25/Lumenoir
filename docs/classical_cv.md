# Classical CV primitives, LumenScan, LumenRetinex, EdgeGuard and HueLock

Status: the primitives, LumenScan, LumenRetinex, EdgeGuard and HueLock are
implemented, unit-tested and wired into the API pipeline. `POST /api/enhance`
returns images; classical mode works without a model, while deep-learning and
hybrid modes require a ready Zero-DCE++ model.

All functions work on float32 RGB in [0, 1] (sRGB, no linearisation) or on
single-channel float32 arrays. `Y = 0.299R + 0.587G + 0.114B`.

## Primitives

| Module | Function | Definition |
| --- | --- | --- |
| `filters.py` | `box_filter(image, r)` | `cv2.boxFilter`, `(2r+1)²`, reflected borders |
| `filters.py` | `guided_filter(I, p, r, eps)` | `a = (box(Ip) − box(I)box(p)) / (box(I²) − box(I)² + eps)`, `b = box(p) − a·box(I)`, `q = box(a)I + box(b)` |
| `noise.py` | `noise_sigma(gray)` | Immerkaer: `σ = √(π/2)/6 · mean|Y*N|` over the flattest 50% by Sobel |
| `noise.py` | `noise_map(gray, r=7, sigma=…)` | local σ, optionally clipped to `[0.5σ̂, 2σ̂]` |
| `illumination.py` | `estimate_illumination(rgb)` | `T0 = max_c I_c`, guided filter (`guide=Y`, `input=T0`, half resolution), clipped to `[1e-3, 1]` |

Implementation notes:

- The illumination radius defaults to `round(0.03 · min(H, W))`, scaled to the
  working resolution. An explicit `radius` is interpreted in working-resolution
  pixels.
- The illumination map is returned at the resolution it is computed on. The
  guided filter is the only smoothing step; `T ≥ T0` is deliberately not
  enforced.
- `backend/evaluation/noise.py` now re-exports this implementation, so metrics
  and the pipeline share one Immerkaer definition.

## LumenScan features

Features are measured on a copy whose longest edge is at most **256 px**, except
noise, which uses the full-resolution luminance (`final_decision.md` §5.1).

| Feature | Definition |
| --- | --- |
| `mean` | `mean(Y)` |
| `p5`, `p50`, `p95` | `percentile(Y, [5, 50, 95])` |
| `dynamic_range` | `p95 − p5` |
| `nonuniformity` | `std(log2(T + 1/64))`, in stops; `<1` uniform, `1–2` moderate, `>2` strong |
| `noise_sigma` | Immerkaer on the full-resolution `Y` |
| `colour_cast_a`, `colour_cast_b`, `colour_cast_magnitude` | mean LAB `(a*, b*)` on mid-tone, low-chroma pixels of a brightened copy (`C* < 30`, `25 ≤ L* ≤ 75`) |
| `highlight_fraction` | `mean(max_c I ≥ 0.98)` |
| `black_fraction` | `mean(max_c I ≤ 0.02)` |
| `darkness` | `extreme < 0.06`, `dark < 0.20`, `dim < 0.45`, `normal ≥ 0.45` (by `p50`) |

Colour-cast detection deliberately ignores pixels with `C* ≥ 30`, so a
uniformly saturated colour (for example pure orange) produces no cast estimate.
That is the specified behaviour; the guards for genuinely coloured light belong
to HueLock.

## Parameters selected

`scan(image)` returns `LumenParameters` implementing §5.3 exactly:

| Parameter | Rule |
| --- | --- |
| `gamma` | `clip(ln(0.45)/ln(p50), 0.30, 1)`; `1` when `p50 ≥ 0.45`; `0.30` when `p50 ≈ 0` |
| `gain_cap` | `clip(0.06/σ̂, 3, 10)`; `10` when `σ̂ = 0` |
| `guided_eps` | `1e-3`, or `3e-3` when `σ̂ > 0.02` |
| `clahe_clip` | `clip(1 + 3·lowcon − 40·σ̂, 1, 4)`, `lowcon = clip((0.45 − DR)/0.45, 0, 1)` |
| `clahe_tiles` | `(8, 8)` up to 1 MP, `(12, 12)` above |
| `clahe_blend` | `0.6` |
| `dl_strength` | `clip(1 − 0.6·(p50 − 0.15)/0.35, 0.4, 1)` |
| `fusion_*`, `fusion_tau` | `σ̂ < 0.008`: `(1, 0.5, 1, 1)`, τ 0.08; `< 0.02`: `(0.7, 0.5, 1, 1.5)`, τ 0.08; else `(0.5, 0.5, 1, 2)`, τ 0.06 |
| `hue_lock_beta` | `clip((c − 6)/20, 0, 0.5)`; the colour-spread guards are not applied here |
| `output_blend_alpha` | `1 − smoothstep(0.25, 0.50, p50)` |
| `bypass_gate` | `p50 ≥ 0.45 and DR ≥ 0.45 and s_hi < 0.01` |

The denoise-strength table entry from §5.3 is not a scalar: it depends on the
effective gain computed after enhancement, so it belongs to EdgeGuard (later).
The closed-loop proxies (§5.7) and Optuna tuning (§5.6) are also not
implemented.

## Usage

```python
import numpy as np
from backend.classical_cv.scan import scan

rgb = np.asarray(image, dtype=np.float32) / 255.0
result = scan(rgb)

result.features.as_dict()
result.parameters.as_dict()
result.illumination  # analysis-resolution map used for non-uniformity
result.noise_map     # full-resolution local sigma
```

## LumenRetinex

`backend/classical_cv/retinex.py` implements the classical enhancement branch.

Order of operations:

1. illumination `T` from `estimate_illumination` (half-resolution guided
   filter, upsampled), using `LumenScan.parameters.guided_eps`;
2. gain `G = clip(T^(γ−1), 1, G_cap)`;
3. highlight attenuation `h = 1 − smoothstep(0.65, 0.95, max_c I)`;
4. `G' = 1 + (G−1)·h`, then `G'' = min(G', 1/(max_c I + 1e-3))`;
5. the same `G''` is applied to R, G and B;
6. CLAHE on LAB `L*` with `clahe_clip`, `clahe_tiles` and the blended
   `L = 0.4·L + 0.6·CLAHE(L)`;
7. the per-channel soft-knee shoulder (`y = x` up to 0.85, tanh above);
8. final clip to [0, 1].

Notes:

- The illumination map is recomputed at the input resolution (its guided filter
  runs at half resolution) rather than upscaling LumenScan's 256 px analysis
  map. `enhance(..., illumination=...)` accepts an override.
- CLAHE uses a 16-bit `L*` buffer to reduce quantisation. Only `L*` changes;
  `a*` and `b*` are kept, then the LAB→RGB conversion is clipped and the
  shoulder is applied.
- `LumenRetinexConfig.respect_bypass_gate` is **True** by default: when
  `LumenScan` reports the §5.8 bypass gate, the input is returned unchanged.
  Set it to `False` to measure raw Retinex. The gate uses analysis-copy
  features, so a bimodal image with `p50 ≥ 0.45`, `DR ≥ 0.45` and no clipped
  highlights is bypassed even if it contains a large dark region.
- `apply_clahe=False` and `apply_shoulder=False` isolate the gain stage for
  tests and ablations.
- The output image keeps the input dimensions exactly; the result also carries
  the final gain map and illumination map (both `None` when bypassed).

```python
from backend.classical_cv.retinex import enhance
from backend.classical_cv.scan import scan

scan_result = scan(rgb)
retinex = enhance(rgb, scan_result=scan_result)
retinex.image, retinex.gain, retinex.bypassed
```

## EdgeGuard

`backend/classical_cv/edge_guard.py` implements the post-fusion denoiser from
§3.5. It expects the fused image and the original low-light image (the latter is
used only for the effective gain), plus the LumenScan noise estimate.

Luma:

1. `G = clip(blur(Y_F, 5) / (blur(Y_0, 5) + 0.02), 1, 30)`, where `Y` is
   Rec.601 luma;
2. `G_ref = P75(G)`;
3. `w = clip((G − 1.5) / (G_ref − 1.5), 0, 1)^0.8`;
4. `Y_s = guided_filter(Y_F, Y_F, r, eps)`, `eps = (1.2·σ̂·G_ref)²`;
5. `Y_out = w·Y_s + (1−w)·Y_F`.

`r` is 4 up to 1 MP and 6 above. If `G_ref <= 1.5`, no region was meaningfully
amplified: `w = 0` and the guided filter is skipped (this also avoids an
`eps = 0` case when `σ̂ = 0`).

Chroma:

- convert to YCrCb; take Cr/Cb at half resolution;
- `sigmaColor = clip(3·255·σ̂·√G_med, 8, 40)` with `G_med = median(G)`;
- `bilateralFilter(..., d=7, sigmaSpace=5)`, then upsample;
- reconstruct YCrCb with `Y_out` and the filtered chroma.

The result also carries the effective gain, luma weight, `G_ref`,
`sigmaColor` and the chosen luma radius. Inputs and output are float32 RGB in
[0,1] with identical dimensions.

Usage:

```python
from backend.classical_cv.edge_guard import denoise

result = denoise(fused_rgb, original_rgb, scan_result=scan_result)
result.image, result.effective_gain, result.luma_weight
```

## HueLock

`backend/classical_cv/hue_lock.py` implements §3.7 using LumenScan's cast and
raw beta:

- `cast = (mean a*, mean b*)`, `c = ‖cast‖`, `beta_raw` from LumenScan
  (`clip((c−6)/20, 0, 0.5)`);
- the mid-tone `a*`/`b*` spread is measured on the same brightened mid-tone
  mask; if `max(std(a*), std(b*)) > 15`, beta is halved;
- the applied shift is capped: `beta = min(beta, 8 / c)`, so
  `‖Δ(a*,b*)‖ <= 8`;
- only `a*` and `b*` are shifted. `L*` is untouched, so highlights, shadows and
  overall lightness are preserved; chroma is never boosted.

If `beta_raw` is zero (cast at or below 6) the input is returned unchanged with
no LAB round trip. The correction is a global cast shift, as specified; the
mid-tone/low-chroma rule selects the estimate rather than masking the shift.
`HueLockConfig(halve_on_spread=False)` disables the spread guard for ablation.

```python
from backend.classical_cv.hue_lock import correct

result = correct(rgb, scan_result=scan_result)
result.image, result.beta_raw, result.beta_applied, result.shift_magnitude
```

## Tests

- `tests/test_classical_filters.py`: box constants/impulses, guided identity,
  flat preservation, edge preservation versus box, invalid arguments.
- `tests/test_classical_noise.py`: zero noise on flat images, Gaussian estimate,
  flat mask, local map clipping, shared implementation with evaluation.
- `tests/test_lumen_scan.py`: normal/bright, dark, noisy and non-uniform images;
  percentiles, gamma, gain cap, guided epsilon, CLAHE rule and tiles, fusion
  buckets, blend alpha, bypass gate, clipping fractions, colour cast, shapes,
  JSON serialisation and invalid inputs.
- `tests/test_lumen_retinex.py`: the §3.3 gain formula against a manual
  calculation; dark, extremely dark, noisy and non-uniform images; gain cap and
  highlight shoulder; CLAHE contrast and hue preservation; bypass behaviour for
  well-lit and bimodal images; dimension and colour consistency; invalid inputs.
- `tests/test_edge_guard.py`: gain map and reference, luma/chroma noise
  reduction, edge and texture retention, well-lit no-op, dark-region gain,
  LumenScan sigma reuse, gain cap, large-image radius, invalid inputs.
- `tests/test_hue_lock.py`: neutral no-op, warm-cast reduction, 8-unit shift
  cap, beta provenance, spread guard (mocked and real high-spread scenes),
  lightness preservation, no chroma boost, highlight/shadow lightness, optional
  scan result, invalid inputs.

## Not verified

- Real photographs and the do-no-harm set were not available; tests use
  synthetic images only.
- No metric comparison (PSNR/SSIM/LPIPS/NIQE) has been run: there is no
  end-to-end pipeline and no dataset, so the enhancement quality is unmeasured.
- The coefficients are the untuned starting values from `final_decision.md`.
- The effect of using a 256 px analysis copy on percentile and clipping
  estimates for very small highlights is not measured against full-resolution
  statistics.
- Illumination radius scaling at reduced resolution is an engineering
  interpretation of "3% of the shorter side"; it is parameterised and not
  benchmarked.
- CLAHE runs on a 16-bit `L*` buffer; its exact agreement with an 8-bit or
  reference implementation was not compared.
- The shoulder is per-channel and applied after CLAHE; hue behaviour in very
  bright saturated regions was not measured.
- The bypass gate can skip a bimodal but genuinely non-uniform image (encoded
  by a test), because §5.8 only inspects p50, dynamic range and clipped
  highlights.
- No performance/latency benchmark for LumenRetinex, EdgeGuard or HueLock was
  run.
- EdgeGuard's `blur(...,5)` is implemented as a 5×5 kernel; the document does
  not distinguish kernel size from radius, and the radius-5 alternative was not
  compared.
- EdgeGuard denoising strength, chroma subsampling and bilateral settings were
  not tuned or benchmarked; colour bleeding was checked only on synthetic
  patches.
- HueLock applies the cast shift globally, as specified. The "std of (a*,b*)"
  guard is implemented as `max(std(a*), std(b*))`; the document does not specify
  how to combine the two, and alternatives were not compared.
- HueLock has no spatial mask beyond the mid-tone/low-chroma detection; L* is
  untouched, but a global a*/b* shift does affect low-chroma highlights and
  shadows by construction.
- Skin-tone preservation was checked only synthetically, not on portraits.
- EdgeGuard and HueLock are now applied in sequence by the API pipeline; no
  metric comparison or real-image validation has been run.
- The raw HueLock beta is produced by LumenScan, but the "std > 15" halving and
  the 8-unit shift cap are now applied by HueLock (LumenScan still reports the
  raw beta).
