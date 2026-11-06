# LumenFuse

`backend/fusion/lumen_fuse.py` implements the noise-aware, Laplacian-pyramid
fusion from `final_decision.md` §4. It is wired into the API pipeline as the
`fusion` stage.

## Inputs

Three float32 RGB candidates in [0, 1] with identical dimensions:

1. `I0` — the original low-light image;
2. `C` — the LumenRetinex output (pre-denoise);
3. `D` — the Zero-DCE++ output (pre-denoise).

LumenScan information is required. If `scan_result` is omitted, `scan(I0)` is
computed; it supplies `sigma`, the full-resolution `noise_map` (`σ_loc`), and
the fusion exponents/`tau`.

## Weights

Per pixel and candidate `k`:

```
W_k = (Contrast_k + ε)^wc · (Sat_k + ε)^ws · (Expo_k + ε)^we · Noise_k^wn
```

- `Contrast = |Laplacian(GaussianBlur(Y_k, σ=1), ksize=3)|`
- `Sat = std(R_k, G_k, B_k)` across the three channels
- `Expo = exp(−(Y_k − 0.5)² / (2·0.2²))`
- `Noise = 1 / (1 + (Ĝ_k·σ_loc/τ)²)`
- `Ĝ_k = clip(blur(Y_k,5)/(blur(Y_0,5)+0.02), 1, 30)`

`(wc, ws, we, wn, τ)` come from LumenScan's fusion bucket. The original
candidate is multiplied by 3 where `max_c I0 > 0.85` (highlight protection).

Normalization follows the specification exactly:

```
Ŵ_k = (W_k + 1e-6) / Σ_j (W_j + 1e-6)
```

Each map is then refined with `guided_filter(guide=Y0, r=8, eps=1e-2)`,
clipped to non-negative values and renormalized. If a pixel's refined weights
sum to zero, it falls back to uniform `1/3` weights.

## Pyramid fusion

```
L = min(5, floor(log2(min(H, W))) - 3)
```

`L` is the top pyramid index, so `L + 1` levels are used. Weight maps are
Gaussian pyramids; candidates are Laplacian pyramids; each level is combined as
`Σ_k GaussianPyr_l(W_k) · LaplacianPyr_l(candidate_k)` and collapsed with
explicit destination sizes. The result is clipped to [0, 1] and has the exact
input dimensions. OpenCV's `createMergeMertens` is not used.

## Usage

```python
from backend.classical_cv.scan import scan
from backend.fusion.lumen_fuse import fuse

scan_result = scan(original)
result = fuse(original, classical, deep_learning, scan_result=scan_result)
result.image, result.weights, result.levels
```

`compute_weights` is public for diagnostics; `pyramid_levels`,
`gaussian_pyramid`, `laplacian_pyramid`, `collapse_pyramid` and
`normalize_weights` are testable helpers.

## Tests

`tests/test_lumen_fuse.py` covers: normalization and zero-weight fallback,
identical-candidate reconstruction, well-exposed preference, dark-region
enhancement, noise penalty, non-uniform regional selection, highlight
protection, colour-distorted candidates, extremely dark/noisy and all-black
stability, halo rate at a strong edge, odd dimensions, pyramid round-trip and
level count, invalid candidates/configurations and mismatched LumenScan data.

## Not verified / limitations

- No real photographs, dataset or end-to-end pipeline; tests are synthetic.
- No PSNR/SSIM/LPIPS/NIQE comparison; fusion quality is unmeasured.
- No latency or memory benchmark.
- `blur(...,5)` is implemented as a 5×5 kernel, matching EdgeGuard's
  interpretation; the spec does not distinguish kernel size from radius.
- `L` is interpreted as the top pyramid index (L+1 levels), matching the
  pseudocode's `0..L` loop.
- In very flat regions the contrast and saturation factors are near zero, so the
  `+1e-6` normalization term makes the weights close to uniform. This follows
  the specified formula, but it means exposure differences have little effect in
  perfectly smooth areas.
- Weight exponents and `tau` are the untuned LumenScan starting values.
- The noise term penalises candidates by effective gain, not by measured
  candidate noise; a clean and a noisy candidate with the same gain receive the
  same penalty (as specified).
- The API pipeline applies EdgeGuard, HueLock and the adaptive alpha blend after
  fusion. The optional closed-loop passes are not implemented.
- No quality comparison has been run, so the fusion contribution is unmeasured.
