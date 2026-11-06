I'll verify the dataset sizes and the repo details first, since you asked for confirmed numbers.

I checked the dataset sizes and the Zero-DCE++ facts against papers and dataset cards. Two things I could not verify are marked below, and a few of my earlier design choices have changed.

**Changes from my earlier proposal**
- I dropped the chroma pre-denoise, because denoising once after fusion is simpler and cheaper.
- Optuna and the closed loop move to the advanced tier. The MVP uses a rule-based controller.
- LOL-v2 may overlap with LOL-v1, so a dedupe step is now mandatory (section 1.4).

---

# 1. Dataset and data preparation

**1.1 and 1.2 Verified datasets and splits**
- **LOL-v1:** 500 pairs, 485 train and 15 test, 400×600.
- **LOL-v2-Real:** 789 pairs, 689 train and 100 test.
- **LOL-v2-Synthetic:** 900 train and 100 test pairs at 384×384.
- **SICE:** 589 multi-exposure sequences with 4,413 images in total. Part 1 has 360 sequences and Part 2 has 229.
- **Zero-DCE's own training split:** 3,022 Part 1 images split randomly into 2,422 for training and the rest (600) for validation, resized to 512×512.
- **Zero-DCE's SICE test protocol:** Part 2 low-light images only, 767 pairs, resized to 1200×900.
- **No-reference sets:** NPE 84, LIME 10, MEF 17, DICM 64 and VV 24 images, 199 in total.
- **ExDark:** 7,363 images across 10 lighting conditions.

**1.3 Role of each dataset**

| Role | Data | Size |
|---|---|---|
| Fine-tune NoirNet (zero-reference, GT never used) | SICE P1 train (2,422) + LOL-v1 `our485` low images + LOL-v2-Real train low images, excluding the 100 held out for validation below + synthetic extreme-dark set (below) | about 3,900 to 4,500 |
| Validation (controller tuning, early stopping) | SICE P1 val (600, zero-reference losses only) + 100 random LOL-v2-Real train pairs + 100 random LOL-v2-Syn train pairs (seed 42, excluded from fine-tuning) | 200 pairs + 600 |
| Test with GT | LOL-v1 eval15, LOL-v2-Real test, LOL-v2-Syn test, SICE P2 low-light random 200 of 767 (seed 42) | 15 / 100 / 100 / 200 |
| Unseen without GT (touched once) | DICM, LIME, MEF, NPE, VV (199 total), ExDark stratified 20 per lighting type (200), own captures (60) | about 460 |
| Do-no-harm set | 100 well-lit or over-exposed SICE images (0EV and +1EV frames) | 100 |

**1.4 Leakage prevention**
- Never fine-tune on any image from a test split.
- The LOL-v2 paper-based sources say LOL-v2 contains images from LOL-v1, so train and test scenes may overlap across versions. I could not confirm how much, so this is a hard rule:
  1. Compute a 64-bit pHash (`imagehash`) on every train and validation image and every test image.
  2. Drop any train or validation image within Hamming distance ≤ 8 of any test image.
  3. As a second pass, drop any pair with SSIM > 0.9 on 64×64 grayscale copies.
  4. Log the dropped count per dataset and put it in the report.
- Freeze all parameters before any test or unseen set is run, and keep a commit hash with the frozen config.
- Do not tune on DICM, LIME, MEF, NPE, VV, ExDark or your own images.

**1.5 Preprocessing and augmentation**
- **Pipeline:** work in float32 sRGB [0,1]. Decode with OpenCV and convert BGR to RGB. Strip EXIF orientation consistently.
- **Size:** SICE images are large, so resize the long edge to 1200 (area interpolation), then take random 384×384 crops. LOL images take random 384×384 crops (400-px height). Synthetic LOL stays at 384×384.
- **Augmentation:** flips and rot90.
- **Darkening augmentation (p = 0.5):** apply linear-domain darkening and noise (code below).
- **No colour jitter**, because the colour-fidelity target would be corrupted.
- **Inference cap:** long edge ≤ 1920 for the app.

```python
def synth_dark(img, rng):
    lin = img ** 2.2
    s = 10 ** rng.uniform(np.log10(0.03), np.log10(0.5))      # darkness scale
    x = lin * s
    sig_s = 10 ** rng.uniform(-3, -2)                          # shot-noise coeff
    sig_r = 10 ** rng.uniform(-3, np.log10(1.5e-2))            # read noise
    n = rng.normal(0, 1, x.shape) * np.sqrt(sig_s * x + sig_r**2)
    out = np.clip(x + n, 0, 1) ** (1/2.2)
    return (np.round(out * 255) / 255).astype(np.float32)      # 8-bit quantisation
```

**1.6 Exclusive unseen sets:** DICM, LIME, MEF, NPE, VV, ExDark and own captures. These are also the sets many published methods quote NIQE on, so you can compare against published numbers.

**1.7 Own images: yes, 60**
- 20 extremely dark (phone night, less than about 5 lux).
- 20 non-uniform (backlit, street lamps, mixed indoor and outdoor).
- 20 normal or bright (do-no-harm).
- For about 20 scenes, mount the phone on a tripod and take a high-ISO short exposure as input and a low-ISO long exposure as pseudo-reference. This gives you paired data you control.
- Evaluate with no-reference metrics, artifact metrics, a failure-case table, and a blind A/B test (at least 10 raters, 20 images, randomised order, "which looks better and more natural").

---

# 2. Deep learning model: NoirNet

**2.1 Decision:** Zero-DCE++ is the NoirNet backbone. Original Zero-DCE is a baseline row only.

**2.2 References**
- **Zero-DCE++ paper:** Li, Guo, Loy, TPAMI 2022, arXiv 2103.00860. The project page (li-chongyi.github.io/Proj_Zero-DCE++.html) links to the code. I believe the repo is `Li-Chongyi/Zero-DCE_extension`, but confirm from the project page.
- **Zero-DCE paper:** Guo et al., CVPR 2020, arXiv 2001.06826.

**2.3 Why Zero-DCE++**
- **Size and speed:** about 10K parameters and 0.115G FLOPs for a 1200×900×3 image, with a default downsample factor of 12. Zero-DCE has about 79K parameters. The authors report 0.09 s on an i9 CPU versus 10 s for Zero-DCE.
- **Structure:** it outputs curve parameters, not pixels. The curve is monotonic and bounded, so it cannot invent texture, and it preserves ordering of neighbouring intensities.
- **Training:** zero-reference, so I can fine-tune on any dark photos without GT.
- **Rejected alternatives:**
  - RetinexNet needs paired data and is noisy.
  - EnlightenGAN has GAN instability and hallucination risk.
  - Retinexformer and similar transformers are heavy and tuned to LOL.
  - SCI is smaller still, and a good optional second backbone for an ablation.

**2.4 Size and cost (my arithmetic, not measured)**
- **Checkpoint:** about 10K parameters in fp32, roughly 45 KB.
- **RAM:** activations are the only real cost. With 32-channel feature maps at 512×512 and scale factor 1, one map is about 34 MB, so roughly 200 MB peak. At downscale 2 it is about 50 MB.
- **CPU time at 512×512:** from the paper's FLOP count (about 15 kFLOPs per input pixel), scale 1 is about 4 GFLOPs, so expect about 0.1 to 0.3 s on 4 cores. Downscale 2 gives about 0.05 s. Measure it on your machine.

**2.5 Pretrained vs fine-tune:** both, as separate rows. Pretrained is the fair baseline. The fine-tuned model is the NoirNet used in the system.

**2.6 Fine-tuning configuration**

*Losses (Zero-DCE definitions)*
- Curve: `LE(x) = x + α·x·(1−x)`, applied 8 times with the predicted α map.
- **Spatial consistency:** `L_spa = (1/K) Σ_i Σ_{j∈Ω(i)} (|Y_i − Y_j| − |I_i − I_j|)²`, on 4×4 average-pooled regions with 4 neighbours.
- **Exposure:** `L_exp = (1/M) Σ_k |Y_k − E|`, on 16×16 patches with E = 0.6 (the repo default).
- **Colour constancy:** `L_col = Σ_{(p,q)∈{(R,G),(R,B),(G,B)}} (J^p − J^q)²`, where J is the channel mean.
- **Illumination smoothness:** `L_tvA = (1/N) Σ_n Σ_c (|∇x A_n^c| + |∇y A_n^c|)²`.

*Our two additional losses (this is the contribution)*
- **Highlight guard:** `M = smoothstep(0.70, 0.90, Y_in)`, then `L_hl = mean(M · ReLU(Y_out − Y_in))`. Weight 5.
- **Chromaticity keeper:** `L_hue = mean(1 − cos(I_rgb, O_rgb))`, which penalises changes to RGB direction. Weight 2. Lower `W_col` by about 30% so the gray-world pull does not erase real coloured light.

*Loss weights.* The paper uses `W_col = 0.5` and `W_tvA = 20` (with Adam, fixed lr 1e-4, batch 8, 512×512 inputs). I recall the public training script using different scalings (about 1 / 10 / 5 / 200 for spa / exp / col / tv), but **I could not verify this**. Open the repo's training script and copy its defaults, then add the two new terms on top.

*Fine-tuning recipe*
- **Optimizer:** Adam, lr 2e-5 (about 5× lower than from scratch), weight decay 1e-4, gradient-norm clip 0.1.
- **Batch and crop:** batch 8, random 384×384 crops, scale_factor 1 during training.
- **Epochs:** at most 30 (about 500 iterations each).
- **Validation (every epoch):** on the 200 paired val pairs, compute `V = LPIPS + 0.5·(1−SSIM) + 5·new_clip%` on the NoirNet output.
- **Early stopping:** patience 5, min-delta 0.002, keep the best checkpoint.
- **Cost:** a model this small should train in well under an hour on a free GPU.
- **Optional ablation:** a supervised variant with an added L1 term on LOL train pairs.

**2.7 Contribution beyond integrating Zero-DCE++**
1. Noise-aware fusion and denoising driven by the classical illumination and effective-gain map.
2. The highlight and chromaticity losses.
3. The controller with closed-loop proxy checks and a do-no-harm gate.
4. The full evaluation protocol (section 6).

**2.8 Handling failures**
- **Overexposure:** highlight loss, the gain cap, the highlight shoulder, and the original image as a fusion candidate.
- **Colour:** chromaticity loss at training time plus HueLock at the end.
- **Noise:** gain-aware denoising, the fusion noise term, and a noise-based gain cap.

---

# 3. Classical CV pipeline

All images are float32 RGB in [0,1] (sRGB, no linearisation, as in the LIME paper). `Y = 0.299R + 0.587G + 0.114B`. Core primitives:

```python
import cv2, numpy as np

def box(x, r):
    return cv2.boxFilter(x, -1, (2*r+1, 2*r+1), normalize=True,
                         borderType=cv2.BORDER_REFLECT)

def guided_filter(I, p, r, eps):            # I guide (H,W), p input (H,W), float32
    mI, mp = box(I, r), box(p, r)
    a = (box(I*p, r) - mI*mp) / (box(I*I, r) - mI*mI + eps)
    b = mp - a*mI
    return box(a, r)*I + box(b, r)           # O(N), no opencv-contrib needed

K = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]], np.float32)
def immerkaer(Y, flat_pct=50):
    R = np.abs(cv2.filter2D(Y, -1, K, borderType=cv2.BORDER_REFLECT))[1:-1,1:-1]
    g = np.hypot(cv2.Sobel(Y, cv2.CV_32F, 1, 0), cv2.Sobel(Y, cv2.CV_32F, 0, 1))[1:-1,1:-1]
    m = g <= np.percentile(g, flat_pct)       # flat pixels only
    return float(np.sqrt(np.pi/2)/6 * R[m].mean())

def noise_map(Y, r=7):                        # local sigma for gain-aware steps
    R = np.abs(cv2.filter2D(Y, -1, K, borderType=cv2.BORDER_REFLECT))
    return np.sqrt(np.pi/2)/6 * box(R, r)
```

**3.1 Illumination estimation (LIME-style, partial correction)**
1. `T0 = max_c I_c` (the max-RGB prior, since reflectance ≤ 1).
2. `T = GuidedFilter(guide = Y, input = T0, r = round(0.03·min(H,W)), eps = 1e-3)`. Run it at half resolution (ds = 2) and upsample, because the map is smooth.
3. Clip `T` to [1e-3, 1]. I do not enforce `T ≥ T0`, because that would copy noise from `T0` into the map. The gain cap handles clipping instead.

**3.2 Guided filter settings**

| Use | Guide | Input | r | eps |
|---|---|---|---|---|
| Illumination | Y | T0 | 3% of the shorter side (about 8 at 256 px, 15 at 512) | 1e-3 (adaptive: ×3 if σ̂ > 0.02) |
| Luma denoise | Y_enh | Y_enh | 4 (≤ 1 MP), 6 above | `(1.2·σ̂·G_ref)²` |
| Fusion weights | Y_0 | each W_k | 8 | 1e-2 |

**3.3 Illumination correction (exact formula)**
```
G(x)      = clip( T(x)^(γ−1), 1, G_cap )
h(x)      = 1 − smoothstep(0.65, 0.95, max_c I_c(x))      # highlight shoulder
G'(x)     = 1 + (G(x) − 1)·h(x)
G''(x)    = min( G'(x), 1 / (max_c I_c(x) + 1e-3) )       # guaranteed no clipping
C(x)_c    = I_c(x)·G''(x)                                  # same gain on R, G, B
```
- **Gamma selection:** `γ = clip( ln(T_target) / ln(p50), γ_min, 1 )`. The defaults are `T_target = 0.45` and `γ_min = 0.30`. If `p50 ≥ 0.45`, `γ = 1`. Here `p50` is the median of the input Y.
- **Gain cap:** `G_cap = clip(0.06 / σ̂, 3, 10)`, so output noise `σ̂·G ≲ 0.06`.
- **Adaptive:** `γ`, `G_cap`, `eps`, and `r` are all adaptive.

**3.4 CLAHE:** apply to LAB L\*, which keeps chroma stable (unlike HSV V, which shifts saturation).
- `clip = clip(1 + 3·lowcon − 40·σ̂, 1.0, 4.0)`, with `lowcon = clip((0.45 − (p95−p5))/0.45, 0, 1)`.
- Tiles: (8, 8) up to about 1 MP, (12, 12) above.
- Blend `L = 0.4·L + 0.6·CLAHE(L)`.
- Apply it after enhancement and denoising, on the classical branch only.

**3.5 Denoising (both filters, post-fusion in EdgeGuard)**
- **Luma (guided filter):** `Ĝ(x) = clip( blur(Y_F, 5) / (blur(Y_0, 5) + 0.02), 1, 30 )` is the effective gain. `G_ref = P75(Ĝ)`. Compute `Y_s = GF(Y_F, Y_F, r=4, eps=(1.2·σ̂·G_ref)²)`. Then `w(x) = clip((Ĝ(x) − 1.5)/(G_ref − 1.5), 0, 1)^0.8` and `Y_out = w·Y_s + (1−w)·Y_F`. Regions that were barely amplified keep their texture.
- **Chroma (bilateral):** convert to YCrCb, then `bilateralFilter(Cr/Cb at half-res, d=7, sigmaColor=clip(3·255·σ̂·√Ĝ_med, 8, 40), sigmaSpace=5)`.
- **Why both:** guided preserves edges and has no gradient reversal on luma, and chroma noise is strongest in dark images and tolerates heavier smoothing.

**3.6 Noise estimation:** the Immerkaer function above, run on the input Y. The response is `R = |Y * N|`. For Gaussian noise, `E|R| = 6σ√(2/π)`, so `σ̂ = √(π/2)/6 · mean|R|`. I average only over the flattest 50% of pixels by Sobel gradient, so texture does not inflate the estimate. Use `noise_map` for local σ, and clip it to `[0.5σ̂, 2σ̂]`.

**3.7 Colour cast detection and correction (HueLock)**
1. Make a temporary brightened copy `I_t = clip(I·0.45/max(p50, 0.02), 0, 1)`, since a dark input is too noisy to judge colour.
2. Convert to LAB. Take mid-tone, low-chroma pixels: `25 ≤ L* ≤ 75` and `C* < 30`.
3. `cast = (mean a*, mean b*)`, magnitude `c = ‖cast‖`.
4. If `c > 6`, shift: `β = clip((c−6)/20, 0, 0.5)` and `(a*, b*) −= β·cast`.
5. Guards: halve β if the std of (a*, b*) is above 15 (a genuinely colourful scene), and cap the shift at 8 LAB units. This preserves real coloured light such as sodium lamps and neon.

**3.8 Highlight protection:** four layers.
1. Gain cap `1/max_c` (no clipping).
2. Gain attenuation `h(x)`, so bright regions are not boosted.
3. The soft-knee shoulder: `y = x` for `x ≤ 0.85`, else `0.85 + 0.15·tanh((x−0.85)/0.15)`.
4. The original as a fusion candidate, with its weight multiplied by 3 where `max_c I > 0.85`.

---

# 4. Adaptive fusion: LumenFuse

**4.1 Candidates:** `{I₀ (original), C (classical Retinex output, pre-denoise), D (NoirNet output, pre-denoise)}`. Denoising happens once, after fusion, so noise is not smoothed twice.

**4.2 Algorithm:** custom Mertens-style Laplacian-pyramid fusion. OpenCV's `createMergeMertens` has no noise term and no guided weight refinement, so I write it myself (about 50 lines).

**4.3 Weight formula (per pixel, per candidate k)**
```
W_k = (Contrast_k + ε)^wc · (Sat_k + ε)^ws · (Expo_k + ε)^we · Noise_k^wn
```
- **Contrast:** `|Laplacian(Gaussian(Y_k, σ=1))|` (3×3 kernel). Compute it on the lightly blurred copy, so noise is not rewarded as contrast.
- **Saturation:** `std_c(C_k)` across RGB channels.
- **Exposure:** `exp(−(Y_k − 0.5)² / (2·0.2²))`.
- **Noise:** `1 / (1 + (Ĝ_k(x)·σ_loc(x) / τ)²)`, with effective gain `Ĝ_k = clip(blur(Y_k, 5)/(blur(Y₀, 5)+0.02), 1, 30)` and `τ = 0.08`. This estimates the noise each candidate would carry per pixel.

Normalise: `Ŵ_k = (W_k + 1e-6) / Σ_j (W_j + 1e-6)`. Refine each map with the guided filter (guide = Y₀, r = 8, eps = 1e-2) and renormalise.

**4.4 Effect of each term:** contrast and saturation reward detail and colour. Exposure steers toward mid-tones, so both overly dark and blown regions lose weight. The noise term penalises candidates that amplify noise most.

**4.5 Noisy regions:** three places. The noise term, the gain cap upstream, and gain-aware denoising downstream. Noisy dark patches therefore lean toward less-amplified candidates.

**4.6 Granularity:** weights are pixel-wise (smoothed by the guided filter and pyramid), while the exponents `(wc, ws, we, wn, τ)` are global per image from LumenScan.

**4.7 Edge and texture preservation:** Laplacian-pyramid blending (no seams), guided-filtered weights (no halos at edges), and denoising strength modulated by gain.

**4.8 Pseudocode**
```
inputs: I0, C, D, σ̂, σ_loc, params(wc,ws,we,wn,τ)
Y0 = gray(I0)
for k in {I0, C, D}:
    Yk   = gray(k);  Ysm = GaussianBlur(Yk, σ=1)
    Con  = |Laplacian(Ysm)|;  Sat = std_c(k)
    Exp  = exp(-(Yk-0.5)^2 / 0.08)
    Ĝk   = clip(blur(Yk,5)/(blur(Y0,5)+0.02), 1, 30)
    Nse  = 1 / (1 + (Ĝk*σ_loc/τ)^2)
    Wk   = (Con+ε)^wc (Sat+ε)^ws (Exp+ε)^we Nse^wn
    if k == I0: Wk *= (1 + 2*[max_c I0 > 0.85])
normalise W; W = guided_filter(Y0, W, r=8, eps=1e-2); normalise again
L = min(5, floor(log2(min(H,W))) - 3)
for l in 0..L: R_l = Σ_k GaussPyr_l(Wk) * LapPyr_l(k)
F = collapse(R)
return F
```

**4.9 Improvement over a weighted average:** a fixed average cannot give different regions different sources, and a global blend cannot avoid halos and seams. Ablation G (section 7) measures this directly.

---

# 5. Adaptive controller: LumenScan

**5.1 and 5.2 Features (all on the input image; downscale to about 256 px for speed except noise)**

| Feature | Formula |
|---|---|
| Mean | `μ = mean(Y)` |
| Percentiles | `p5, p50, p95 = percentile(Y, [5, 50, 95])` |
| Dynamic range | `DR = p95 − p5` |
| Illumination non-uniformity | `NU = std(log2(T + 1/64))`, in stops; < 1 uniform, 1 to 2 moderate, > 2 strong |
| Noise | `σ̂ = immerkaer(Y)` |
| Colour cast | `(ā, b̄)` and `c = ‖(ā, b̄)‖` as in 3.7 |
| Clipped highlights | `s_hi = mean(max_c I ≥ 0.98)` |
| Crushed blacks | `s_lo = mean(max_c I ≤ 0.02)` |
| Darkness class | extreme `p50 < 0.06`, dark `< 0.20`, dim `< 0.45`, normal `≥ 0.45` |

**5.3 How features set parameters**

| Parameter | Rule | Default |
|---|---|---|
| γ | `ln(T_target)/ln(p50)`, clipped | T_target 0.45, γ_min 0.30 |
| G_cap | `clip(0.06/σ̂, 3, 10)` | 6 |
| Guided eps (illum.) | 1e-3, ×3 if σ̂ > 0.02 | 1e-3 |
| CLAHE clip | formula in 3.4 | 2.0 |
| DL strength s | `s = clip(1 − 0.6·(p50−0.15)/0.35, 0.4, 1)`, applied as `α' = s·α` | 1.0 |
| Denoise strength | follows σ̂ and Ĝ (formulas in 3.5) | |
| Fusion exponents | σ̂ < 0.008: (1, 0.5, 1, 1), τ = 0.08. 0.008 to 0.02: (0.7, 0.5, 1, 1.5), τ = 0.08. > 0.02: (0.5, 0.5, 1, 2), τ = 0.06 | |
| HueLock β | `clip((c−6)/20, 0, 0.5)` | 0 |
| Output blend α | `α = 1 − smoothstep(0.25, 0.50, p50)` | |

Final output: `out = (1−α)·I + α·F`.

**5.4 Rule-based vs optimisation:** rule-based for the MVP. Tune the coefficients afterwards with Optuna (advanced tier). The rule-based controller is explainable, which matters for judges, and it needs no data.

**5.5 Initial thresholds:** as in the tables above. Treat all numbers as starting values.

**5.6 Optuna (advanced)**
- **Search space (about 10 parameters):** `T_target ∈ [0.35, 0.55]`, `γ_min ∈ [0.25, 0.5]`, noise-cap constant `∈ [0.03, 0.12]`, CLAHE base `∈ [1, 3]`, CLAHE blend `∈ [0, 1]`, `wn ∈ [0.5, 2]`, `τ ∈ [0.03, 0.15]`, guided eps (log) `∈ [1e-4, 1e-2]`, `β_max ∈ [0.2, 0.8]`, `s_min ∈ [0.5, 1.0]`.
- **Data:** the 200 paired val images plus 100 synthetic extreme-dark images made from SICE val references.
- **Objective (minimise):** `J = LPIPS + 0.5·(1 − SSIM) + 0.03·NIQE + 5·new_clip% + 0.005·ΔE00`.
- **Setup:** TPE sampler, 150 trials, seed 0. Cache the NoirNet outputs so each trial only recomputes the classical branch and fusion.

**5.7 Closed loop (advanced)**
After producing F, compute cheap proxies and take the first matching action:

| Check | Action |
|---|---|
| New clipped pixels > 1% | `G_cap × 0.8`, shoulder start − 0.10 |
| Output median Y outside [0.30, 0.60] | `T_target` + 0.05 if low, − 0.07 if high (only if the noise cap is not binding) |
| `σ̂_out > 0.025` | denoise strength × 1.5, `G_cap × 0.85` |
| Mean hue shift on coloured pixels > 12° | reduce HueLock β and DL strength s |

- **Stop rule:** at most 2 extra passes. Stop earlier if all checks pass or the proxy score `Q = 10·new_clip + 2·|median − 0.45| + 10·σ̂_out` fails to improve. Return the lowest-Q result.
- **Why proxies, not NIQE or LPIPS:** they are too slow for a live loop.

**5.8 Not making a good image worse**
- The α blend sends well-exposed images to near-identity.
- Bypass gate: if `p50 ≥ 0.45`, `DR ≥ 0.45`, and `s_hi < 1%`, return the input unchanged.
- SSIM guard: if `SSIM(I, F) < 0.85` on a non-dark image (`p50 ≥ 0.25`), blend back toward I.
- Verified on the do-no-harm set with targets: input-output SSIM ≥ 0.90 and NIQE change ≤ +0.1.

---

# 6. Evaluation and metrics

**6.1 Mandatory:** PSNR, SSIM, LPIPS, NIQE, latency, noise suppression, edge preservation. I also strongly recommend a colour-shift metric (ΔE00) and new-clipping %.

**6.2 Libraries**
- **PSNR and SSIM:** `skimage.metrics.peak_signal_noise_ratio(gt, out, data_range=1.0)` and `structural_similarity(gt, out, channel_axis=2, data_range=1.0)`. Report both raw and GT-mean-aligned versions.
- **LPIPS:** `pip install lpips`, `lpips.LPIPS(net='alex')`, with inputs scaled to [−1, 1].
- **NIQE:** `pyiqa.create_metric('niqe')` from the IQA-PyTorch package. Implementations differ slightly from the MATLAB original, so use it for relative comparisons only, with identical code for all methods.
- **LOE (optional):** the NPE-paper definition, computed on 50×50 downscaled max-RGB maps.

**6.3 No-reference evaluation:** NIQE, LOE, plus the artifact metrics below and the blind A/B study.

**6.4 Artifact metrics**

| Aspect | Metric |
|---|---|
| Noise suppression | `σ̂_flat(out) / mean(Y_out[flat])` (relative noise), plus residual vs GT in flat regions when paired |
| Edge preservation | Pearson correlation of gradient-magnitude maps (out vs GT, or out vs input when unpaired) on the strongest 10% of edges; local-std retention in textured patches |
| Colour distortion | ΔE00 vs GT (`skimage.color.deltaE_ciede2000`); unpaired: mean hue difference on pixels with `C* > 10` |
| Overexposure | new clipped % = `frac(max_c out ≥ 0.98) − frac(max_c in ≥ 0.98)`, and the share of input pixels in [0.85, 0.98] pushed to clipping |
| Halo | gradient-reversal rate: share of strong-edge pixels (top 5% input gradient after σ=2 blur) where the output gradient sign flips |

**6.5 Image counts:** all of LOL-v1 (15), LOL-v2-Real (100), LOL-v2-Syn (100), SICE P2 subset (200), the five classic no-reference sets (199), ExDark (200), own (60), do-no-harm (100). The severity sweep is 50 images × 6 darkness levels × 4 noise levels = 1,200 renders.

**6.6 Reporting:** mean ± std, median, P10 and P90, and the worst five images per metric. Add a Wilcoxon signed-rank test (hybrid vs each baseline). Tabulate separately for the hard slice (darkest 20% by `p50` plus `NU > 2`).

**6.7 Latency and memory:** `time.perf_counter()`, 10 warm-up runs then 100 timed runs (50 at 1080p), batch size 1, fixed threads (`torch.set_num_threads(4)`, `cv2.setNumThreads(4)`). Report median and P95, with a per-stage breakdown. Test three sizes: 640×480, 1280×720, 1920×1080. Track peak RSS with `psutil.Process().memory_info().rss` sampled on a thread.

**6.8 Targets (expectations, not guarantees)**
- **Baseline context:** published tables list Zero-DCE++ at about 14.7 dB PSNR on LOL-v1 and NIQE of about 3.5 to 4.0 on DICM, LIME, MEF and NPE (e.g. DICM 3.54, MEF 3.4, NPE 3.47, LIME 3.97). These come from an aggregator, so use them only as rough context. Zero-reference methods score well below supervised ones on PSNR.
- **Hybrid goals:**
  - PSNR at least 2 dB above our own Zero-DCE++ baseline on LOL-v1 and LOL-v2-Real (raw).
  - LPIPS better than both single-branch baselines.
  - NIQE equal to or better than the best baseline.
  - New-clip % under 1% on average.
  - The do-no-harm targets from 5.8.
- **Latency goals:** at most 1.5 s at 512×512 on 4 CPU cores, at most 10 s at 1080p.

---

# 7. Ablation studies

Run leave-one-out on the full system, evaluated on LOL-v2-Real, LOL-v2-Syn and the hard slice, with all other settings frozen. The headline table compares the three required approaches: **classical only** (LumenRetinex + CLAHE + EdgeGuard + HueLock), **DL only** (NoirNet pretrained and fine-tuned, raw and with the same EdgeGuard + HueLock), and **hybrid** (full).

| ID | Change | Compare | Metrics | Validated if |
|---|---|---|---|---|
| A | Adaptive controller → fixed global parameters | Full vs fixed | Per-class LPIPS, new-clip %, do-no-harm SSIM | Fixed is worse on the hard slice and on normal images |
| B | Guided illumination → Gaussian (SSR-style) | Halo score and SSIM | Gradient-reversal rate, SSIM | Halo rate drops ≥ 30% with guided |
| C | Remove CLAHE | Contrast and NIQE | Entropy, NIQE | NIQE or entropy improves with CLAHE |
| D | Remove gain-aware denoising | Noise vs texture | Relative noise, NIQE | Noise rises clearly without it |
| E | Gain-aware → uniform strong denoising | Edge and texture loss | Edge correlation, SSIM | Edge correlation drops with uniform |
| F | Fine-tuned → pretrained NoirNet | DL-branch quality | PSNR, LPIPS, hard slice | Fine-tuned better, especially on extreme-dark |
| G | Pyramid fusion → 50/50 average or global α | Fusion quality | LPIPS, halo, seams | Pixel-wise fusion beats both |
| H | Remove HueLock or highlight guard | Colour and clipping | ΔE00, new-clip % | Either removal worsens its target metric |
| I | Fusion noise term off (wn = 0) | Noise amplification | Relative noise, NIQE | Dark regions get noisier |
| J | Remove original-image candidate | Highlight safety | New-clip %, hue shift | Clipping rises without it |
| K | (Advanced) Closed loop off | Tail failures | LPIPS on worst 10% | Worst-case improves with loop |

Each ablation changes exactly one item. I'd expect modest, not dramatic, gains on average, with larger differences on the hard slice. If an ablation shows no difference, report it honestly and simplify the system.

---

# 8. Architecture and implementation

**8.1 Final pipeline**
```
Input I
 → LumenScan (features, T, σ̂, parameters)
 → in parallel: [Classical: LumenRetinex → C]   [NoirNet: Zero-DCE++ α-map → D]
 → LumenFuse({I, C, D}) → F
 → EdgeGuard (gain-aware luma guided + chroma bilateral)
 → CLAHE on L (inside classical path only, or after fusion if C is skipped)
 → HueLock → highlight shoulder
 → closed-loop proxy check (advanced; up to 2 passes)
 → α blend with the original → Output
```

**8.2 Responsibilities**
- **LumenScan:** statistics, illumination map, noise estimate, and all parameter decisions.
- **NoirNet:** a Zero-DCE++ curve estimator producing D; the α map is multiplied by DL strength `s`.
- **LumenFuse:** pixel-wise, noise-aware pyramid fusion of the three candidates.
- **EdgeGuard:** gain-aware luma and chroma denoising, plus halo suppression via guided weights.
- **HueLock:** cast detection and partial correction, with guards for coloured light.

**8.3 Parallel vs sequential:** classical branch and NoirNet run in parallel (`ThreadPoolExecutor(2)`; OpenCV and ONNX Runtime release the GIL). LumenScan comes first; everything after fusion is sequential.

**8.4 Implement first:** the metrics and evaluation harness (so every later step is measured), then the Retinex branch.

**8.5 Essential for v1:** LumenScan (basic stats + σ̂), LumenRetinex, pretrained Zero-DCE++, LumenFuse (with the noise term), EdgeGuard, and a simple UI.

**8.6 Advanced:** fine-tuning, the highlight and hue losses, HueLock refinements, Optuna, the closed loop, and the severity sweep.

**8.7 Cut for simplicity:** the chroma pre-denoise, a learned fusion network, oracle-parameter distillation, supervised fine-tuning as a main path, MSRCR/SRIE, non-local means, and extra GAN or transformer baselines (add one only as an upper-bound reference if time permits).

---

# 9. Performance and deployment

These are **estimates, not measurements**, for a 4-core laptop CPU. Free hosted CPU tiers often have fewer cores (check your platform), so scale accordingly.

**9.1 Time at 512×512:**

| Stage | Estimate |
|---|---|
| LumenScan | 30 to 60 ms |
| Classical branch | 80 to 150 ms |
| NoirNet (downscale 2) | about 50 ms |
| LumenFuse | 100 to 200 ms |
| EdgeGuard + HueLock | 70 to 120 ms |
| **Total** | **about 0.4 to 0.7 s**, plus about 0.3 s per closed-loop pass |

**9.2 RAM:** about 0.8 to 1.2 GB at 512×512, dominated by importing PyTorch (about 300 to 500 MB); under about 2 GB at 1080p. With ONNX Runtime instead of PyTorch, expect noticeably less.

**9.3 10-second target:** yes. Even with two closed-loop passes, 512×512 should be under 2 s and 1080p should fit in about 5 to 8 s. Measure it before the demo.

**9.4 Likely bottlenecks:** the pyramid fusion at full resolution, repeated box filters in guided filtering, full-resolution bilateral on chroma, and PyTorch warm-up and import time on the first request.

**9.5 Optimisations that keep quality**
- Compute illumination and weight maps at 1/2 to 1/4 resolution and upsample (they are smooth by design).
- Run NoirNet at downscale 2 to 4 for large images, and apply the curve at full resolution outside the network.
- Use `float32` throughout and cap the long edge at 1920.
- Preload the model and warm it up at startup.
- Run the classical and DL branches in parallel.

**9.6 Inference stack:** develop in PyTorch, then export NoirNet to ONNX for deployment (`torch.onnx.export`, opset 17). Export the α-map network only (resize and curve application stay in NumPy), which avoids dynamic-resize export issues. Serve with Gradio (fastest for a hackathon) or FastAPI + a small web UI, inside Docker.

---

# 10. Final technical decisions

| Component | Algorithm | Math | Initial parameters | Libraries | Input → Output | Required? | Cost |
|---|---|---|---|---|---|---|---|
| LumenScan | Statistics + rule-based mapping | Percentiles, `std(log2 T)`, Immerkaer σ̂ | thresholds in §5 | numpy, opencv | RGB → feature dict + params | Mandatory | Low |
| LumenRetinex | LIME-style max-RGB + guided illumination, partial gain | `G = T^(γ−1)`, caps, shoulder | r = 3%, eps 1e-3, T_target 0.45, γ_min 0.3 | opencv, numpy | RGB, params → C, G | Mandatory | Low |
| NoirNet | Zero-DCE++ | 8-iteration curve `x + αx(1−x)` | scale 2 to 4, s = 1 | torch → onnxruntime | RGB → D | Mandatory | Low |
| LumenFuse | Pixel-wise Laplacian-pyramid fusion with noise term | `W = Con·Sat·Exp·Noise`, guided refine | (1, 0.5, 1, 1), τ 0.08, 5 levels | numpy, opencv | {I, C, D} → F | Mandatory | Medium |
| EdgeGuard | Gain-aware guided luma + bilateral chroma | `w(x)`, `eps = (1.2σ̂G_ref)²` | r = 4, σ_color 8 to 40 | opencv | F → F' | Mandatory | Medium |
| CLAHE | On LAB L\*, blended | clip formula §3.4 | clip 2.0, tiles 8×8, blend 0.6 | opencv | RGB → RGB | Mandatory (required) | Low |
| HueLock | LAB mid-tone cast shift with guards | `β·(ā, b̄)` | threshold 6, β ≤ 0.5, cap 8 | opencv, skimage | F' → F'' | Mandatory | Low |
| Closed loop | Proxy checks, ≤ 2 passes | `Q = 10·clip + 2·|med−0.45| + 10σ̂` | thresholds §5.7 | numpy | F'' → out | Optional | Medium |
| Evaluation | Metric suite | §6 | | skimage, lpips, pyiqa, scipy | → tables | Mandatory | Offline |

**A. Final MVP:** LumenScan (stats + σ̂) → LumenRetinex (adaptive γ and G_cap) ‖ pretrained Zero-DCE++ → LumenFuse → EdgeGuard → HueLock → α blend. Plus the metric harness and a Gradio demo.

**B. Advanced version, in this order:** (1) fine-tuned NoirNet with the highlight and hue losses; (2) CLAHE and its adaptive clip; (3) Optuna tuning; (4) closed loop; (5) severity sweep, blind A/B test, and automatic failure tagging; (6) ONNX export.

**C. Implementation order**
1. Data loaders, the dedupe script, and the split files.
2. Metric harness (PSNR, SSIM, LPIPS, NIQE, artifact metrics, latency).
3. Guided filter, Immerkaer noise, LumenScan.
4. LumenRetinex; evaluate it as the classical baseline.
5. Pretrained Zero-DCE++ inference; evaluate it as the DL baseline.
6. LumenFuse; evaluate the first hybrid.
7. EdgeGuard and HueLock.
8. Controller rules wired in; do-no-harm check.
9. Gradio app with intermediate visualisations (illumination map, gain map, fusion weight maps, per-stage outputs).
10. Fine-tune NoirNet and re-run all tables.
11. Ablations A to J.
12. Optuna tuning, then freeze.
13. Test on unseen sets, severity sweep, and own images.
14. Failure analysis and report.

**D. References**
- Zero-DCE++: Li, Guo, Loy, TPAMI 2022, arXiv 2103.00860; project page li-chongyi.github.io/Proj_Zero-DCE++.html.
- Zero-DCE: Guo et al., CVPR 2020, arXiv 2001.06826.
- LIME: Guo, Li, Ling, TIP 2017.
- Guided filter: He, Sun, Tang, ECCV 2010 / TPAMI 2013.
- Exposure fusion: Mertens, Kautz, Van Reeth, Pacific Graphics 2007. Guided-filter fusion weights: Li, Kang, Hu, TIP 2013.
- Noise estimation: Immerkaer, CVIU 1996.
- CLAHE: Zuiderveld, Graphics Gems IV, 1994.
- LOL / RetinexNet: Wei et al., BMVC 2018. LOL-v2: Yang et al., TIP 2021.
- SICE: Cai, Gu, Zhang, TIP 2018.
- LPIPS: Zhang et al., CVPR 2018 (`richzhang/PerceptualSimilarity`). NIQE: Mittal et al., 2013. `pyiqa` (IQA-PyTorch, Chaofeng Chen). LOE: Wang et al. (NPE), TIP 2013.
- Optional baselines: SCI (CVPR 2022), RUAS (CVPR 2021), EnlightenGAN (TIP 2021), Retinexformer (ICCV 2023).

Venues and years are from memory; check the DOIs before putting them in a report.

**E. What is and is not novel**
Retinex, Zero-DCE++, guided and bilateral filtering, CLAHE and exposure fusion are all existing techniques, so we should not claim them. The contribution is how they are combined:
1. **One effective-gain map** drives DL curve strength, fusion noise penalties and denoising strength, so noise control is applied exactly where amplification happened.
2. **Noise-penalised exposure fusion** across {original, classical, DL} candidates with guided-refined weights.
3. **Highlight-guard and chromaticity losses** for zero-reference fine-tuning.
4. **LumenScan:** an interpretable controller with a do-no-harm gate and proxy-based closed loop.
5. **Evaluation:** a severity sweep, hard-slice reporting, and automated failure tagging.

I would phrase it in the report as "a controlled hybrid design with measured ablations", not as a new state-of-the-art model, since judges and reviewers will check.

---

Do you want me to put this into a shareable doc, or start with the code for the metric harness and LumenScan?