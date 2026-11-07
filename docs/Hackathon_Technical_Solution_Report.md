# Lumenoir - Hackathon Technical Solution Report

**Robust Low-Light Image Restoration and Enhancement**

| Field | Value |
| --- | --- |
| Project | Lumenoir |
| Problem ID | 26035 (not confirmed by project records - see Section 7) |
| Problem title | Robust Low-Light Image Restoration and Enhancement (Problem Statement P1) |
| Team | [TEAM NAME] |
| Members / USNs | [NAME - USN], [NAME - USN], [NAME - USN] |
| Repository | https://github.com/Chiranthcommits25/Lumenoir |
| Evaluated commit | e020c54383a66800e191d7139f74371625e18aea |
| Live demo | Pending - no deployment URL recorded |
| Report date | 7 November 2026 |

> Evidence note: figures marked VERIFIED are supported by a file in the repository or by a command run for this report. Figures marked UNVERIFIED come from the project brief, have no artifact behind them, and are never combined with verified data.

## Executive Summary

Lumenoir is an illumination-adaptive low-light image restoration system. It analyses an uploaded photograph, estimates its illumination, noise and colour characteristics, and then combines a classical Retinex-style branch with a lightweight neural curve estimator through a noise-aware multiscale fusion stage. The repository contains a complete, tested implementation of that pipeline behind a FastAPI service and a React workspace.

This report documents the work that exists in the repository at the evaluated commit. It was produced by inspecting the source modules, the configuration files, the documentation and the test suite; it did not modify code, retrain any model, or run new evaluation experiments. Where the project brief supplies figures that have no corresponding artifact in the repository, those figures are reproduced in a clearly separated table marked UNVERIFIED and are never combined with repository evidence.

**Verified at the evaluated commit:**

- The seven-stage enhancement pipeline (LumenScan, LumenRetinex, Zero-DCE++, LumenFuse, EdgeGuard, HueLock, adaptive alpha blend) is implemented and wired into `POST /api/enhance`, in three selectable modes: classical, deep learning and hybrid.
- A unified device runtime selects CPU, CUDA or Apple MPS once per pipeline; FP32 is the default precision, FP16 is opt-in, and the CPU reference path runs without PyTorch installed.
- The full test suite passes: 394 passed, 3 skipped, 8 deselected, with Ruff clean across backend, tests and training.
- Local accelerator parity is measured: FP32 fusion agrees with the CPU reference to MAE <= 1.23e-6 and PSNR >= 103.67 dB; FP16 MAE <= 1.4e-4 and PSNR >= 75 dB.
- Local latency is measured and recorded in the repository for the fusion stage and for the full pipeline on a 10-image, 600x400 set.

**Not evidenced in the repository:**

- No dataset-level quality results exist. `training/reports/` is empty and the only CSV/JSON report files are explicitly-labelled placeholders in `training/kaggle_eval/samples/`. The six-way LOL comparison supplied in the brief therefore cannot be cross-checked and is marked UNVERIFIED.
- No fine-tuned checkpoint exists. `backend/weights/zero_dce_plus_plus.pth` is the 52,395-byte released Zero-DCE++ checkpoint; the fine-tuning pipeline is implemented but has not been run to completion in this repository.
- No visual comparison images exist anywhere in the repository outside the dataset folder.
- The full pipeline has never been executed on CUDA hardware; only Apple MPS and CPU were available.
- No public deployment URL is recorded.

> Reading guide: every table is labelled VERIFIED (backed by a file in the repository, or by a command run for this report on this machine) or UNVERIFIED (supplied by the brief or an earlier session with no artifact). Pending items are collected in Section 7.

---

## 1. Problem Understanding & Solution Strategy

### 1.1 The restoration problem

Severely underexposed photographs are not simply dark. They combine several degradations that interact, and correcting one usually worsens another. The project brief (Problem Statement P1, recorded in `context.md`) names eight conditions the system must handle; the implemented architecture addresses each explicitly:

- **Extreme underexposure.** Sensor values in the darkest regions sit close to the quantisation floor, so a global brightness curve turns them into flat grey. Lumenoir measures the median luminance and derives the correction exponent from it instead of applying a fixed curve.
- **Non-uniform illumination.** Light falls off across the frame, and a global gain over-brightens the already-lit side. Lumenoir estimates a spatial illumination map and derives a per-pixel gain from it.
- **Sensor noise.** Amplifying shadows amplifies noise. Lumenoir estimates the noise level with an Immerkaer estimator, feeds that estimate into the fusion weights, and applies gain-aware denoising afterwards.
- **Colour distortion.** Low-light captures drift in hue, and grey-world assumptions can strip real colour. Lumenoir measures the LAB colour cast on mid-tone low-chroma pixels and applies a bounded cast shift that leaves lightness untouched.
- **Loss of fine detail.** Aggressive denoising destroys texture. The edge-preserving guided filter blends denoising strength by effective gain, so smoothed regions are those that were actually amplified.
- **Highlight clipping after enhancement.** Bright regions that were already correct become white. Highlight attenuation in the gain, a soft-knee shoulder, and a threefold fusion weight on the original candidate above a highlight threshold all protect them.
- **Halo artefacts.** Illumination correction around strong edges produces halos. The pipeline exposes a halo-rate metric and a do-no-harm bypass gate, and keeps the original image as a fusion candidate.
- **Latency and resource limits.** The brief targets roughly ten seconds per image and affordable CPU hosting. The chosen network has 10,561 parameters and a 52 KB checkpoint, and the full pipeline runs on CPU with NumPy and OpenCV only.

### 1.2 The Lumenoir methodology

Lumenoir treats enhancement as a measurement problem before it treats it as a filtering problem. The first stage, LumenScan, produces a compact statistical description of the image: luminance percentiles and dynamic range, an illumination non-uniformity measure in stops, an Immerkaer noise estimate, a LAB colour cast, highlight and crushed-black fractions, and a darkness class. From those measurements a documented rule set selects every downstream coefficient: the Retinex gamma and gain cap, the guided-filter epsilon, the CLAHE clip limit and tile grid, the deep-learning strength, the fusion exponents and noise time constant, the colour-correction strength, and the final blend alpha. No stage downstream of LumenScan chooses its own parameters, so the system behaves consistently and every decision is reproducible from the measured statistics.

Two enhancement branches then run from the same input. The classical branch, LumenRetinex, decomposes the image into an illumination map and a reflectance-like ratio, applies a gamma-derived gain, attenuates the gain where the image is already bright, caps the gain so that no pixel is pushed past unity, equalises local contrast with CLAHE on the L* channel only, and applies a soft-knee highlight shoulder. The neural branch runs Zero-DCE++, a curve-estimation network that predicts a per-pixel tone curve and applies it eight times. The two outputs, together with the untouched original, become the three candidates for fusion.

LumenFuse then decides, per pixel and per scale, how much of each candidate to keep. Its weights combine local contrast, colour saturation, well-exposedness and a gain-aware noise penalty; the weights are refined with a guided filter and blended through Laplacian pyramids. Finally EdgeGuard removes gain-proportional noise with a luma guided filter and a chroma bilateral filter, HueLock applies the bounded cast correction, and an adaptive alpha blend with the original image produces the delivered result. A do-no-harm gate short-circuits the whole pipeline when the input is already well exposed.

### 1.3 Why an adaptive hybrid architecture

A purely classical pipeline is fast, deterministic and interpretable, but its gain is derived from an illumination prior (max-RGB) rather than from learned appearance. On extremely dark frames that prior collapses toward the noise floor, and the resulting gain either under-corrects or produces a flat, washed-out image. CLAHE then amplifies the remaining noise. Classical methods also have no notion of what a natural result looks like, so their failure mode is a plausible-looking but perceptually wrong image.

A purely neural pipeline has the opposite profile. Zero-DCE++ is reference-free and extremely small, but its global tone curve is driven by content rather than by measured illumination, so it can leave non-uniform lighting uncorrected, drift in colour, and brighten already-correct highlights. On its own it is a strong baseline rather than a complete restoration system.

The hybrid design exploits the complementarity. The classical branch is strong exactly where the network is weak (spatially varying illumination, explicit gain caps, interpretable colour handling), and the network is strong where the classical branch is weak (learned natural appearance in extremely dark, noisy regions). Fusion decides between them using measured local evidence rather than a fixed blend, and the noise term prevents the network's brighter output from winning in regions where it would only add noise. All three modes remain separately selectable, which is what makes the classical-versus-neural comparison measurable rather than rhetorical.

The architecture also reflects the deployment constraint in the brief. Because the expensive network is small and optional, classical mode runs on a CPU-only host with no PyTorch installed, and deep-learning or hybrid requests degrade to a typed 503 rather than a crash.

---

## 2. Technical Architecture & Implementation

The pipeline is orchestrated by `backend/enhancement/pipeline.py`, whose stage objects in `backend/enhancement/stages.py` delegate to the algorithm modules. The stage classes contain no algorithm logic: each one calls an existing module, publishes its artifact, and records timing. The executed order for hybrid mode is analysis, classical, deep learning, fusion, denoising, colour correction, quality assessment.

**Table 1 - Implemented stages and their delegated modules (VERIFIED from source).**

| Stage | Responsibility | Module |
| --- | --- | --- |
| analysis | Features, illumination and noise maps, rule-based parameters, bypass decision | `classical_cv/scan.py` |
| classical_cv | Adaptive gain, highlight shoulder, CLAHE on L* | `classical_cv/retinex.py` |
| deep_learning | Zero-DCE++ curve estimation through the load-once manager | `deep_learning/zero_dce_plus_plus.py` |
| fusion | Noise-aware weights, guided refinement, Laplacian pyramid blend | `fusion/lumen_fuse.py` |
| denoising | Gain-aware luma guided filter, chroma bilateral filter | `classical_cv/edge_guard.py` |
| colour_correction | Bounded LAB cast correction | `classical_cv/hue_lock.py` |
| quality_assessment | Adaptive alpha blend with the original, final validation | `enhancement/stages.py` |

### 2.1 Classical algorithmic baseline

**Illumination estimation** (`backend/classical_cv/illumination.py`) follows the LIME prior: the initial illumination map is the per-pixel maximum across the three channels, which is then smoothed by a guided filter using luminance as the guide at half resolution, and clipped to [1e-3, 1]. The guided filter's radius defaults to three percent of the shorter side. This map is what makes the correction spatially aware: dark regions receive a larger gain than bright ones.

**LumenRetinex** (`backend/classical_cv/retinex.py`) implements the documented classical branch in a fixed order. It recomputes the illumination map at input resolution, derives a gain from the selected gamma as the illumination raised to (gamma - 1) and clamped to [1, gain_cap], attenuates that gain by a smoothstep ramp of the maximum channel so already-bright pixels are protected, and further caps it at the reciprocal of the maximum channel plus a small epsilon so no channel can exceed unity. The same gain is applied to all three channels, which preserves hue. CLAHE then operates on the L* channel only, using a 16-bit lightness buffer and a fixed 40/60 blend between the original and equalised lightness; a* and b* are untouched. A per-channel soft-knee shoulder compresses values above 0.85 with a tanh curve, and the result is clipped to [0, 1] while preserving dimensions exactly.

**Contrast limiting** is therefore adaptive in strength but conservative in placement. The clip limit is derived from both the dynamic range and the noise estimate, the tile grid switches from 8x8 to 12x12 above one megapixel, and the blend factor keeps 40 percent of the un-equalised lightness, which is what prevents the flat, crunchy look of full CLAHE on noisy input.

**Filtering primitives.** `backend/classical_cv/filters.py` is the single implementation of the box filter (OpenCV boxFilter with reflected borders) and the guided filter, which computes local means, variance and covariance and returns the linear-coefficient reconstruction. Every guided filter in the system - illumination smoothing, LumenRetinex, LumenFuse weight refinement and EdgeGuard luma denoising - uses this one function. The chroma denoiser in EdgeGuard uses OpenCV's bilateral filter, chosen because chroma noise is better handled by a range filter than by a linear-coefficient filter.

**Noise estimation** (`backend/classical_cv/noise.py`) uses the Immerkaer estimator: a Laplacian-like second-difference kernel, scaled by sqrt(pi/2)/6, averaged over the flattest half of the image selected by Sobel gradient percentile. A local noise map of the same estimator over a 7-pixel box is produced for the fusion stage, optionally clipped to [0.5 sigma, 2 sigma].

**EdgeGuard** (`backend/classical_cv/edge_guard.py`) denoises after fusion, where the amplification is known. It computes an effective gain as the ratio of the blurred fused luminance to the blurred original luminance, clamped to [1, 30], takes its 75th percentile as a reference, and derives a per-pixel blend weight from that ratio. Luma is smoothed by a guided filter whose epsilon scales with the noise estimate and the reference gain squared; the blend weight decides how much of the smoothed luma survives. If the reference gain is at or below 1.5, nothing meaningful was amplified, so the guided filter is skipped entirely. Chroma is filtered at half resolution with a bilateral filter whose colour sigma scales with the noise estimate and the median gain.

**HueLock** (`backend/classical_cv/hue_lock.py`) consumes LumenScan's detected cast and raw strength, halves that strength when the mid-tone colour spread is high, and caps the applied shift at eight LAB units. It shifts only a* and b*, so L* - and therefore lightness, highlights and shadows - is preserved exactly, and chroma is never boosted. With a zero cast estimate it returns the input unchanged without a colour-space round trip.

**Adaptive parameter selection** is centralised in `backend/classical_cv/scan.py`. Measurement happens on a copy whose longest edge is at most 256 pixels, except the noise estimate, which uses full-resolution luminance. The following table reproduces the implemented rules; the coefficients are the documented starting values from the project decision record, and no stage is tuned against evaluation data.

**Table 2 - LumenScan measurements and the parameter rules they drive (VERIFIED from source).**

| Measurement | Definition | Parameters it drives |
| --- | --- | --- |
| p5 / p50 / p95 | Luminance percentiles on the analysis copy | gamma, dl_strength, output_blend_alpha, bypass |
| dynamic range | p95 - p5 | clahe_clip, bypass |
| non-uniformity | std(log2(T + 1/64)) in stops | reported feature; 1-2 moderate, >2 strong |
| noise sigma | Immerkaer on full-resolution luminance | gain_cap, guided_eps, clahe_clip, fusion exponents, EdgeGuard |
| colour cast | mean LAB (a*, b*) on brightened mid-tone low-chroma pixels | hue_lock_beta |
| highlight fraction | share of pixels with max channel >= 0.98 | bypass |
| black fraction | share of pixels with max channel <= 0.02 | reported feature |
| darkness class | extreme < 0.06, dark < 0.20, dim < 0.45, normal >= 0.45 | reported feature |

**Table 3 - Implemented parameter rules (VERIFIED from `classical_cv/scan.py` and `docs/classical_cv.md`).**

| Parameter | Rule |
| --- | --- |
| gamma | clip(ln(0.45)/ln(p50), 0.30, 1) |
| gain_cap | clip(0.06 / noise_sigma, 3, 10) |
| guided_eps | 1e-3, or 3e-3 when noise_sigma > 0.02 |
| clahe_clip | clip(1 + 3*lowcon - 40*noise_sigma, 1, 4), lowcon = clip((0.45 - DR)/0.45, 0, 1) |
| clahe_tiles | (8, 8) up to 1 MP, (12, 12) above |
| clahe_blend | 0.6 equalised / 0.4 original lightness |
| dl_strength | clip(1 - 0.6*(p50 - 0.15)/0.35, 0.4, 1) |
| fusion exponents, tau | noise < 0.008: (1, 0.5, 1, 1), tau 0.08; < 0.02: (0.7, 0.5, 1, 1.5), tau 0.08; else (0.5, 0.5, 1, 2), tau 0.06 |
| hue_lock_beta | clip((cast_magnitude - 6)/20, 0, 0.5) |
| output_blend_alpha | 1 - smoothstep(0.25, 0.50, p50) |
| bypass_gate | p50 >= 0.45 and DR >= 0.45 and highlight_fraction < 0.01 |

> A consequence of the bypass rule, acknowledged in the repository documentation and encoded by a test: a bimodal image with a well-exposed median, wide dynamic range and no clipped highlights is bypassed even when it contains a large dark region, because the gate inspects only those three statistics.

### 2.2 Deep learning pipeline

**Architecture.** Zero-DCE++ is reproduced from the authors' implementation in `backend/deep_learning/zero_dce_net.py`. Seven depthwise-separable blocks (a 3x3 depthwise convolution followed by a 1x1 pointwise convolution) with 32 feature channels produce a curve map, with skip concatenations between the early and late blocks. The curve map is passed through tanh and applied eight times as x + alpha*(x^2 - x). Downsampling uses bilinear interpolation with align_corners=False; upsampling uses the explicit align_corners=True variant that matches the authors' UpsamplingBilinear2d, which matters because the PyTorch default would change the output. There is no batch normalisation and no mean/std normalisation.

**Table 4 - Zero-DCE++ facts verified in this repository.**

| Item | Value |
| --- | --- |
| Parameters | 10,561 in 28 float32 tensors |
| Checkpoint size | 52,395 bytes |
| Checkpoint SHA-256 | ca8855b90df9a80fa4195a831f33d3476b1964f787eb70602797c773067f3b84 |
| Upstream source | Li-Chongyi/Zero-DCE_extension, Zero-DCE++/snapshots_Zero-DCE++/Epoch99.pth |
| Licence | CC BY-NC 4.0 (academic research use) |
| Loading | torch.load(weights_only=True), strict key match, warm-up once |
| Geometry | padded bottom/right to a multiple of the scale factor, cropped back to exact input size |

**Why Zero-DCE++.** The brief requires a model that is small, fast, CPU-capable and deployable without a GPU. Zero-DCE++ satisfies all four by a wide margin: 10,561 parameters against tens of millions for RetinexNet-class models, a 52 KB checkpoint, and a forward pass that is a handful of 3x3 and 1x1 convolutions. It is also reference-free during training, which suits a restoration problem where paired ground truth is scarce and often misaligned. Its weakness - a global tone curve that ignores scene structure - is precisely what the classical branch and the fusion stage compensate for.

**Fine-tuning procedure.** `training/finetune/` implements fine-tuning of the existing network without changing the inference architecture. The released checkpoint is read-only input; outputs are written to a separate directory as `best.pth` (a bare state dict accepted by the strict inference wrapper) and `last_training.pt` (optimizer, scheduler, scaler and epoch state for resumption). Training reads the materialised paired splits, rejects any path containing a test-split component, refuses to start if train and validation sets overlap, applies seed 42 to Python, NumPy, PyTorch and CUDA, enables cuDNN determinism, and uses AMP only on CUDA.

**Table 5 - Fine-tuning configuration (VERIFIED from `training/finetune/configs/zero_dce_plus_plus_lol.json`).**

| Setting | Value |
| --- | --- |
| Optimizer | Adam, learning rate 2e-5, weight decay 1e-4 |
| Schedule | ReduceLROnPlateau, factor 0.5, patience 2, minimum learning rate 1e-6 |
| Epochs | 30 maximum |
| Batch / crop | 8 images, 384x384 deterministic crops |
| Gradient clipping | 0.1 |
| Early stopping | patience 5 epochs, minimum improvement 0.002 |
| Checkpoint selection | lowest validation objective; best.pth holds the state dict |
| Validation objective | LPIPS + 0.5*(1 - SSIM) + 5*max(0, new_clipping) |
| Seed / determinism | 42; deterministic data crops derived from epoch and index |
| Scale factor during training | 1 |

**Loss functions.** Six terms are implemented in `training/finetune/losses.py`. The first four are the Zero-DCE definitions; the last two are the project's documented additions. All are differentiable and combined by a weighted sum whose weights are data in the configuration file, not hidden constants.

**Table 6 - Implemented loss terms and their behaviour (VERIFIED from `training/finetune/losses.py`).**

| Term | Weight | Implemented behaviour |
| --- | --- | --- |
| Spatial consistency | 1.0 | On 4x4 average-pooled regions, penalises squared differences between output and input gradient magnitudes in x and y, preserving local contrast |
| Exposure control | 10.0 | Pulls 16x16 local mean luminance toward 0.6 with an absolute-error penalty |
| Colour constancy | 3.5 | Penalises squared differences between the three channel means (grey-world term), reduced from the authors' 5.0 |
| Illumination total variation | 1600.0 | Squared mean absolute horizontal plus vertical variation of the predicted curve map, smoothing the curve field |
| Highlight protection | 5.0 | Smoothstep mask on input luminance (0.70 to 0.90) times ReLU(output - input) luminance, penalising brightening of already-bright pixels |
| Hue preservation | 2.0 | 1 - cosine similarity between per-pixel RGB direction vectors of output and input, preventing colour drift |

**Training data and splits.** Fine-tuning uses the materialised paired LOL splits: 1,565 training pairs and 391 validation pairs (Table 32). The 215 test pairs are used only for leakage checking and are never loaded by the trainer. No supervised L1 objective and no perceptual loss are enabled by default.

> **Status.** The fine-tuning implementation is complete and unit-tested (a one-epoch trainer test uses a randomly initialised checkpoint and stubbed validation metrics), but no fine-tuned checkpoint or training log exists in this repository. The only checkpoint present is the released Zero-DCE++. Any result attributed to a fine-tuned model therefore has no artifact behind it here.

### 2.3 Hybrid integration and adaptive mechanism

**Branches and candidates.** The hybrid stage consumes three float32 RGB candidates of identical size: the original low-light image, the LumenRetinex output, and the Zero-DCE++ output. Both enhancement branches run on the same input and receive the same LumenScan parameters, so the comparison between them is controlled.

**LumenFuse weights.** For each pixel and candidate k, the weight multiplies four factors: local contrast (the absolute Laplacian of the Gaussian-blurred luminance), saturation (the standard deviation across RGB channels), well-exposedness (a Gaussian centred at mid-grey), and a gain-aware noise penalty. The penalty uses the ratio of the blurred candidate luminance to the blurred original luminance, so a candidate that only brightened a region is penalised in proportion to how much it brightened it and how noisy the region already was. The original candidate is multiplied by three wherever the input is already bright, which is the highlight-protection term. Weights are normalised with a 1e-6 offset, refined by a guided filter on the original luminance, then renormalised, with a uniform one-third fallback when a pixel's refined weights vanish.

**Table 7 - LumenFuse weight factors and fusion settings (VERIFIED from `fusion/lumen_fuse.py`).**

| Element | Definition |
| --- | --- |
| Contrast | \|Laplacian(GaussianBlur(Y, sigma=1), ksize=3)\| |
| Saturation | standard deviation of R, G, B at each pixel |
| Exposure | exp(-(Y - 0.5)^2 / (2 * 0.2^2)) |
| Noise penalty | 1 / (1 + (G_eff * sigma_local / tau)^2), G_eff = clip(blur(Y_k,5)/(blur(Y_0,5)+0.02), 1, 30) |
| Highlight protection | original candidate multiplied by 3 where max channel > 0.85 |
| Refinement | guided filter, guide = Y_0, radius 8, epsilon 1e-2, then renormalise |
| Pyramid depth | L = min(5, floor(log2(min(H,W))) - 3), so L+1 levels |
| Blend | sum over candidates of GaussianPyr(weight) * LaplacianPyr(candidate), then collapse and clip |

**Execution order and safeguards.** The seven stages run in the order analysis, classical, deep learning, fusion, denoising, colour correction, quality assessment. Denoising and colour correction act on the fused image, not on the individual candidates, so noise that fusion would have suppressed is never re-injected. The final stage blends the enhanced result with the original using an alpha derived from the median luminance, which means a moderately exposed image is partially self-preserved. Before any of this, the do-no-harm gate can return the input unchanged, and during fusion the original candidate is always available as a safe fallback in bright regions.

**Table 8 - Stage participation by mode (VERIFIED from `enhancement/pipeline.py`).**

| Stage | Classical | Deep learning | Hybrid |
| --- | --- | --- | --- |
| analysis | yes | yes | yes |
| classical_cv | yes | no | yes |
| deep_learning | no | yes | yes |
| fusion | no | no | yes |
| denoising | yes | yes | yes |
| colour_correction | yes | yes | yes |
| quality_assessment | yes | yes | yes |

**Balancing illumination against noise and colour.** Three mechanisms do the balancing. First, the noise penalty is computed from measured local noise rather than a constant, so the fusion weight of an aggressively brightened candidate falls exactly where the input was noisy. Second, EdgeGuard's blend weight is a function of effective gain, so denoising strength follows amplification instead of being uniform. Third, HueLock's shift is bounded to eight LAB units and never touches lightness, so colour correction cannot brighten or darken the image and cannot boost chroma. The alpha blend then limits how far the delivered image can move from the original.

### 2.4 Accelerator execution path

The repository contains a second execution path in which every stage operates on device tensors. A single runtime resolved per pipeline selects the device and precision; `backend/runtime/torch_ops.py` provides device-resident equivalents of the classical primitives (LAB conversion with the OpenCV 5 trilinear LUT, CLAHE with 65,536-bin histograms, exact area resizing, box/Gaussian/Laplacian/pyramid filters, guided and bilateral filtering, Immerkaer noise, quantiles), and `backend/classical_cv/torch_*.py` reimplement each classical stage on those primitives while reusing the CPU modules' rule functions and dataclasses. LumenFuse keeps one algorithm implementation and swaps only its numeric primitives, so coefficients cannot drift between paths. Intermediate tensors are carried on the stage context, and the per-image shared-stage cache replays tensor fields as well as NumPy ones.

**Table 9 - Device and precision policy (VERIFIED from `backend/runtime/config.py`).**

| Setting | Behaviour |
| --- | --- |
| LUMENOIR_DEVICE=auto (default) | CUDA when available, otherwise the CPU reference; MPS is never auto-selected |
| LUMENOIR_DEVICE=cpu | NumPy/OpenCV reference path; PyTorch is never imported |
| LUMENOIR_DEVICE=cuda / cuda:N | Explicit CUDA device; raises when unavailable, never falls back |
| LUMENOIR_DEVICE=mps | Explicit Apple Silicon device; raises when unavailable |
| LUMENOIR_PRECISION=float32 | Default on every device |
| LUMENOIR_PRECISION=float16 | Opt-in on CUDA/MPS; CPU always uses float32 |

This path is documented separately in `docs/gpu_pipeline.md` and is measured in Section 3.3. Two limits are stated plainly there and repeated here: the CUDA branch has never executed on NVIDIA hardware in this repository, and on Apple MPS the accelerator path is slower end-to-end than the CPU reference at the measured sizes.

---

## 3. Experimental Benchmarks & Evaluation

### 3.1 Quantitative benchmarks: status and evidence

The evaluation framework is implemented and its metric definitions are documented, but **no evaluation run has been committed to this repository**. This was verified directly: `training/reports/` contains only a `.gitkeep`, the only CSV and JSON report files are the illustrative placeholders in `training/kaggle_eval/samples/` (whose JSON states that its values are placeholders), and a repository-wide search for the metric values quoted in the brief returns no match.

**Table 10 - Evaluation artifact inventory (VERIFIED by inspection).**

| Path | Content | Usable as a quality result? |
| --- | --- | --- |
| training/reports/ | empty except .gitkeep | No |
| training/kaggle_eval/samples/sample_report.csv | 3 rows, all statistics blank | No - schema example |
| training/kaggle_eval/samples/sample_report.json | explicitly marked illustrative placeholders | No |
| training/splits/ | train.json (1,565), validation.json (391), split_spec.json, split_report.md | Splits only, no metrics |
| training/benchmarks/results/ | does not exist | No |
| visual outputs | no PNG/JPG anywhere outside datasets/ | No |
| failure logs (failures.jsonl/csv) | not present | No |

Consequently the required six-way comparison (original input, classical baseline, pretrained Zero-DCE++, pretrained hybrid, fine-tuned Zero-DCE++, fine-tuned hybrid) cannot be produced from repository evidence. The figures supplied in the project brief are reproduced below exactly as given, clearly separated, and must not be read as verified: they are attributed to a fine-tuned hybrid configuration that has no checkpoint in this repository, and there is no CSV, JSON or log from which to cross-check them.

**Table 11 - UNVERIFIED figures supplied by the project brief (fine-tuned hybrid; no artifact in the repository).**

| Dataset | Paired images | PSNR (dB, higher better) | SSIM (higher better) | LPIPS (lower better) | dE00 (lower better) |
| --- | --- | --- | --- | --- | --- |
| LOL-v1 | 15 | 16.7886 | 0.5157 | 0.4285 | 13.6128 |
| LOL-v2 Real | 100 | 17.8614 | 0.5462 | 0.3835 | 13.0332 |
| LOL-v2 Synthetic | 100 | 17.3289 | 0.7741 | 0.2339 | 13.3937 |

The corresponding cells for the other five configurations (original input, classical baseline, pretrained Zero-DCE++, pretrained hybrid, fine-tuned Zero-DCE++) are not available and are recorded as pending in Section 7. No dataset-level average has been computed across LOL-v1 and LOL-v2 in this report, because the three datasets have different content, resolutions and difficulty and a pooled mean would be misleading.

> The 215 paired test images are consistent with repository records: `training/splits/split_spec.json` specifies 15 LOL-v1 test pairs, 100 LOL-v2 Real and 100 LOL-v2 Synthetic, and `split_report.md` confirms that 215 test pairs were used for leakage checking. The test-set composition is therefore verified even though the metrics are not.

**Metric definitions and directions.** All metrics are implemented in `backend/evaluation/` and used identically by every mode:

**Table 12 - Metric definitions, directions and implementations (VERIFIED from `docs/evaluation.md`).**

| Metric | Direction | Implementation |
| --- | --- | --- |
| PSNR | Higher is better | skimage peak_signal_noise_ratio, data_range = 1.0 |
| SSIM | Higher is better | skimage structural_similarity, channel-averaged |
| LPIPS | Lower is better | lpips with AlexNet backbone, inputs scaled to [-1, 1] |
| CIEDE2000 (dE00) | Lower is better | Mean colour difference between LAB conversions |
| NIQE | Lower is better | pyiqa no-reference metric (reference-free sets) |
| LOE | Lower is better | Lightness Order Error on 50x50 max-RGB maps |
| Edge preservation | Higher is better | Gradient-magnitude correlation on the strongest 10% of edges |
| Halo rate | Lower is better | Gradient-reversal rate on the strongest 5% of input edges |
| New clipping | Lower is better | Change in the share of pixels at or above 0.98 |
| Black crushing | Lower is better | Change in the share of pixels at or below 0.02 |

Each metric is registered once and evaluated by a harness that records a null value plus a note when a dependency, reference or valid input is missing. No metric can silently produce a fabricated number; this behaviour is covered by tests.

### 3.2 Latency benchmarks

Three classes of measurement exist and they must not be conflated: (i) an independent LumenFuse benchmark recorded in the repository, measured on Apple Silicon with PyTorch MPS; (ii) a full-pipeline harness measurement on ten 600x400 images, also local and also MPS; and (iii) figures reported for a Kaggle T4 which have no artifact in this repository. Only (i) and (ii) are verified here.

**Independent LumenFuse benchmark, 2304x1728 (VERIFIED - `docs/fusion_benchmarks.md`):**

**Table 13 - LumenFuse stage latency at 2304x1728 (Apple Silicon MPS; cold start excluded from the median).**

| Backend | Cold (ms) | Median (ms) | P90 (ms) | MAE vs CPU | PSNR (dB) | SSIM |
| --- | --- | --- | --- | --- | --- | --- |
| CPU (OpenCV) | 1013.49 | 840.03 | 863.63 | 0 | infinite | 1.000 |
| torch FP32 (MPS) | 1060.24 | 627.90 | 647.81 | 8.49e-08 | 137.06 | 1.000 |
| torch FP16 (MPS) | 411.29 | 484.15 | 566.58 | 5.57e-04 | 62.13 | 1.000 |

**Table 14 - LumenFuse median latency by size (Apple Silicon MPS, milliseconds).**

| Size | CPU | torch FP32 | torch FP16 |
| --- | --- | --- | --- |
| 256x256 | 14.25 | 31.33 | 29.14 |
| 640x480 | 65.24 | 50.16 | 46.69 |
| 1280x960 | 260.78 | 202.92 | 148.55 |
| 2304x1728 | 840.03 | 627.90 | 484.15 |

At 2304x1728 the MPS path reduces fusion time by about 25 percent in FP32 and 42 percent in FP16 relative to the CPU path on the same machine. At 256x256 it is roughly twice as slow, because transfer and kernel-launch overhead dominate; that asymmetry is the reason the automatic device policy never selects MPS by itself.

**Full pipeline, ten LOL-v2 Real images at 600x400 (VERIFIED - `docs/fusion_benchmarks.md`):**

**Table 15 - Full-pipeline harness comparison, ten images at 600x400 (Apple Silicon MPS fusion).**

| Measurement | CPU fusion | Torch FP32 MPS fusion |
| --- | --- | --- |
| Classical mode, 10 images | 652.5 ms | 653.0 ms |
| Deep-learning mode, 10 images | 509.2 ms | 508.0 ms |
| Hybrid mode, 10 images | 1264.8 ms | 1193.3 ms |
| LumenFuse stage, mean per image | 52.83 ms | 45.23 ms |
| Harness total, shared stages | 6.62 s | 6.54 s |

At 600x400 the fusion change is only a few percent of the harness total, because metric computation, image I/O and report writing dominate. The representative-size measurement in Table 13 is the meaningful one for the accelerator backend. This run also produced an output-equivalence check between the CPU and MPS pipelines over 30 saved hybrid visuals: maximum MAE 2.97e-05, minimum PSNR 69.33 dB, minimum SSIM 0.99999.

**A verification run performed for this report.** The pipeline benchmark in `training/benchmarks/benchmark_pipeline.py` was executed on this machine during report preparation with the real checkpoint, hybrid mode, one warm-up and two timed repeats. These numbers are new measurements, not previously recorded results, and they are the only full-pipeline accelerator comparison in this document.

**Table 16 - Pipeline total, measured for this report (Apple Silicon, torch 2.15.0, synthetic images, median of 2).**

| Device | 640x480 | 1280x960 | 2304x1728 |
| --- | --- | --- | --- |
| CPU float32 | 159.9 ms | 600.9 ms | 1943.6 ms |
| MPS float32 | 346.9 ms | 837.1 ms | 3226.9 ms |

**Table 17 - Per-stage medians at 1280x960 from the same run (milliseconds).**

| Stage | CPU | MPS |
| --- | --- | --- |
| analysis | 52.5 | 109.0 |
| classical_cv | 113.1 | 149.4 |
| deep_learning | 43.7 | 43.4 |
| fusion | 273.1 | 200.6 |
| denoising | 83.1 | 95.6 |
| colour_correction | 25.3 | 220.6 |
| quality_assessment | 10.1 | 18.4 |

On this hardware the accelerator path is slower end-to-end: only fusion and Zero-DCE++ improve, while analysis (quantiles, CLAHE), EdgeGuard and HueLock lose more to kernel-launch overhead and small-tensor dispatch than they gain from the GPU. This is a property of the measured machine and workload, not a defect in the implementation, and it is why the CPU path remains the default for non-CUDA hosts.

**CPU stage timings recorded in the README (VERIFIED, single machine, hybrid mode):**

**Table 18 - Full-pipeline CPU latency by stage (Apple Silicon CPU, milliseconds).**

| Size | Total | Analysis | Retinex | Zero-DCE++ | Fusion | EdgeGuard | HueLock | Final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512x512 | 206.9 | 16.1 | 24.4 | 9.2 | 123.3 | 19.7 | 6.9 | 2.2 |
| 1280x720 | 774.8 | 49.4 | 92.0 | 37.2 | 508.0 | 58.6 | 21.9 | 7.6 |
| 1920x1080 | 1773.6 | 123.4 | 208.7 | 90.0 | 1158.1 | 132.3 | 44.5 | 16.5 |

**Kaggle T4 fusion figures supplied by the brief (UNVERIFIED - no artifact in this repository):**

**Table 19 - UNVERIFIED Kaggle T4 LumenFuse figures at 2304x1728 supplied in the brief.**

| Configuration | Reported time |
| --- | --- |
| CPU | 2068.88 ms |
| CUDA FP32 | 529.19 ms |
| CUDA FP16 | 418.76 ms |

These three values appear nowhere in the repository: there is no benchmark JSON, log or report containing them, and no CUDA device was available while preparing this document. They also do not correspond to the repository's own local MPS numbers (Table 13), which differ in both absolute value and ordering. They are recorded here for completeness and must be regenerated with `training/benchmarks/benchmark_lumen_fuse.py` on a T4 before being cited as a result.

**Trade-offs.** The measurements support three conclusions. Quality and perceptual similarity are essentially preserved by the accelerator path (Section 3.3), so the choice between CPU and GPU is purely a latency question on a given host. Latency is dominated by fusion at large sizes and by analysis at small sizes, so the accelerator only helps the large-image regime. FP16 is materially faster than FP32 on MPS fusion but loosens numerical agreement by roughly three orders of magnitude, which is why it remains opt-in and FP32 remains the default. A full-pipeline CUDA speed-up cannot be claimed: the expanded pipeline has not run on T4.

### 3.3 Numerical parity of the accelerator path

Parity was measured for this report on Apple MPS against the CPU reference, using the repository's own comparison cases. These runs used the same coefficients and the same inputs on both paths; the only difference is the arithmetic backend.

**Table 20 - LumenFuse FP32, CPU reference versus MPS (measured for this report).**

| Case | MAE | Max difference | PSNR | SSIM |
| --- | --- | --- | --- | --- |
| textured | 3.7e-08 | 2.7e-07 | 146.05 dB | 1.00000000 |
| odd 101x77 | 1.23e-06 | 7.9e-05 | 103.67 dB | 0.99999982 |
| colour cast | 1.8e-08 | 1.0e-07 | 152.47 dB | 1.00000000 |
| noisy dark | 2.4e-08 | 1.8e-07 | 149.43 dB | 1.00000000 |

**Table 21 - LumenFuse FP16 on MPS versus the CPU reference (measured for this report).**

| Case | MAE | Max difference | PSNR | SSIM |
| --- | --- | --- | --- | --- |
| textured | 1.40e-04 | 7.85e-04 | 75.02 dB | 0.99998909 |
| odd 101x77 | 1.27e-04 | 6.73e-04 | 75.92 dB | 0.99998951 |
| colour cast | 7.53e-05 | 3.97e-04 | 80.48 dB | 0.99999613 |
| noisy dark | 8.51e-05 | 4.18e-04 | 79.46 dB | 0.99999791 |

**Table 22 - Per-stage parity on a noisy dark image, FP32 MPS (measured for this report).**

| Stage | MAE | Max difference | PSNR | SSIM |
| --- | --- | --- | --- | --- |
| LumenRetinex | 8.4e-05 | 2.9e-02 | 66.05 dB | 0.99999714 |
| EdgeGuard | 3.3e-03 | 2.8e-02 | 47.28 dB | 0.99946982 |
| HueLock | 3.3e-03 | 2.8e-02 | 47.28 dB | 0.99946982 |

LumenScan features agree to float32 round-off: the largest absolute difference is 4.4e-09 on the noise estimate and 1.9e-09 on the median luminance, so the rule set selects identical coefficients on both paths. EdgeGuard's residual follows from its bilateral chroma filter, which uses the textbook kernel rather than OpenCV's exact implementation; HueLock introduces no additional error on an image without a measurable cast.

**Table 23 - Full pipeline, CPU versus MPS, real checkpoint, 320x240 (measured for this report).**

| Mode | MAE | Max difference | PSNR | SSIM |
| --- | --- | --- | --- | --- |
| classical | 9.5e-04 | 1.6e-02 | 54.21 dB | 0.999849 |
| deep_learning | 8.6e-04 | 1.2e-02 | 54.65 dB | 0.997078 |
| hybrid | 9.4e-04 | 1.2e-02 | 54.29 dB | 0.999795 |

> The previously reported figure of a minimum full-pipeline SSIM of 0.99999 did not reproduce. The measured minimum across modes is 0.9971 in deep-learning mode, which is explained by the CPU path quantising the network output through an 8-bit PIL round trip while the tensor path retains float precision. The repository documentation now records the measured value rather than the earlier claim.

### 3.4 Visual outputs and intermediate representations

The pipeline publishes intermediate artifacts as part of its normal response: the original image, the illumination map, the noise map, the classical output, the deep-learning output, the fused output, the enhancement-strength map derived from the fusion weights, and the final image. The frontend renders only artifacts that the backend actually produced. The Kaggle runner and the stage-attribution experiment are both capable of writing before/after and stage-grid PNGs.

However, **no generated image exists in this repository**. A repository-wide search for PNG, JPEG and WebP files outside `datasets/` returns nothing, and `assets/` contains only a README describing future diagrams. No visual comparison can therefore be reproduced in this document. The figures below are numbered, captioned and marked pending so that they can be inserted later without renumbering.

**Table 24 - Figure register. All requested visuals are pending.**

| Figure | Subject | Status |
| --- | --- | --- |
| Figure 1 | Extreme low-light input and enhanced output, side by side | PENDING - no output image exists in the repository |
| Figure 2 | Non-uniform or backlit scene before and after | PENDING - no output image exists |
| Figure 3 | Estimated illumination map | PENDING - the pipeline emits it, no stored sample |
| Figure 4 | Reflectance-like map (input divided by estimated illumination) | PENDING - not persisted by any run |
| Figure 5 | Fusion weights and intermediate stage outputs | PENDING - emitted only during evaluation runs |

*Figure 1 - Extreme low-light input and enhanced output.*  
Status: PENDING. To be produced by POST /api/enhance or the Kaggle runner; no generated image is committed.

*Figure 2 - Non-uniform illumination and backlit scene.*  
Status: PENDING. Requires an evaluation run with save_visuals enabled.

*Figure 3 - Estimated illumination map.*  
Status: PENDING. The analysis stage publishes ILLUMINATION_MAP; no sample has been stored.

*Figure 4 - Reflectance-like map.*  
Status: PENDING. Not persisted by any existing run, and no artifact can be reconstructed without executing the pipeline.

*Figure 5 - Fusion weights and intermediate outputs.*  
Status: PENDING. Stage-attribution tooling writes these; it has not been run into the repository.

### 3.5 Additional ablation experiments - pending

No module-removal ablation has been performed. The preparation exists and is substantial, but preparation is not evidence, and stage attribution alone is not a module-removal ablation: it measures the intermediate stages of the complete pipeline rather than re-running the system with a module disabled.

**Table 25 - Ablation tooling present in the repository, and whether it has produced results.**

| Tool | What it measures | Result available? |
| --- | --- | --- |
| training/experiments/stage_attribution.py | Original, raw Zero-DCE++, classical, fused, fused+EdgeGuard, fused+EdgeGuard+HueLock, final and ground truth, with per-image metrics and latency | No run committed |
| training/kaggle_eval/benchmark.py | Shared-stage caching A/B: total time, per-mode time, execution counts, peak GPU memory, output equivalence | No run committed |
| configs/classical_baseline.json | Original, classical and hybrid modes on the same images | Config only |
| configs/deep_learning.json | Original, deep-learning and hybrid modes on the same images | Config only |
| LumenRetinexConfig.apply_clahe / apply_shoulder | Per-stage switches intended for isolating the gain stage | Available, unused in an ablation |
| LumenRetinexConfig.respect_bypass_gate | Disables the do-no-harm gate to measure raw Retinex | Available, unused in an ablation |
| HueLockConfig.halve_on_spread | Disables the colour-spread guard | Available, unused in an ablation |
| LUMENOIR_PRECISION, LUMENOIR_DEVICE | FP32 versus FP16 and CPU versus accelerator execution | Measured for parity (Section 3.3), not as a quality ablation |

**Additional Ablation Experiments - Pending.** A genuine ablation programme requires disabling each module in turn (Retinex, CLAHE, fusion, EdgeGuard, HueLock, original-candidate highlight protection) and re-measuring the same metrics on the same held-out images, together with the fine-tuned-versus-pretrained comparison. The switches and runners needed to do this exist; the experiments have not been run, and this report does not infer their outcome.

---

## 4. Failure Modes & Limitations

This section separates two different things: limitations that are documented and evidenced in the repository (source code, tests or documentation), and observed failures on real images. The second category requires evidence that does not exist here, and is marked pending rather than invented.

### 4.1 Documented design limitations

**Table 26 - Limitations documented and evidenced in the repository.**

| Limitation | Evidence and technical cause | Status |
| --- | --- | --- |
| Bypass gate can skip a genuinely non-uniform image | `scan.py` gate inspects only p50, dynamic range and highlight share; a bimodal image meeting all three is returned unchanged. Encoded by a test in `tests/test_lumen_retinex.py`. | Design limitation, tested |
| Chroma bilateral differs from OpenCV | `edge_guard.py` uses the textbook bilateral kernel rather than OpenCV's exact weights; measured effect is the EdgeGuard parity residual (MAE 3.3e-03). | Measured |
| Accelerator parity is approximate, not bit-exact | CLAHE binning, LAB LUT interpolation and bilateral weights differ in the last bits; worst observed MAE 1.23e-06 (FP32) and 1.4e-04 (FP16). | Measured |
| MPS is slower end-to-end than CPU | Kernel-launch and small-tensor overhead exceed GPU gains outside fusion; Table 16. | Measured |
| CUDA path unexecuted | No NVIDIA device was available; `torch.cuda.is_available()` is false on all recorded runs. | Unverified |
| uint16 CLAHE round trip on CUDA | Accelerator CLAHE returns a torch.uint16 tensor, a limited-support dtype verified only on CPU and MPS. | Unverified |
| Fusion weights are untuned | Exponents and tau are the documented starting coefficients; no tuning has been run. | Documented |
| Noise penalty is gain-based, not candidate-noise-based | A clean and a noisy candidate with the same effective gain receive the same penalty, by specification. | Documented |
| Global colour shift | HueLock applies one a*/b* shift to the whole frame; L* is untouched but low-chroma highlights and shadows are affected by construction. | Documented |
| Inference does not resize for SICE-scale inputs | The evaluator passes native resolution; 5997x3999 inputs in model modes need a separate protocol. | Documented |
| Remaining closed-loop work | Closed-loop proxy passes and coefficient tuning are not implemented. | Documented |

### 4.2 Observed failure examples - pending

No failure case can be documented from this repository. Reproducing an observed failure requires an actual output image plus the corresponding metric row, and neither exists: there are no saved visuals, no `failures.jsonl`, no per-image metric records, and no halo or clipping measurements from a real run. Describing a plausible artefact as an observed failure would be fabrication.

**Table 27 - Failure categories to be evidenced by the first real evaluation run.**

| Category | Metric that would evidence it | Required artifact |
| --- | --- | --- |
| Halo artefacts | halo_rate, visually at strong edges | Per-image record plus before/after PNG |
| Highlight clipping | new_clipping, highlight_push | Per-image record plus output PNG |
| Colour shift | delta_e00 (paired), hue_shift_deg (unpaired) | Per-image record |
| Noise amplification in extreme shadows | noise_relative, noise_residual | Per-image record, dark subgroup |
| Detail loss | texture_retention, edge_preservation | Per-image record |
| Generalisation failure on unseen data | NIQE, LOE on LIME/MEF/NPE/VV/ExDark/DICM | Unpaired evaluation report |

Once a run exists, each item should be documented with the image identifier, the observed degradation, the plausible technical cause and a proposed mitigation, and the mitigation must be labelled as proposed rather than demonstrated.

### 4.3 Research and evaluation limitations

- **Evaluation scope.** No dataset-level quantitative result exists. The metric framework, the runner and the split files are complete; the measurements are not.
- **Dataset coverage.** Paired evaluation is limited to LOL-v1, LOL-v2 Real and LOL-v2 Synthetic (215 test pairs). SICE, ExDark, DICM, LIME, MEF, NPE and VV are supported by the runner but have not been evaluated. SICE Part 1 was part of the model's pretraining data, so paired SICE results could not be presented as held out.
- **Fine-tuning generalisation.** Unknown by construction: no fine-tuned checkpoint exists, so neither its in-domain gain nor its behaviour on unseen illumination has been observed.
- **GPU hardware validation.** CUDA is untested. The accelerator path is verified on Apple MPS and CPU only, and the MPS path is slower than CPU end-to-end at the measured sizes.
- **Benchmark limitations.** All recorded latency numbers come from one Apple Silicon machine with synthetic or ten-image workloads. They are not T4 predictions, they are not a multi-user server benchmark, and they exclude network transfer.
- **Colour-space and library dependence.** Metric values depend on library versions; comparisons are only valid within one environment, which the runner documents in `run_config.json`.

---

## 5. Deployment & Quickstart Guide

### 5.1 Implemented stack

**Table 28 - Verified technology stack.**

| Layer | Implementation | Evidence |
| --- | --- | --- |
| Frontend | React 19 + TypeScript + Tailwind CSS 4, built with Vite 8; lint via oxlint | frontend/package.json |
| Backend | Python 3.11+ (verified on 3.14), FastAPI + Uvicorn | requirements.txt, README.md |
| Image processing | NumPy and OpenCV (headless) | requirements.txt |
| Deep learning | PyTorch 2.4+ (optional, lazy), Zero-DCE++ NoirNet | requirements-inference.txt |
| Metrics | scikit-image; optional lpips and pyiqa | requirements-eval.txt, requirements-metrics.txt |
| Container | python:3.12-slim image running uvicorn, backend only | Dockerfile |
| State | Stateless: no database, no authentication, no image storage | docs/architecture.md |

The API is deliberately thin. Routes validate and delegate: upload validation and decoding live in `backend/preprocessing/`, orchestration in `backend/enhancement/`, model lifecycle in `backend/inference/`. Every request is independent, uploaded bytes are never written to disk, and the pipeline runs in a worker thread so the event loop is not blocked. The model is loaded once at startup and inference is serialised inside the manager.

### 5.2 API surface

**Table 29 - Implemented HTTP endpoints (VERIFIED from `backend/api/`).**

| Method | Path | Purpose |
| --- | --- | --- |
| GET | /api/health | Service status, model state and readiness |
| GET | /api/images/limits | Accepted formats, size and dimension limits |
| POST | /api/images/upload | Validate an image, return information and statistics |
| GET | /api/enhance/status | Stage, mode, model, artifact and inference-setting availability |
| POST | /api/enhance | Run the pipeline and return a base64 PNG at the input dimensions |
| GET | /api/evaluation/metrics | Declared metrics, reference requirements and dependencies |

Rejections always use a stable envelope of the form `{ ok: false, error: { code, message } }`. Invalid uploads keep their 4xx codes, an unavailable model or device returns 503 with a typed code, and an unexpected stage failure returns 500.

### 5.3 Runtime and checkpoint requirements

- Python 3.11 or newer (the development environment uses 3.14), plus Node 20+ for the frontend.
- Classical mode: `requirements.txt` only. No PyTorch, no checkpoint, no API keys.
- Deep-learning and hybrid modes: `requirements-inference.txt` plus a checkpoint at `backend/weights/zero_dce_plus_plus.pth` and `LUMENOIR_MODEL_ID=zero_dce_plus_plus`. The checkpoint is git-ignored and must be provisioned separately; the repository documents the exact upstream URL and SHA-256.
- The evaluation dependencies (`requirements-eval.txt`, optionally `requirements-metrics.txt`) are needed only for dataset work, metrics and the Kaggle runner.
- No secrets are required or committed; `.env` and `.env.*` are git-ignored.

### 5.4 Environment variables

**Table 30 - Documented environment variables (VERIFIED from `.env.example` and `backend/config.py`).**

| Variable | Default | Purpose |
| --- | --- | --- |
| LUMENOIR_DEVICE | auto | Pipeline backend: auto (CUDA else CPU), cpu, mps, cuda, cuda:N |
| LUMENOIR_PRECISION | float32 | float32 or float16 on CUDA/MPS; CPU is always float32 |
| LUMENOIR_MODEL_ID | (empty) | Registered model id; zero_dce_plus_plus enables the network |
| LUMENOIR_MODEL_WEIGHTS | (empty) | Explicit checkpoint path |
| LUMENOIR_WEIGHTS_DIR | backend/weights | Directory searched for <model_id>.pth |
| LUMENOIR_INFERENCE_DEVICE | follows LUMENOIR_DEVICE | Device used for model inference |
| LUMENOIR_INFERENCE_PRECISION | follows LUMENOIR_PRECISION | Model inference precision |
| LUMENOIR_EVAL_DEVICE | cpu | GPU PSNR/SSIM selection for evaluation |
| LUMENOIR_ZERODCE_SCALE_FACTOR | 12 | Zero-DCE++ downscale factor, 1 to 12 |
| LUMENOIR_INFERENCE_MAX_EDGE | 1600 | Longest edge used for inference |
| LUMENOIR_INFERENCE_MAX_PIXELS | 4000000 | Hard processing ceiling per request |
| LUMENOIR_MAX_UPLOAD_BYTES | 10485760 | Maximum accepted upload size |
| LUMENOIR_CORS_ORIGINS | localhost:5173, 127.0.0.1:5173 | Allowed browser origins |

### 5.5 Quickstart (commands taken from the repository, not invented)

```bash
# 1. Backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000

# 2. Frontend (second terminal)
cd frontend
npm install
npm run dev

# 3. Optional: enable the pretrained Zero-DCE++
pip install -r requirements-inference.txt
LUMENOIR_MODEL_ID=zero_dce_plus_plus \
LUMENOIR_INFERENCE_DEVICE=cpu \
uvicorn backend.main:app --reload --port 8000

# 4. Checks
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check backend tests training
cd frontend && npm run build && npm run lint
```

The API listens on `http://localhost:8000` (health at `/api/health`, interactive docs at `/docs`) and the development frontend on `http://localhost:5173`, proxying `/api/*` to the backend so no URL configuration is needed. There is no `run_inference.py` script in this repository; inference is reached through the API, the pipeline benchmark or the Kaggle runner, and those are the commands used above.

### 5.6 Docker

```bash
docker build -t lumenoir-api .
docker run --rm -p 8000:8000 lumenoir-api
```

The image installs `requirements.txt` and copies `backend/`. It intentionally excludes PyTorch and the checkpoint, so it serves classical mode out of the box and requires `requirements-inference.txt` plus a mounted checkpoint for the other two modes. The frontend is a separate static build.

### 5.7 Deployment status

> **Pending.** No public deployment URL exists in the repository or its documentation, and none was observed during this inspection. The brief's target of roughly ten seconds per image is supported by the CPU measurements in Table 18 for typical image sizes, but a live end-to-end measurement including upload and network transfer has not been made. The deployment platform has not been chosen in the repository.

---

## 6. References & Reproducibility

### 6.1 Module reference

**Table 31 - Principal modules, key entry points and responsibilities.**

| Path | Key entries | Responsibility |
| --- | --- | --- |
| backend/enhancement/pipeline.py | EnhancementPipeline.run | Stage ordering, per-mode stage sets, timing, failure isolation, capability reporting |
| backend/enhancement/stages.py | AnalysisStage ... QualityAssessmentStage | Thin orchestration over the algorithm modules; artifact publication |
| backend/enhancement/shared.py | SharedStageCache | Per-image reuse of analysis, classical and deep-learning results across modes |
| backend/classical_cv/scan.py | scan, LumenScanConfig, LumenScanFeatures | Measurements and the rule set that selects every coefficient |
| backend/classical_cv/retinex.py | enhance, LumenRetinexConfig | Illumination decomposition, adaptive gain, shoulder, CLAHE on L* |
| backend/classical_cv/edge_guard.py | denoise, EdgeGuardConfig | Gain-aware luma guided filter and chroma bilateral filter |
| backend/classical_cv/hue_lock.py | correct, HueLockConfig | Bounded LAB colour-cast correction |
| backend/classical_cv/illumination.py | estimate_illumination | Max-RGB prior with guided smoothing |
| backend/classical_cv/filters.py | box_filter, guided_filter | Single implementation of both filters |
| backend/classical_cv/noise.py | noise_sigma, noise_map | Immerkaer estimator and local noise map |
| backend/fusion/lumen_fuse.py | fuse, fuse_tensors, compute_weights | Noise-aware weight computation, guided refinement, pyramid blend |
| backend/fusion/torch_backend.py | TorchFusionOps | Accelerator primitives for the shared fusion core |
| backend/deep_learning/zero_dce_net.py | ZeroDCEPlusPlusNet | Zero-DCE++ architecture |
| backend/deep_learning/zero_dce_plus_plus.py | ZeroDCEPlusPlusModel | Checkpoint validation, registration, PIL and tensor inference |
| backend/inference/manager.py | ModelManager, infer, infer_tensor | Load-once lifecycle, serialisation, timing, typed errors |
| backend/runtime/config.py | resolve_runtime, PipelineRuntime | Device and precision resolution, CPU fallback policy |
| backend/runtime/torch_ops.py | TorchImageOps | Device-resident image primitives |
| backend/evaluation/harness.py | evaluate_arrays, evaluate_pair | Metric execution with null-plus-note failures |
| backend/evaluation/reference_metrics.py | compute_psnr, compute_ssim, compute_lpips, compute_delta_e00 | Reference-based metrics |
| backend/evaluation/gpu_metrics.py | compute_psnr, compute_ssim | GPU PSNR/SSIM selected by LUMENOIR_EVAL_DEVICE |
| training/kaggle_eval/runner.py | runner | Dataset discovery, four-mode evaluation, reports, visuals |
| training/experiments/stage_attribution.py | stage attribution | Intermediate-stage capture and per-stage metrics |
| training/finetune/trainer.py | trainer | Fine-tuning loop, checkpoint selection, resumption |
| training/benchmarks/benchmark_lumen_fuse.py | CLI | Independent fusion benchmark |
| training/benchmarks/benchmark_pipeline.py | CLI | Full-pipeline CPU/accelerator benchmark |

### 6.2 Dataset splits

**Table 32 - Materialised paired splits (VERIFIED from `training/splits/split_report.md`).**

| Dataset | Source pairs | Eligible | Train | Validation | Test |
| --- | --- | --- | --- | --- | --- |
| LOL-v1 | 485 | 389 | 311 | 78 | 15 |
| LOL-v2 Real | 689 | 667 | 534 | 133 | 100 |
| LOL-v2 Synthetic | 900 | 900 | 720 | 180 | 100 |
| Total | 2074 | 1956 | 1565 | 391 | 215 |

The split uses seed 42 and a 20 percent validation fraction. An automated leakage audit compared candidate train/validation pairs against the 215 test pairs and excluded 118 leaking pairs; 733 duplicate-connected pairs were kept together in one split rather than deleted. The report records zero overlap between the three sets. Unpaired sets available to the runner but not evaluated are LIME (10), MEF (17), NPE with its three extensions (85), VV (24), DICM (64) and ExDark.

### 6.3 Checkpoints

**Table 33 - Checkpoint inventory.**

| Checkpoint | State | Notes |
| --- | --- | --- |
| Pretrained Zero-DCE++ (52,395 bytes, 10,561 parameters) | Present, git-ignored | SHA-256 ca8855b9...; strict load verified; used by all recorded runs |
| Fine-tuned checkpoint (best.pth) | Absent | The trainer writes it; no training run has been committed |
| Training state (last_training.pt) | Absent | Resumption state; not produced |
| ONNX export | Not implemented | Documented as future work |

### 6.4 Configuration files

- `training/finetune/configs/zero_dce_plus_plus_lol.json` - fine-tuning: optimizer, schedule, epochs, crops, loss weights, early stopping.
- `training/kaggle_eval/configs/classical_baseline.json` - original, classical and hybrid modes.
- `training/kaggle_eval/configs/deep_learning.json` - original, deep-learning and hybrid modes.
- `training/kaggle_eval/example_config.json` - documented configuration template.
- `training/splits/split_spec.json` - split rules, seeds and expected counts per dataset.
- `.env.example` - every supported environment variable with defaults.
- `pytest.ini`, `ruff.toml` - test selection (integration marker excluded by default) and lint configuration.

### 6.5 Evaluation and benchmark commands

```bash
# Kaggle evaluation (four modes, manifests, records, aggregates, visuals)
python -m training.kaggle_eval --config /kaggle/working/eval_config.json

# Shared-stage caching A/B benchmark
python -m training.kaggle_eval.benchmark \
  --config /kaggle/working/eval_config.json --images 10

# Stage attribution on the validation split
python -m training.experiments.stage_attribution \
  --validation-manifest training/splits/validation.json \
  --data-root datasets/raw --output-dir /tmp/stage-attribution \
  --weights backend/weights/zero_dce_plus_plus.pth --metrics psnr,ssim,lpips,delta_e00

# Independent fusion benchmark
python -m training.benchmarks.benchmark_lumen_fuse --device auto --warmups 2 --runs 10

# Full-pipeline CPU/accelerator benchmark
python -m training.benchmarks.benchmark_pipeline \
  --devices cpu,cuda --precision float32 \
  --sizes 640x480,1280x960,2304x1728 --warmup 1 --repeat 3

# Test suite and lint
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check backend tests training
```

These are the commands the repository documents and the ones used for the measurements in this report. The Kaggle T4 counterparts, including the CUDA-specific failure modes to watch, are recorded in `docs/gpu_pipeline.md`.

### 6.6 Hardware and environment used

**Table 34 - Environment for the measurements in this report.**

| Item | Value |
| --- | --- |
| Machine | Apple Silicon (M-series), macOS, arm64 |
| Python / PyTorch | 3.14.3 / 2.15.0 |
| OpenCV / NumPy | 5.0.0 / 2.5.4 |
| CUDA | Unavailable (torch.cuda.is_available() is false) |
| Apple MPS | Available |
| Test suite | 394 passed, 3 skipped, 8 deselected |
| Lint | Ruff 0.17.4, zero findings across backend, tests and training |

### 6.7 Metric definitions and reproducibility rules

- Arrays are float32 RGB in [0, 1]; PSNR and SSIM use data_range = 1.0 and are computed with scikit-image; SSIM is channel-averaged.
- LPIPS uses the AlexNet backbone with inputs scaled to [-1, 1]; NIQE uses pyiqa. Both report relative values within one environment rather than MATLAB-comparable absolutes.
- CIEDE2000 is the mean colour difference between LAB conversions of output and reference.
- Missing dependency, missing reference or invalid input yields a null value plus a note; the harness has no path that fabricates a number.
- The runner writes the configuration, seed, git commit, resolved device and model status into `run_config.json`, so any future report can be traced to its exact conditions.
- Paired and unpaired datasets are reported separately, and PSNR/SSIM/LPIPS are never computed for unpaired sets.
- No parameter is tuned on evaluation data; test records are excluded from every training and tuning path.

---

## 7. Pending Work Register

**Table 35 - Pending items. Nothing in this table is complete.**

| Item | Status | Evidence / required action |
| --- | --- | --- |
| LOL quantitative evaluation (six-way comparison, Section 3.1) | PENDING | No CSV/JSON/metrics exist (`training/reports/` empty). Run `training.kaggle_eval` on the 215 test pairs for all four modes, then fine-tune and re-run. |
| Cross-check of the brief's fine-tuned hybrid figures | PENDING | Table 11 reproduced verbatim; no artifact to verify against. Requires the fine-tuned checkpoint and its evaluation CSVs. |
| Fine-tuned Zero-DCE++ checkpoint | PENDING | No `best.pth` or training log in the repository. Run `training/finetune` on the 1,565/391 split. |
| Full-pipeline CUDA / Kaggle T4 validation | PENDING | No NVIDIA hardware available; CUDA code paths unexecuted. Run the T4 commands in `docs/gpu_pipeline.md`. |
| Kaggle T4 fusion numbers (Table 19) | UNVERIFIED | Supplied by the brief; no benchmark artifact. Re-run `benchmark_lumen_fuse.py --device cuda` on a T4. |
| Genuine module-removal ablations | PENDING | Stage attribution is not a module-removal ablation. Tooling exists (Table 25); experiments not run. |
| Classical vs deep-learning vs hybrid comparison tables | PENDING | Configs exist (`classical_baseline.json`, `deep_learning.json`); no results. |
| Fusion-weight and alpha-blend analysis | PENDING | Weights are emitted during diagnostics and the strength map is published; no stored analysis exists. |
| Visual comparisons (Figures 1-5) | PENDING | No PNG/JPEG exists outside `datasets/`. Produce via the API or the runner with `save_visuals`. |
| Observed failure examples | PENDING | No `failures.jsonl`, per-image records or output images. Document after the first real run. |
| Unseen-dataset evaluation (SICE, ExDark, DICM, LIME, MEF, NPE, VV) | PENDING | Runner supports them; no run. SICE Part 1 is not held out for this model and must be labelled accordingly. |
| Public deployment URL | PENDING | No URL in the repository; platform not chosen. Docker image exists for the backend. |
| End-to-end latency including upload and network | PENDING | Only in-process CPU/accelerator timings exist (Tables 13-18). |
| Team name, members and USNs | PENDING | Not recorded in the repository; placeholders retained on the cover. |
| Confirmation of problem ID 26035 | PENDING | The identifier does not appear anywhere in the repository. Confirm against the official problem statement. |
| Coefficient tuning and closed-loop proxy passes | PENDING | Documented as unimplemented; coefficients are the untuned starting values. |

## Appendix A. Verification Log

The following checks were executed while preparing this report, on the machine described in Table 34. No source file, model weight or dataset was modified.

**Table 36 - Checks executed for this report.**

| Check | Command | Result |
| --- | --- | --- |
| Lint | ruff check backend tests training | All checks passed (0 findings) |
| Full test suite | python -m pytest -q | 394 passed, 3 skipped, 8 deselected |
| GPU parity subset | pytest tests/test_gpu_ops.py tests/test_lumen_fuse_torch.py | 54 passed, 3 skipped |
| Shared-stage cache | pytest tests/test_shared_stage_cache.py | 9 passed |
| Device resolution | resolve_runtime(...) for auto/cpu/mps/cuda | auto to CPU, MPS explicit, CUDA raises, FP32 default |
| CPU-only operation | API request with torch imports blocked | HTTP 200; torch never imported |
| Cache replay on MPS | two modes with one SharedStageCache | bit-identical to uncached runs |
| Pipeline benchmark | python -m training.benchmarks.benchmark_pipeline | Tables 16-17 |
| Evidence search | repository-wide grep for brief metrics and CSVs | No matches; no evaluation artifacts |
| Image inventory | find for PNG/JPEG/WebP outside datasets | None found |

Skipped and deselected tests are environmental, not suppressed failures: the CUDA-only memory test, the accelerator-only branches under the CPU parameterisation, and the eight tests marked `integration` (which require the real checkpoint or LPIPS/NIQE weights) excluded by `pytest.ini`.
