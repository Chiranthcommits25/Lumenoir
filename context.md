LUMENOIR

Robust Low-Light Image Restoration and Enhancement

1. PROJECT OVERVIEW

Build Lumenoir, an intelligent, illumination-adaptive image restoration and enhancement application.

The objective is to develop a lightweight, efficient and intelligent system that enhances severely underexposed, noisy and colour-distorted images while preserving fine textures, edges and structural information.

This must be more than a basic brightness adjustment application.

The system must intelligently analyse an uploaded image, understand its illumination and quality characteristics, select suitable enhancement techniques and produce a natural-looking enhanced image.

The application must be:

* Computationally efficient.
* Lightweight enough for affordable cloud deployment.
* Optimised for fast image processing.
* Fully responsive on mobile phones, tablets and desktops.
* Visually polished, following Modify.app as the primary design reference.
* Reliable when processing previously unseen images.

Product name: LUMENOIR.

⸻

2. COMPLETE PROBLEM STATEMENT

Problem Statement 1 (P1):

Robust Low-Light Image Restoration and Enhancement.

Challenge:

How can we design an intelligent, illumination-adaptive image restoration system that enhances severely underexposed, noisy and colour-distorted images while preserving fine textures, edges and structural information across different lighting conditions?

The system must handle:

1. Extremely dark environments.
2. Non-uniform illumination.
3. Sensor noise.
4. Colour distortion.
5. Loss of fine details.
6. Overexposed regions after enhancement.
7. Halo artefacts.
8. Unnatural colour reproduction.

The system must go beyond simply increasing brightness.

It must estimate illumination, suppress noise, restore colour fidelity and preserve structural details.

A. Classical Computer Vision

Mandatory techniques:

1. Illumination estimation and decomposition.
2. Retinex-based enhancement.
3. Contrast Limited Adaptive Histogram Equalization (CLAHE).
4. Edge-preserving denoising using bilateral or guided filtering.
5. Contrast correction.
6. Colour correction.
7. Adaptive parameter selection based on image characteristics.

B. Deep Learning

Requirements:

1. Implement or adapt an appropriate low-light enhancement architecture.
2. Consider Zero-DCE, Zero-DCE++, RetinexNet or another suitable architecture.
3. Fine-tune or adapt the model wherever suitable training data is available.
4. Demonstrate generalisation across different illumination conditions.
5. Include an image-quality-aware enhancement strategy.

C. Hybrid Intelligence

Integrate classical computer vision with deep learning.

The system must:

* Combine classical illumination estimation with deep learning restoration.
* Adaptively select or combine enhancement outputs.
* Preserve edges and textures.
* Minimise noise amplification.
* Handle extremely dark regions.
* Handle non-uniform illumination.

D. Advanced Requirements

1. Adaptive enhancement instead of fixed parameters.
2. Compare classical-only, deep-learning-only and hybrid approaches.
3. Perform ablation studies.
4. Minimise overexposure, colour shifts and detail loss.
5. Test on previously unseen images.

⸻

3. PROVIDED RESEARCH RESOURCES

The hackathon has already provided these resources:

* LOL Dataset.
* LOL-v2 Dataset.
* Zero-DCE Research Implementation.
* MIT-Adobe FiveK Dataset.

Do not waste time searching for replacements without a technical reason.

Study the provided resources and determine the most suitable dataset combination.

Define:

* Training dataset.
* Validation dataset.
* Testing dataset.
* Preprocessing strategy.
* Proper train/validation/test separation.
* Generalisation testing strategy.

Avoid data leakage.

⸻

4. HACKATHON EVALUATION RUBRIC

These judging criteria must influence every major development decision.

A. Evaluation Metrics – 25 Marks

Target:
Performance metrics should be competitive with the top 10% of participants.

Required evaluation:

* PSNR.
* SSIM.
* LPIPS.
* NIQE or another suitable no-reference metric.
* Processing latency.
* Noise suppression.
* Edge preservation.

Important:
PSNR and SSIM require suitable ground-truth reference images.

Do not optimise one metric at the expense of overall perceptual quality.

B. Implementation – 20 Marks

Top-band requirements:

* Core features work on unseen inputs.
* Proper training/validation split.
* Clean and modular code.
* Invalid and corrupted inputs handled correctly.

The application must not rely on a small collection of demonstration images.

C. UI/UX and Deployment – 15 Marks

Top-band requirements:

* Public deployment URL.
* Works on first click from phones and laptops.
* Response time of approximately 10 seconds.
* Clear interface.
* Loading states.
* Error states.
* Repository runs using one command or provides a Dockerfile.
* No committed secrets or API keys.

D. Project Documentation – 15 Marks

Top-band requirements:

* Architecture diagram matches actual implementation.
* Design document includes every required template section.
* README explains how to reproduce the results.

Documentation must reflect the actual system, not an imaginary architecture.

Note: These are the four supplied judging categories. Do not assume additional scoring categories without receiving the remaining rubric.

⸻

5. TECHNICAL ARCHITECTURE

Before implementation, propose an architecture with the following components.

A. Frontend

Preferred:

* React.
* TypeScript.
* Tailwind CSS.
* shadcn/ui.

Design reference:
Modify.app.

The visual experience should be clean, premium, minimal and image-focused.

B. Backend

Preferred:

* Python.
* FastAPI.
* OpenCV.
* PyTorch.

Keep the image-processing pipeline separate from the frontend.

C. Deep Learning

Investigate Zero-DCE++ and the provided research implementation.

Prioritise:

* Lightweight architecture.
* Small model size.
* Low inference latency.
* Low memory consumption.
* CPU inference compatibility.
* Easy deployment.

Do not choose a large model simply because it produces visually impressive results.

Compare suitable architectures based on actual model size, inference speed, restoration quality and deployment requirements.

D. Training

Use Kaggle GPU or Google Colab for training/fine-tuning.

Training should happen remotely.

Save the final trained weights as a .pth file.

The deployed backend must load the trained weights and perform inference independently.

Do not require a continuously running Kaggle or Colab session.

⸻

6. PERFORMANCE OPTIMISATION – VERY IMPORTANT

Before writing the actual implementation, analyse the computational cost of every stage.

The goal is to achieve high-quality restoration with minimum unnecessary computation.

Do not implement expensive operations everywhere simply because they are available.

A. Image Analysis Before Enhancement

Analyse the uploaded image first.

Calculate inexpensive image characteristics such as:

* Average brightness.
* Brightness distribution.
* Contrast.
* Approximate noise level.
* Colour distribution.
* Illumination uniformity.

Use these characteristics to determine which enhancement operations are necessary.

Avoid executing every operation at maximum strength on every image.

B. Adaptive Processing

Design efficient decision-making scenarios.

Example scenarios:

SCENARIO 1 – Slightly Underexposed Image

* Mild illumination correction.
* Limited contrast enhancement.
* Avoid unnecessary aggressive denoising.
* Preserve natural colours.

SCENARIO 2 – Severely Dark Image

* Stronger illumination correction.
* Appropriate Retinex enhancement.
* Deep learning restoration.
* Controlled noise reduction.
* Protect existing bright regions.

SCENARIO 3 – Noisy Low-Light Image

* Estimate noise level.
* Apply suitable edge-preserving denoising.
* Avoid excessive smoothing.
* Preserve important textures.

SCENARIO 4 – Non-Uniform Lighting

* Estimate spatial illumination distribution.
* Apply region-aware enhancement.
* Avoid brightening already illuminated regions unnecessarily.

SCENARIO 5 – Colour-Distorted Image

* Analyse colour imbalance.
* Apply controlled colour correction.
* Prevent unnatural saturation.

SCENARIO 6 – Image With Bright and Dark Regions

* Identify excessively bright regions.
* Protect highlights.
* Enhance darker regions independently.
* Avoid overexposure.

These scenarios are initial engineering guidelines, not fixed rules. Validate and refine them using experiments.

C. Avoid Unnecessary Computation

Implement:

1. Reuse intermediate calculations wherever possible.
2. Avoid repeated colour-space conversions.
3. Avoid running multiple expensive filters unnecessarily.
4. Load the deep learning model once when the backend starts.
5. Process one image at a time initially.
6. Use efficient NumPy and OpenCV operations.
7. Avoid unnecessary model inference.
8. Avoid excessive image resizing.
9. Preserve aspect ratio.
10. Release temporary memory after processing.

Use image resizing carefully.

If processing at a lower resolution improves speed, determine whether the quality loss is acceptable.

The final output should retain an appropriate resolution.

⸻

7. LIGHTWEIGHT MODEL AND DEPLOYMENT REQUIREMENTS

This is a major project constraint.

We have limited resources and want affordable deployment.

Prioritise a model that:

* Has relatively few parameters.
* Has a small checkpoint.
* Supports CPU inference.
* Requires manageable RAM.
* Produces acceptable results within the latency target.
* Can run without a dedicated cloud GPU.

Investigate Zero-DCE++ first.

Before selecting the final model, measure:

1. Number of parameters.
2. Checkpoint size.
3. CPU inference time.
4. Peak memory consumption.
5. GPU inference time.
6. Restoration quality.

Do not assume a model is lightweight merely because its architecture is described as lightweight.

Verify actual performance.

Consider optimisations such as:

* PyTorch inference mode.
* Reduced precision where supported.
* ONNX Runtime if beneficial.
* Model optimisation and quantisation if quality remains acceptable.

Do not introduce optimisations that damage restoration quality.

The application should be able to operate using affordable CPU-based cloud hosting if the benchmarks support it.

⸻

8. INNOVATION FEATURES

Design the architecture so that the following features can be integrated modularly.

1. Region-Aware Adaptive Enhancement

Enhance different image regions according to their illumination characteristics.

2. Noise-Aware Intelligent Denoising

Adjust denoising strength based on estimated noise characteristics.

3. Automatic Quality-Based Output Selection

Evaluate available enhancement outputs and select the most suitable result using a validated quality-assessment strategy.

4. Explainable AI Enhancement Dashboard

Visualise:

* Original image.
* Illumination map.
* Noise map.
* Enhancement strength map.
* Classical CV output.
* Deep learning output.
* Final hybrid output.

5. Automatic Overexposure Protection

Identify bright regions and prevent unnecessary enhancement.

6. Edge and Texture Preservation

Protect structural information during denoising and enhancement.

These are planned innovation modules.

Do not claim that they are already implemented.

Build the baseline pipeline first and integrate these features in stages.

⸻

9. IMAGE PROCESSING PIPELINE

Propose an efficient processing sequence.

Expected high-level workflow:

1. User uploads image.
2. Validate file type, size and dimensions.
3. Preprocess the image.
4. Analyse illumination, contrast, colour and noise.
5. Estimate image characteristics.
6. Generate classical CV enhancement output.
7. Generate deep learning enhancement output when required.
8. Apply adaptive fusion.
9. Apply necessary quality-preservation operations.
10. Generate final restored image.
11. Calculate evaluation metrics when reference images are available.
12. Return output to the frontend.

Important:

The final implementation may use a different execution order if research and benchmarking justify it.

Avoid unnecessary duplicate processing.

The architecture should remain modular.

⸻

10. UI/UX DESIGN REQUIREMENTS

The website must follow Modify.app as its primary visual design reference.

Do not create a generic AI chatbot interface.

Create a premium image-restoration workspace.

Design direction

* Minimal interface.
* Dark charcoal or carefully balanced neutral background.
* Clean typography.
* Subtle accent colours.
* Smooth but lightweight animations.
* Clear image previews.
* Professional dashboard.
* No unnecessary visual clutter.

Required screens

1. Landing page.
2. Image upload workspace.
3. Before/after comparison.
4. Enhancement settings.
5. Classical CV output.
6. Deep learning output.
7. Hybrid output.
8. Explainable AI visualisations.
9. Evaluation dashboard.
10. Download section.

Before/After Comparison

This is a major interaction.

Provide an interactive comparison slider.

Users should be able to inspect the original and enhanced image at different positions.

Mobile Optimisation

Mobile responsiveness is mandatory.

The application must work properly on:

* Desktop computers.
* Laptops.
* Tablets.
* Android phones.
* iPhones.

On mobile:

* Use a single-column layout where appropriate.
* Make upload controls touch-friendly.
* Avoid horizontal overflow.
* Keep comparison controls usable with touch.
* Scale image previews to the available screen width.
* Make buttons sufficiently large.
* Keep loading and error messages visible.
* Ensure evaluation charts remain readable.
* Avoid heavy animations.
* Avoid unnecessary large assets.

Do not simply shrink the desktop design.

Design the mobile experience intentionally.

⸻

11. LATENCY AND RESPONSE TIME

The judging target is approximately 10 seconds.

Treat this as an engineering requirement.

Measure actual end-to-end latency, including:

* Image upload.
* Preprocessing.
* Image analysis.
* Classical CV.
* Model inference.
* Adaptive fusion.
* Output generation.
* Response transfer.

Optimise the pipeline based on measurements.

Do not promise 10-second processing without testing.

Display a clear processing indicator.

If processing takes longer, show progress or a meaningful status.

Handle timeouts and failures gracefully.

⸻

12. DEPLOYMENT

The deployed application should:

* Have a publicly accessible URL.
* Work on mobile and desktop.
* Load reliably.
* Process uploaded images.
* Return downloadable enhanced images.
* Handle invalid uploads.
* Avoid exposing API keys.
* Avoid storing user images unnecessarily.

Evaluate affordable deployment options such as Railway and Hugging Face Spaces.

Choose the hosting platform based on actual RAM, CPU, model size, latency and cost.

Do not assume that a free hosting plan can run the final model without benchmarking.

Provide:

* requirements.txt.
* Dockerfile if appropriate.
* Environment configuration.
* README.
* Deployment instructions.

⸻

13. EVALUATION AND EXPERIMENTS

Implement three independently measurable approaches.

A. Classical CV only.

B. Deep learning only.

C. Hybrid enhancement.

Evaluate using:

* PSNR.
* SSIM.
* LPIPS.
* NIQE.
* Latency.
* Noise suppression.
* Edge preservation.

Perform ablation studies.

Suggested experiments:

1. Complete hybrid system.
2. Without Retinex.
3. Without CLAHE.
4. Without denoising.
5. Without colour correction.
6. Without adaptive fusion.
7. Without region-aware enhancement.
8. Without noise-aware denoising.
9. Without overexposure protection.

Choose the final experiments based on the actual implemented modules.

Maintain consistent test data and evaluation conditions.

Include previously unseen images.

Document failure cases.

⸻

14. CODE QUALITY AND PROJECT STRUCTURE

Use a modular architecture.

Suggested structure:

LUMENOIR/
│
├── frontend/
│   ├── src/
│   ├── components/
│   └── pages/
│
├── backend/
│   ├── api/
│   ├── classical_cv/
│   ├── deep_learning/
│   ├── enhancement/
│   ├── fusion/
│   ├── evaluation/
│   ├── preprocessing/
│   ├── weights/
│   └── main.py
│
├── training/
│   ├── kaggle_notebooks/
│   └── training_config/
│
├── tests/
│
├── docs/
│
├── requirements.txt
├── Dockerfile
└── README.md

Adjust this structure if research or deployment requires it.

Keep individual modules independently testable.

Avoid placing the entire project inside one large Python file.

⸻

15. DEVELOPMENT RULES FOR THE AI CODING AGENT

Follow these rules throughout development.

1. Do not start coding the entire application immediately.
2. First inspect the problem statement and provided research resources.
3. Produce a complete technical architecture.
4. Identify computational bottlenecks.
5. Compare lightweight model options.
6. Define image-processing scenarios.
7. Define the backend/frontend communication.
8. Define deployment requirements.
9. Identify potential memory limitations.
10. Present the proposed implementation plan before writing code.

Once the architecture is approved:

* Implement one module at a time.
* Test every module independently.
* Integrate modules incrementally.
* Benchmark latency and memory.
* Fix issues before proceeding.
* Maintain clean modular code.
* Never fabricate evaluation results.
* Never claim features work without testing.

When making technical decisions, prioritise measurable performance over unnecessary complexity.

Do not introduce a database, authentication system or external services unless there is a clear requirement.

⸻

16. FINAL SUCCESS CRITERIA

Lumenoir should ultimately deliver:

1. A working classical CV enhancement pipeline.
2. A working lightweight deep learning pipeline.
3. An adaptive hybrid restoration system.
4. Region-aware enhancement.
5. Noise-aware denoising.
6. Overexposure protection.
7. Edge and texture preservation.
8. Quality-based output selection.
9. Explainable intermediate visualisations.
10. Classical vs DL vs hybrid comparisons.
11. Proper ablation studies.
12. Generalisation testing.
13. A polished Modify.app-inspired website.
14. Fully responsive mobile and desktop UI.
15. Public deployment.
16. Reproducible evaluation results.
17. Complete project documentation.

⸻

17. IMPLEMENTATION STATUS

Last updated: 6 November 2026. This section records what the repository actually contains. It does not change any requirement above, and it does not record any algorithm, weight or threshold as decided.

A. IMPLEMENTED AND TESTED

* Upload validation: JPG/JPEG/PNG only; extension and decoded format must agree; 10 MB byte limit; 64x64 minimum; 8000x8000 and 40 MP maximum; corrupted, truncated and spoofed files rejected with stable error codes.
* Safe decoding: one decode per request, detached RGB image, alpha removed, file handle closed.
* Basic image statistics: per-channel mean/min/max/stddev and a 16-bin luminance histogram, measured with Pillow. These are descriptive measurements, not illumination analysis or quality assessment.
* Safe resizing: aspect-preserving, longest-edge cap, never upscales, configurable pixel ceiling, original size restorable after processing.
* Pipeline orchestration and API integration: the seven stages are implemented and run in the order LumenScan → LumenRetinex ‖ Zero-DCE++ → LumenFuse → EdgeGuard → HueLock → adaptive alpha blend, with the do-no-harm bypass decided by LumenScan. POST /api/enhance validates the upload, runs the pipeline in a worker thread, returns the enhanced image as a base64 PNG at the exact input dimensions, and reports measured per-stage timings and available artifacts. Classical mode runs without a model; deep-learning and hybrid modes require a ready Zero-DCE++ checkpoint and otherwise return 503 with a typed error. Inference errors propagate as 503; stage failures return 500; invalid uploads keep their existing 4xx codes. Concurrent classical requests can run in parallel; ModelManager serialises model inference. The temporary 501 path has been removed.
* Inference plumbing: model interface (preprocess/infer/postprocess template with output validation), model registry (registers zero_dce_plus_plus), and a manager that loads weights once at startup, reports missing weights or load failures without crashing the app, serialises inference, measures duration and runs on CPU by configuration (LUMENOIR_INFERENCE_DEVICE, default cpu).
* API: health with model readiness, upload limits, upload validation with statistics, pipeline/model/mode/artifact status, enhancement request, and an evaluation registry that now reports the implemented metrics with their reference and dependency requirements.
* Frontend: landing page, upload workspace, method selector, before/after comparison slider (pointer and keyboard, aspect-ratio aware), intermediate output gallery that renders an output only when the backend produced it, image statistics and histogram, original download, enhanced download disabled until a real result exists, responsive single-column mobile layout.
* Dataset preparation pipeline (training/data_pipeline/): tolerant dataset discovery and low/high pairing by filename stem; per-image integrity checks with Pillow and OpenCV; 64-bit pHash duplicate detection with an exact four-block Hamming index; cross-dataset leakage removal (Hamming <= 8, then SSIM > 0.9 on 64x64 grayscale); deterministic split generation with the exact counts and seed 42 from the final decisions; explicit count-mismatch failures; 384x384 crop generation with EXIF-aware decoding, long-edge-1200 SICE resizing, flips/rot90 augmentation and the specified synth_dark formula; JSONL manifests, split files and cleaning reports. Raw images are never modified or deleted.
* Evaluation framework (backend/evaluation/): PSNR and SSIM (raw and GT-mean-aligned), LPIPS, NIQE via pyiqa, CIEDE2000 colour distortion, unpaired hue shift, relative noise and paired noise residual (Immerkaer), edge preservation, texture retention, new-clipping and highlight-push, black-level crushing, LOE on 50x50 max-RGB lightness maps, halo rate; reference-based and no-reference metrics live in separate modules and registries; optional dependencies are imported lazily; a metric that cannot run returns no value plus a note; aggregation (mean/std/median/P10/P90, worst five, Wilcoxon), latency benchmarking with fixed threads and peak RSS, and JSON/CSV/Markdown reports.
* Deep learning inference: Zero-DCE++ (NoirNet) is registered as zero_dce_plus_plus and loads the published 52,395-byte checkpoint (SHA-256 ca8855b9..., 10,561 parameters, 28 tensors, strict key match). Loading uses torch.load(weights_only=True), rejects mismatched or original Zero-DCE checkpoints, and warms the model once. Input is padded to a multiple of the scale factor and cropped back to the exact input size; the curve map is exposed as both the raw tanh output and the documented alpha (-raw). Device selection supports cpu, mps, cuda, cuda:N and auto (MPS -> CUDA -> CPU); explicit devices never fall back. Torch and NumPy are imported lazily, so the API still starts without them, and typed errors cover torch_missing, device_unavailable and checkpoint_invalid. It is wired into the pipeline's deep_learning stage; without torch or weights, deep-learning and hybrid modes return 503 while classical mode remains available.
* Classical CV primitives (backend/classical_cv/): float32 RGB [0,1] validation and Rec.601 luma; reflected-border box filter; guided filter; Immerkaer noise sigma and a clipped local noise map (the evaluation framework now re-exports this single implementation); LIME-style illumination estimation (max-RGB prior, guided filter, half resolution, clipped to [1e-3, 1]); LAB colour-cast detection on a brightened copy with the 25<=L*<=75 and C*<30 mask. No enhancement is applied by these modules.
* LumenScan (backend/classical_cv/scan.py): measures mean, p5/p50/p95, dynamic range, illumination non-uniformity NU = std(log2(T+1/64)), Immerkaer noise, LAB cast, highlight clipping at 0.98, crushed blacks at 0.02 and the darkness class; selects gamma, gain cap, guided epsilon, CLAHE clip/tiles/blend, DL strength, fusion exponents and tau, raw HueLock beta, output blend alpha and the do-no-harm bypass gate, using the exact rules and starting coefficients in final_decision.md section 5. Features use an analysis copy of at most 256 px except noise, which uses full resolution. The module is wired into the pipeline, and no parameter has been tuned or validated against enhanced output.
* LumenRetinex (backend/classical_cv/retinex.py): computes its own illumination map at input resolution using parameters.guided_eps, applies G = clip(T^(gamma-1), 1, G_cap), the highlight attenuation h = 1 - smoothstep(0.65, 0.95, max_c I), G' = 1 + (G-1)h, G'' = min(G', 1/(max_c I + 1e-3)) with one gain for RGB, then CLAHE on LAB L* using the LumenScan clip/tiles and the 0.4/0.6 blend, then the soft-knee shoulder at 0.85 and a final clip. Dimensions are preserved exactly; the result carries the gain and illumination maps. The section 5.8 bypass gate is honoured by default (configurable) and returns the input unchanged. Wired into the pipeline; no metric comparison has been run.
* EdgeGuard (backend/classical_cv/edge_guard.py): post-fusion denoising. Luma: effective gain G = clip(blur(Y_F,5)/(blur(Y_0,5)+0.02),1,30), G_ref = P75(G), guided filter with eps = (1.2*sigma*G_ref)^2 and radius 4 up to 1 MP / 6 above, blended by w = clip((G-1.5)/(G_ref-1.5),0,1)^0.8. Chroma: YCrCb Cr/Cb bilateral at half resolution with d=7, sigmaSpace=5 and sigmaColor = clip(3*255*sigma*sqrt(G_med),8,40). Uses the LumenScan noise sigma; returns the effective gain and weight maps. Wired into the pipeline; no tuning or benchmark has been run.
* HueLock (backend/classical_cv/hue_lock.py): uses LumenScan's detected LAB cast and raw beta, halves beta when max(std(a*), std(b*)) over the mid-tone low-chroma mask exceeds 15, caps the applied shift at 8 LAB units, and shifts only a*/b* globally. L* is untouched, so lightness, highlights and shadows are preserved and chroma is not boosted; a zero beta returns the input unchanged. Wired into the pipeline; no portrait or metric validation has been run.
* LumenFuse (backend/fusion/lumen_fuse.py): noise-aware Mertens-style fusion of the original, LumenRetinex and Zero-DCE++ candidates. Per-pixel weights combine contrast |Laplacian(Gaussian(Y, sigma=1))|, RGB standard deviation, exposure exp(-(Y-0.5)^2/0.08) and a gain-aware noise penalty 1/(1+(G*sigma_loc/tau)^2), with the exponents and tau from LumenScan; the original candidate is multiplied by 3 above 0.85. Weights are normalized with +1e-6, refined by a guided filter (guide=Y0, r=8, eps=1e-2), renormalized with a uniform fallback, and blended through Gaussian weight pyramids and Laplacian candidate pyramids with L = min(5, floor(log2(min(H,W)))-3) top level. OpenCV createMergeMertens is not used. Output is float32 RGB [0,1] at the input dimensions. Wired into the pipeline; no metric comparison or benchmark has been run.
* Kaggle evaluation runner (training/kaggle_eval/): explicit JSON configuration plus conservative discovery across /kaggle/input, /kaggle/input/datasets, local datasets/raw and the extracted additional root. It supports LOL-v1, LOL-v2 Real/Synthetic, SICE Part 1, ExDark, DICM, LIME, MEF, NPE and VV. LOL-v2 Real pairs low*/normal* stems; SICE supports train/test/samples/*/{Input,GT}.jpg and preserves train/test separation. LOL-v1, LOL-v2 Real, LOL-v2 Synthetic and SICE expose both training and test records for count verification, but the runner validates manifests and evaluates only test/eval/unpaired records. LIME (10 BMP), MEF (17 PNG), NPE (NPE 8 + NPE-ex1 29 + NPE-ex2 18 + NPE-ex3 30 = 85 images, one manifest and subgroup metadata) and VV (24) are unpaired. JPG, JPEG, PNG and BMP are supported; other image-like formats, empty directories and missing NPE subgroups produce warnings. Manifests include image/evaluation/training counts, split counts, format counts, subgroup counts and source roots. Paired metrics are PSNR, SSIM, LPIPS and CIEDE2000; unpaired metrics never include PSNR/SSIM/LPIPS. The runner writes image-level JSONL/CSV records, per-dataset/mode reports, aggregates, per-stage and total latency, warm-up benchmarks, failure logs, device/model status, a config snapshot and before/after PNGs. No full evaluation has been run.
* Tests: 317 lightweight backend, dataset-pipeline, metric, classical-CV/fusion, pipeline/API, Kaggle-evaluation and stage-attribution tests pass in about 11 seconds (`.venv/bin/python -m pytest -q`). Eight integration tests are excluded by default by pytest.ini: five Zero-DCE++ tests (real checkpoint, parameter/keys, CPU/MPS inference, CPU latency), one full hybrid API test with the real checkpoint, and two LPIPS/NIQE tests. The six Zero-DCE++/pipeline integration tests were run explicitly and passed on CPU and Apple Silicon MPS (Zero-DCE++ CPU median 31.8-35.1 ms at 1200x900 across runs); the two LPIPS/NIQE tests were not run because LPIPS downloads a 233 MB AlexNet checkpoint. Frontend production build and lint remain part of the checks.
* Measured full-pipeline latency on this machine (Apple Silicon, Python 3.14, torch 2.15.0, CPU, hybrid, model loaded once, synthetic images): 512x512 total 206.9 ms (analysis 16.1, Retinex 24.4, Zero-DCE++ 9.2, fusion 123.3, EdgeGuard 19.7, HueLock 6.9, final 2.2; median of 3 after warm-up); 1280x720 total 774.8 ms (single run); 1920x1080 total 1773.6 ms (single run). Model load including the first torch import was about 1.2 s. These are observations, not a general benchmark.

B. ARCHITECTURE PREPARED FOR FUTURE INTEGRATION

* backend/inference/: model-agnostic interface, registry and load-once manager. The registry now contains zero_dce_plus_plus; adding another architecture means registering one class and supplying its checkpoint.
* backend/inference/device.py: one place for device resolution and availability errors; the manager resolves once at load and reports the concrete device.
* backend/deep_learning/zero_dce_plus_plus.py: the inference wrapper and checkpoint validation; backend/deep_learning/zero_dce_net.py holds the torch architecture and is imported only at load time.
* backend/classical_cv/: standalone primitives (arrays, filters, noise, illumination, colour), LumenScan (scan.py), LumenRetinex (retinex.py), EdgeGuard (edge_guard.py) and HueLock (hue_lock.py). They accept float32 RGB [0,1], return their maps/parameters, and do not depend on the pipeline or the API.
* backend/classical_cv/scan.py: the only place that selects LumenScan parameters; starting coefficients live in LumenScanConfig so tuning later does not change the rules.
* backend/classical_cv/retinex.py: the classical enhancement branch; it consumes LumenScan parameters and does not select them.
* backend/classical_cv/edge_guard.py: the post-fusion denoiser; it consumes the LumenScan noise estimate and the original image.
* backend/classical_cv/hue_lock.py: applies LumenScan's cast and raw beta with the spread and 8-unit guards.
* backend/fusion/lumen_fuse.py: the noise-aware weight computation, guided refinement and Laplacian-pyramid blend; it consumes LumenScan's noise map and fusion exponents/parameters.
* backend/enhancement/stages.py: seven implemented stages that delegate to the modules above; PendingStage remains only for custom partial pipelines.
* backend/enhancement/pipeline.py: ordered execution, per-mode stage sets, per-stage timing, artifact bookkeeping and model-aware mode availability. The API runs it in a worker thread.
* backend/preprocessing/resize.py: the single place that changes image size for inference.
* backend/evaluation/metrics.py: dependency-light metric declarations and the legacy compute adapter; the harness is reached through lazy exports. The API imports NumPy/OpenCV for the pipeline but never imports torch at startup.
* backend/evaluation/registry.py: adding a metric requires one registration and no harness change.
* training/data_pipeline/config.py: the single place holding the dataset layouts, split counts, seeds, crop size, Hamming threshold and SSIM threshold.
* training/data_pipeline/splits.py: split rules and leakage filtering; real split files are generated once the datasets exist.
* training/kaggle_eval/config.py: the supported dataset catalogue, explicit root/pair override schema, additional search roots, NPE subgroup definitions, evaluation metric sets, mode list, seed and path defaults.
* training/kaggle_eval/discovery.py: deterministic manifest generation, multi-root conservative discovery, exact NPE/NPE-ex1/NPE-ex2/NPE-ex3 grouping, held-out split preference, reference pairing by stem, training-split guards, format/count reporting and unsupported-format warnings.
* training/kaggle_eval/runner.py: four-mode evaluation through the existing EnhancementPipeline, image-level metric/subgroup records, dataset aggregates, per-stage latency, failure logs, device/model/config snapshots and before/after visuals.
* The API contract already carries processing status, measured processing time, selected method, image dimensions, basic statistics and intermediate artifacts, so the frontend does not need reshaping when real outputs appear.
* Every request is stateless: no database, no authentication, no image storage, no background queue.

C. PENDING TECHNICAL DECISIONS (NOT ASSUMED)

* the pipeline is complete; only the optional closed-loop passes remain
* training and fine-tuning strategy, loss functions and hyper-parameters (the architecture itself is selected: Zero-DCE++)
* tuning of the rule-based coefficients (§5.6) and the closed-loop proxies (§5.7)
* quality-based output selection across multiple complete runs
* ablation design, benchmark tables and the deployment platform
* the "textured patch" percentile used by texture retention and the exact GT-mean alignment formula (the decision document names both without fixing them; they are documented parameters, not hidden defaults)

D. CURRENT STATUS VALUES

* Enhancement pipeline: implemented and wired into POST /api/enhance. Classical mode runs without a model; deep_learning and hybrid require a ready Zero-DCE++ checkpoint and otherwise return 503. The temporary 501 path has been removed. The pipeline returns the exact input dimensions and RGB output.
* Classical CV primitives, LumenScan, LumenRetinex, EdgeGuard, HueLock and LumenFuse: implemented, unit-tested and wired into the pipeline. The rules and weights are the untuned starting values from final_decision.md §4-§5, and no metric comparison, real-photograph or dataset validation has been run.
* Registered models: zero_dce_plus_plus (Zero-DCE++, pretrained inference).
* Model weights: backend/weights/zero_dce_plus_plus.pth is present locally (52,395 bytes, git-ignored); no trained or fine-tuned checkpoint exists.
* Dataset pipeline: implemented; local data is present under datasets/raw (LOL-v1, LOL-v2 Real/Synthetic, SICE_Paired and extracted NPE/LIME/MEF/VV), while ExDark and DICM are absent. Only the count specification (training/splits/split_spec.json) has been generated; no real split files or cleaning report exist yet.
* Evaluation framework: implemented, including LOE and black-level crushing; no benchmark has been run and no metric value is reported in this document.
* Kaggle evaluation runner: implemented with deterministic discovery/configuration, train/test split validation, four modes, metric/subgroup records, aggregates, total/stage latency, warm-up benchmarks, failures, device reporting and visuals; local discovery was verified against datasets/raw, but no full evaluation has been run.
* Runtime dependencies: FastAPI, Uvicorn, Pillow, python-multipart, python-dotenv, NumPy and OpenCV (the pipeline needs NumPy/OpenCV). The API does not import torch, scikit-image, lpips or pyiqa at startup.
* Inference dependencies: torch via requirements-inference.txt (optional; the API starts without it and reports torch_missing if a deep-learning/hybrid request is made).
* Development/evaluation dependencies: numpy, opencv-python-headless, scikit-image, imagehash, scipy, psutil (requirements-eval.txt); torch, torchvision, lpips, pyiqa (requirements-metrics.txt, optional); pytest and pytest-timeout.

⸻

FINAL INSTRUCTION

Before implementation, provide:

A. Proposed technical architecture.

B. Final lightweight model recommendation.

C. Dataset and training strategy.

D. Image-processing scenarios and decision logic.

E. Latency and memory optimisation plan.

F. Mobile-first UI/UX plan.

G. Deployment architecture.

H. Development milestones.

I. Testing and benchmarking strategy.

J. Risks, limitations and expected bottlenecks.

Do not begin full implementation until this planning stage is complete.

The ultimate objective is to build a technically strong, lightweight, intelligent and professionally deployed low-light image restoration application that performs well against every supplied hackathon evaluation criterion.