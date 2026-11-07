# LUMENOIR Formal Ablation Study Methodology

## 1. Objective & Motivation

The LUMENOIR enhancement pipeline integrates five specialized modules:
1. **LumenScan**: Rule-based parameter selection & illumination/noise measurement.
2. **LumenRetinex**: Classical Retinex gain, highlight attenuation, CIELAB CLAHE, and soft-knee shoulder.
3. **NoirNet (Zero-DCE++)**: Deep curve network estimating spatial illumination curve maps.
4. **LumenFuse**: Multi-candidate noise-aware Laplacian pyramid fusion.
5. **EdgeGuard**: Gain-aware luma guided filter and chroma bilateral filter.
6. **HueLock**: CIELAB colour-cast detection and guarded chroma shifting.

To isolate the causal contribution of each component without introducing confounding factors, we conduct a **controlled leave-one-module-out ablation study**. Every ablation run evaluates the exact same input images using frozen model weights, identical baseline hyperparameters, and strict component-level substitutions.

---

## 2. Experimental Configurations

| ID | Experiment | Bypassed Module | Replacement Mechanism | Downstream Impact |
|---|---|---|---|---|
| **E0** | `baseline` | None | Real output of all five modules | Complete production hybrid pipeline. |
| **E1** | `without_lumen_scan` | LumenScan | Fixed static parameter set ($\gamma=0.6$, $\sigma=0.01$, default fusion exponents, $\beta=0.0$, $\alpha=1.0$) | Evaluates loss of adaptive parameter tuning across diverse dark slices. |
| **E2** | `without_lumen_retinex` | LumenRetinex | Pass-through identity: $C = I_0$ | Fusion blends $[I_0, I_0, D]$; classical illumination boost is removed. |
| **E3** | `without_edge_guard` | EdgeGuard | Pass-through: $F' = F$ | Post-fusion denoising is skipped; raw fused noise propagates forward. |
| **E4** | `without_hue_lock` | HueLock | Pass-through: $F'' = F'$ | Colour-cast correction is skipped; evaluates chromatic drift. |
| **E5** | `without_lumen_fuse` | LumenFuse | Unweighted 50/50 blend: $F = \text{clip}(0.5 \cdot C + 0.5 \cdot D, 0, 1)$ | Evaluates multiscale pyramid fusion and noise penalties vs naive blending. |

---

## 3. Strict Experimental Controls

1. **Isolation & Cache Disabling**: Shared stage caching (`SharedStageCache`) is strictly bypassed (`cache=None`) during ablation to prevent cross-run pollution.
2. **Deterministic Backends**: Both NumPy/OpenCV (CPU) and PyTorch tensor (GPU) runtime paths implement identical pass-through semantics.
3. **Frozen Weights**: Zero-DCE++ model weights are never updated during the study.
4. **Full Artifact Consistency**: Every ablation guarantees a finite float32 image in $[0.0, 1.0]$ matching input dimensions, with no NaN or infinite values.

---

## 4. Evaluation Metrics

Comparative performance is quantified using the repository's evaluation harness:
- **PSNR (dB) & SSIM**: Full-reference reconstruction fidelity.
- **LPIPS**: Learned Perceptual Image Patch Similarity (AlexNet backbone).
- **CIEDE2000 ($\Delta E_{00}$)**: Colour distortion relative to reference ground truth (validates HueLock).
- **Relative Noise**: Ratio of high-frequency noise variance in flat regions (validates EdgeGuard).
- **Edge Correlation**: Sobel gradient correlation with ground truth (measures edge preservation).
- **Overexposure / New Clipping %**: Fraction of newly saturated pixels (validates LumenRetinex shoulder and LumenFuse highlight preservation).
- **Halo Score**: Gradient-reversal rate at sharp edges (validates LumenFuse pyramid fusion vs naive average).

---

## 5. Execution on Kaggle & Local Environments

### Local / Development Execution:
```bash
python -m training.experiments.ablation_study \
  --manifest training/splits/validation.json \
  --output-dir reports/ablation \
  --weights backend/weights/zero_dce_plus_plus.pth \
  --device cpu \
  --max-images 20
```

### Kaggle GPU Execution:
```bash
python -m training.experiments.ablation_study \
  --manifest /kaggle/input/lumenoir-splits/validation.json \
  --data-root /kaggle/input \
  --output-dir /kaggle/working/reports/ablation \
  --weights /kaggle/input/lumenoir-weights/zero_dce_plus_plus.pth \
  --device cuda
```

### Generated Outputs:
- `ablation_summary.csv`: Machine-readable comparison table of means and standard deviations across all metrics.
- `ablation_summary.json`: Complete serialized metrics and execution timings per image pair.
- `ablation_summary.md`: Clean Markdown comparison table for evaluation reports and documentation.

