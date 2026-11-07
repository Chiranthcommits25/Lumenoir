"""Ablation study execution harness.

Runs the controlled leave-one-module-out ablation experiment across the
six canonical configurations:
  1. baseline: all 5 modules enabled
  2. without_lumen_scan: fixed scan parameters (bypasses adaptive scan)
  3. without_lumen_retinex: original input as classical candidate
  4. without_edge_guard: bypasses denoising
  5. without_hue_lock: bypasses colour correction
  6. without_lumen_fuse: simple 50/50 blend of classical & DL candidates

Enforces strict isolation by completely disabling shared-stage caching during
ablation runs. Computes all standard evaluation metrics (PSNR, SSIM, LPIPS,
CIEDE2000, noise suppression, edge preservation, etc.) and writes comparable
CSV, JSON, and Markdown summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from backend import deep_learning  # noqa: F401  (registers Zero-DCE++)
from backend.config import DEFAULT_WEIGHTS_DIR, InferenceSettings
from backend.enhancement.ablation import (
    ABLATION_EXPERIMENTS,
    ABLATION_TITLES,
    AblationConfig,
    AblationName,
)
from backend.enhancement.pipeline import EnhancementPipeline
from backend.enhancement.types import ProcessingStatus
from backend.evaluation.aggregate import summarise
from backend.evaluation.harness import evaluate_arrays
from backend.evaluation.images import to_float32_rgb
from backend.inference.manager import ModelManager, ModelStatus

DEFAULT_WEIGHTS = DEFAULT_WEIGHTS_DIR / "zero_dce_plus_plus.pth"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "raw"
DEFAULT_METRICS = (
    "psnr",
    "ssim",
    "lpips",
    "colour_distortion",
    "noise_suppression",
    "edge_preservation",
    "overexposure",
    "halo",
)


@dataclass(frozen=True)
class PairRecord:
    dataset: str
    image_id: str
    low: Path
    high: Path


def _resolve(path: Path, base: Path, data_root: Path) -> Path:
    if path.is_absolute():
        return path
    for candidate in (base / path, data_root / path, PROJECT_ROOT / path, Path.cwd() / path):
        if candidate.is_file():
            return candidate.resolve()
    return (base / path).resolve()


def load_pairs(manifest_path: Path, data_root: Path = DEFAULT_DATA_ROOT) -> list[PairRecord]:
    if not manifest_path.is_file():
        raise ValueError(f"Manifest file not found: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_records: list[dict[str, Any]] = []
    if isinstance(payload.get("datasets"), list):
        for ds in payload["datasets"]:
            ds_name = ds.get("dataset") or ds.get("key") or "val"
            for item in ds.get("records", []):
                rec = dict(item)
                rec.setdefault("dataset", ds_name)
                raw_records.append(rec)
    elif isinstance(payload.get("records"), list):
        raw_records = [dict(item) for item in payload["records"]]
    else:
        raise ValueError("Manifest format unknown (expected 'datasets' or 'records').")

    base = manifest_path.parent
    pairs: list[PairRecord] = []
    for item in raw_records:
        low_p = _resolve(Path(str(item["low"])), base, data_root)
        high_p = _resolve(Path(str(item["high"])), base, data_root)
        if low_p.is_file() and high_p.is_file():
            pairs.append(
                PairRecord(
                    dataset=str(item.get("dataset", "val")),
                    image_id=str(item.get("image_id") or item.get("id") or low_p.name),
                    low=low_p,
                    high=high_p,
                )
            )
    return pairs


def run_ablation_study(
    manifest_path: Path,
    output_dir: Path,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    weights_path: Path = DEFAULT_WEIGHTS,
    device: str = "cpu",
    max_images: int | None = None,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    experiments: tuple[AblationName, ...] = tuple(ABLATION_EXPERIMENTS.keys()),
) -> dict[str, Any]:
    """Execute all requested ablation experiments and save comparative reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(manifest_path, data_root)
    if max_images is not None and max_images > 0:
        pairs = pairs[:max_images]

    if not pairs:
        raise ValueError(f"No valid image pairs found from manifest {manifest_path}")

    print(f"Loaded {len(pairs)} image pairs for ablation study.")
    print(f"Experiments to evaluate: {', '.join(experiments)}")

    # Setup pipeline
    manager = None
    if weights_path.is_file():
        settings = InferenceSettings(
            model_id="zero_dce_plus_plus",
            weights_path=weights_path,
            device=device,
        )
        manager = ModelManager(settings=settings)
        manager.load()
    else:
        print(f"Warning: Weights not found at {weights_path}; DL branch will run in stub mode if available.")

    pipeline = EnhancementPipeline(model_manager=manager)

    # Collect per-experiment results
    raw_results: dict[str, list[dict[str, Any]]] = {exp: [] for exp in experiments}
    metric_values: dict[str, dict[str, list[float]]] = {
        exp: {m: [] for m in metrics} for exp in experiments
    }

    for idx, pair in enumerate(pairs):
        print(f"[{idx+1}/{len(pairs)}] Evaluating {pair.dataset}:{pair.image_id}...")
        with Image.open(pair.low) as raw_low, Image.open(pair.high) as raw_high:
            low_img = ImageOps.exif_transpose(raw_low).convert("RGB")
            high_img = ImageOps.exif_transpose(raw_high).convert("RGB")
            ref_array = to_float32_rgb(high_img, name="reference")
            orig_array = to_float32_rgb(low_img, name="original")

            for exp_name in experiments:
                cfg = ABLATION_EXPERIMENTS[exp_name]
                # Enforce no shared stage cache (cache=None)
                res = pipeline.run(
                    low_img,
                    mode="hybrid",
                    cache=None,
                    ablation=cfg,
                    capture_diagnostics=False,
                )

                if res.status is not ProcessingStatus.COMPLETED or res.final_image is None:
                    print(f"  FAILED: {exp_name} on {pair.image_id}: {res.message}")
                    continue

                out_array = to_float32_rgb(res.final_image, name="output")
                eval_metrics = evaluate_arrays(
                    out_array,
                    reference=ref_array,
                    original=orig_array,
                    keys=metrics,
                    assume_valid=True,
                )

                row_metrics = {}
                for m in metrics:
                    val = eval_metrics.get(m)
                    val_num = val.value if (val is not None and val.value is not None and math.isfinite(val.value)) else None
                    row_metrics[m] = val_num
                    if val_num is not None:
                        metric_values[exp_name][m].append(val_num)

                record = {
                    "experiment": exp_name,
                    "title": ABLATION_TITLES[exp_name],
                    "dataset": pair.dataset,
                    "image_id": pair.image_id,
                    "metrics": row_metrics,
                    "total_ms": res.total_ms,
                }
                raw_results[exp_name].append(record)

    # Compute comparative aggregate summaries
    summary_table: list[dict[str, Any]] = []
    for exp_name in experiments:
        row: dict[str, Any] = {
            "experiment": exp_name,
            "title": ABLATION_TITLES[exp_name],
            "count": len(raw_results[exp_name]),
        }
        for m in metrics:
            vals = metric_values[exp_name][m]
            if vals:
                row[f"{m}_mean"] = round(float(np.mean(vals)), 4)
                row[f"{m}_std"] = round(float(np.std(vals)), 4)
            else:
                row[f"{m}_mean"] = None
                row[f"{m}_std"] = None
        summary_table.append(row)

    # Write summary CSV
    csv_path = output_dir / "ablation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        if summary_table:
            writer = csv.DictWriter(f, fieldnames=list(summary_table[0].keys()))
            writer.writeheader()
            writer.writerows(summary_table)

    # Write detailed JSON
    json_path = output_dir / "ablation_summary.json"
    full_export = {
        "generated_at": datetime.now(UTC).isoformat(),
        "pairs_evaluated": len(pairs),
        "experiments": summary_table,
        "details": raw_results,
    }
    json_path.write_text(json.dumps(full_export, indent=2), encoding="utf-8")

    # Generate Markdown summary table
    md_lines = [
        "# LUMENOIR Ablation Study Results",
        "",
        f"**Generated:** {datetime.now(UTC).isoformat()}  ",
        f"**Image count:** {len(pairs)} pairs  ",
        "",
        "| Experiment | Configuration | PSNR (dB) | SSIM | LPIPS | ΔE00 | Noise Suppr. | Edge Preserv. |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in summary_table:
        psnr_str = f"{row.get('psnr_mean', '-')} ± {row.get('psnr_std', '-')}" if row.get('psnr_mean') is not None else "-"
        ssim_str = f"{row.get('ssim_mean', '-')} ± {row.get('ssim_std', '-')}" if row.get('ssim_mean') is not None else "-"
        lpips_str = f"{row.get('lpips_mean', '-')} ± {row.get('lpips_std', '-')}" if row.get('lpips_mean') is not None else "-"
        de_str = f"{row.get('colour_distortion_mean', '-')} ± {row.get('colour_distortion_std', '-')}" if row.get('colour_distortion_mean') is not None else "-"
        noise_str = f"{row.get('noise_suppression_mean', '-')} ± {row.get('noise_suppression_std', '-')}" if row.get('noise_suppression_mean') is not None else "-"
        edge_str = f"{row.get('edge_preservation_mean', '-')} ± {row.get('edge_preservation_std', '-')}" if row.get('edge_preservation_mean') is not None else "-"

        md_lines.append(
            f"| `{row['experiment']}` | {row['title']} | {psnr_str} | {ssim_str} | {lpips_str} | {de_str} | {noise_str} | {edge_str} |"
        )

    md_path = output_dir / "ablation_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"\nAblation study completed successfully.")
    print(f"Summary CSV:  {csv_path}")
    print(f"Summary JSON: {json_path}")
    print(f"Summary MD:   {md_path}\n")

    return full_export


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LUMENOIR Ablation Study.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to paired validation/test JSON manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "ablation",
        help="Output directory for reports.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory for dataset images.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to Zero-DCE++ model weights.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on (cpu, cuda, mps).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Cap on number of images to evaluate.",
    )

    args = parser.parse_args()
    run_ablation_study(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        data_root=args.data_root,
        weights_path=args.weights,
        device=args.device,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()

