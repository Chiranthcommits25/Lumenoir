"""Command-line entry point for the Kaggle evaluation runner.

    python -m training.kaggle_eval --config eval_config.json
    python -m training.kaggle_eval --discover-only --input-root /kaggle/input
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import MODES, EvalConfig
from .discovery import discover, write_manifests
from .runner import run_evaluation


def _csv_list(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.kaggle_eval",
        description=(
            "Evaluate LUMENOIR on datasets mounted under /kaggle/input and "
            "/kaggle/working/lumenoir_additional/extracted."
        ),
    )
    parser.add_argument("--config", type=Path, default=None, help="JSON configuration.")
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument(
        "--additional-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Extra search root, e.g. "
            "/kaggle/working/lumenoir_additional/extracted. Repeatable."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated keys.")
    parser.add_argument(
        "--modes",
        type=str,
        default=None,
        help=f"Comma-separated modes from: {', '.join(MODES)}.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, mps or auto.")
    parser.add_argument("--skip-metrics", type=str, default=None)
    parser.add_argument("--visual-limit", type=int, default=None)
    parser.add_argument(
        "--latency-warmups",
        type=int,
        default=None,
        help="Warm-up iterations for the latency benchmark (default: 1).",
    )
    parser.add_argument(
        "--latency-runs",
        type=int,
        default=None,
        help="Timed iterations for the latency benchmark; 0 disables it (default: 3).",
    )
    parser.add_argument("--no-visuals", action="store_true")
    parser.add_argument(
        "--strict-model",
        action="store_true",
        help="Fail instead of skipping DL/hybrid modes when the model is unavailable.",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Write manifests and exit without loading the pipeline.",
    )
    return parser


def _apply_cli(config: EvalConfig, args: argparse.Namespace) -> EvalConfig:
    changes = {}
    if args.input_root is not None:
        changes["input_root"] = args.input_root
    if args.additional_root is not None:
        changes["additional_roots"] = tuple(args.additional_root)
    if args.output_dir is not None:
        changes["output_dir"] = args.output_dir
    datasets = _csv_list(args.datasets)
    if datasets is not None:
        changes["selected_datasets"] = datasets
    modes = _csv_list(args.modes)
    if modes is not None:
        changes["modes"] = modes
    if args.seed is not None:
        changes["seed"] = args.seed
    if args.max_images is not None:
        changes["max_images"] = args.max_images
    if args.weights is not None:
        changes["weights"] = args.weights
    if args.device is not None:
        changes["device"] = args.device
    skip_metrics = _csv_list(args.skip_metrics)
    if skip_metrics is not None:
        changes["skip_metrics"] = skip_metrics
    if args.visual_limit is not None:
        changes["visual_limit"] = args.visual_limit
    if args.latency_warmups is not None:
        changes["latency_warmups"] = args.latency_warmups
    if args.latency_runs is not None:
        changes["latency_runs"] = args.latency_runs
    if args.no_visuals:
        changes["save_visuals"] = False
    if args.strict_model:
        changes["allow_missing_model"] = False
    return config.replace(**changes) if changes else config


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = (
            EvalConfig.from_file(args.config) if args.config else EvalConfig()
        )
        config = _apply_cli(config, args)
    except (OSError, ValueError) as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 2

    if args.discover_only:
        result = discover(config)
        paths = write_manifests(result, config.output_dir, config=config)
        for manifest in result.manifests:
            status = "present" if manifest.present else "skipped"
            counts = (
                f"images={manifest.image_count:<5} "
                f"eval={manifest.evaluation_count:<5} "
                f"train={manifest.training_count:<5} "
                f"formats={manifest.format_counts} "
                f"subgroups={manifest.subgroup_counts}"
            )
            print(
                f"{manifest.key:<12} {status:<8} "
                f"{counts} {manifest.skipped_reason or ''}"
            )
        print(f"Manifests written: {', '.join(str(path) for path in paths)}")
        return 0

    try:
        summary = run_evaluation(config)
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Output: {summary.output_dir}")
    print(f"Records: {summary.record_count}, failures: {summary.failure_count}")
    if summary.skipped_modes:
        print(f"Skipped modes: {', '.join(summary.skipped_modes)}")
    for warning in summary.warnings:
        print(f"Warning: {warning}")
    print("Reports: " + ", ".join(str(path) for path in summary.aggregate_paths[:5]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
