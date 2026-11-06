"""Command line interface for dataset preparation.

    python -m training.data_pipeline check
    python -m training.data_pipeline split --spec-only
    python -m training.data_pipeline organise
    python -m training.data_pipeline all

Run ``--help`` for every command. Commands that need images fail with a clear
message when the datasets have not been downloaded yet.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from .config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MANIFEST_DIR,
    DEFAULT_PROCESSED_ROOT,
    DEFAULT_REPORT_DIR,
    DEFAULT_SPLIT_DIR,
    SEED,
)
from .organisation import iter_images, scan_all, write_manifest
from .pipeline import (
    dataset_check_report,
    load_valid_scans,
    prepare,
    preprocess_pool,
    split_spec,
)
from .sources import describe_sources, free_disk_gb
from .splits import SplitSpecMismatch
from .statistics import collect_statistics
from .synthetic import generate_synthetic_darks


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N images.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Record dataset count deviations instead of failing.",
    )
    parser.add_argument(
        "--skip-leakage",
        action="store_true",
        help="Skip hashing, dedupe and leakage checks (fast, unsafe).",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    common = _common_parser()
    parser = argparse.ArgumentParser(
        prog="python -m training.data_pipeline",
        description="LUMENOIR dataset preparation pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", parents=[common], help="Report dataset presence.")
    check.add_argument("--sources", action="store_true", help="Print download sources.")
    check.add_argument(
        "--strict", action="store_true", help="Exit 1 when any dataset is missing."
    )

    subparsers.add_parser("sources", help="Print dataset sources and licences.")

    subparsers.add_parser("organise", parents=[common], help="Scan datasets and write manifests.")

    subparsers.add_parser("validate", parents=[common], help="Probe every image for corruption.")

    subparsers.add_parser("dedupe", parents=[common], help="Detect duplicates and leakage.")

    split = subparsers.add_parser("split", parents=[common], help="Generate split files.")
    split.add_argument(
        "--spec-only",
        action="store_true",
        help="Write split_spec.json (counts and rules) without reading images.",
    )
    split.add_argument("--dry-run", action="store_true")

    preprocess = subparsers.add_parser(
        "preprocess", parents=[common], help="Generate crops for one split pool."
    )
    preprocess.add_argument("--pool", default="fine_tune", help="Pool name (default: fine_tune).")
    preprocess.add_argument("--split-file", type=Path, default=None)
    preprocess.add_argument("--crop-size", type=int, default=384)
    preprocess.add_argument("--augment", action="store_true")
    preprocess.add_argument("--darken", action="store_true")

    synthetic = subparsers.add_parser(
        "synthetic", parents=[common], help="Generate extreme-dark images from sources."
    )
    synthetic.add_argument("--sources", type=Path, required=True, help="Source image folder.")
    synthetic.add_argument("--count", type=int, required=True)
    synthetic.add_argument("--output", type=Path, default=None)

    subparsers.add_parser("report", parents=[common], help="Write dataset statistics reports.")

    run_all = subparsers.add_parser("all", parents=[common], help="Run the full pipeline.")
    run_all.add_argument("--dry-run", action="store_true")

    return parser


def _print_validation_summary(validation) -> None:
    print(f"Valid records: {len(validation.valid)}")
    if validation.problems:
        codes = Counter(problem.code for problem in validation.problems)
        print("Rejected records: " + ", ".join(f"{code}={count}" for code, count in codes.items()))
        for problem in validation.problems[:10]:
            print(f"  - {problem.path}: {problem.code} ({problem.detail})")
    else:
        print("Rejected records: none")


def _cmd_check(args: argparse.Namespace) -> int:
    print(dataset_check_report(args.data_root, include_sources=args.sources))
    if args.strict:
        scans = scan_all(args.data_root)
        if any(not scan.present for scan in scans):
            return 1
    return 0


def _cmd_organise(args: argparse.Namespace) -> int:
    scans = scan_all(args.data_root)
    path = write_manifest(scans, args.manifest_dir / "manifest.jsonl")
    for scan in scans:
        status = "present" if scan.present else "MISSING"
        print(f"{scan.spec.label:<18} {status:<8} records={scan.found_total}")
    print(f"Manifest: {path}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    _, validation = load_valid_scans(args.data_root, limit=args.limit)
    _print_validation_summary(validation)
    return 0


def _cmd_dedupe(args: argparse.Namespace) -> int:
    try:
        result = prepare(
            data_root=args.data_root,
            split_dir=args.split_dir,
            report_dir=args.report_dir,
            manifest_dir=args.manifest_dir,
            seed=args.seed,
            allow_count_mismatch=args.allow_count_mismatch,
            detect_leakage=not args.skip_leakage,
            limit=args.limit,
            write=False,
        )
    except SplitSpecMismatch as exc:
        print(f"Split specification mismatch: {exc}", file=sys.stderr)
        return 1

    print(f"Valid records: {len(result.validation.valid)}")
    print(f"Duplicates removed from fine-tune: {len(result.bundle.duplicates)}")
    print(f"Leakage removed: {len(result.bundle.leakage)}")
    for match in (*result.bundle.duplicates, *result.bundle.leakage)[:10]:
        print(
            f"  - dropped {match.dropped_id} (kept {match.kept_id}, "
            f"hamming={match.hamming}, ssim={match.ssim:.3f})"
        )
    return 0


def _cmd_split(args: argparse.Namespace) -> int:
    if args.spec_only:
        path = split_spec(args.split_dir, seed=args.seed)
        print(f"Split spec: {path}")
        return 0

    try:
        result = prepare(
            data_root=args.data_root,
            split_dir=args.split_dir,
            report_dir=args.report_dir,
            manifest_dir=args.manifest_dir,
            seed=args.seed,
            allow_count_mismatch=args.allow_count_mismatch,
            detect_leakage=not args.skip_leakage,
            limit=args.limit,
            write=False,
        )
    except SplitSpecMismatch as exc:
        print(f"Split specification mismatch: {exc}", file=sys.stderr)
        return 1

    for pool, total in result.bundle.totals().items():
        print(f"{pool:<12} {total}")
    if args.dry_run:
        print("Dry run: no split files written.")
        return 0

    written = result.bundle.write(args.split_dir, overwrite=args.force)
    for path in written:
        print(f"wrote {path}")
    return 0


def _cmd_preprocess(args: argparse.Namespace) -> int:
    split_file = args.split_file or (args.split_dir / f"{args.pool}.json")
    if not split_file.is_file():
        print(
            f"Split file '{split_file}' not found. Run 'split' first or pass --split-file.",
            file=sys.stderr,
        )
        return 1

    result = preprocess_pool(
        split_file,
        args.processed_root,
        crop_size=args.crop_size,
        augment=args.augment,
        darken=args.darken,
        seed=args.seed,
        limit=args.limit,
        overwrite=args.force,
    )
    print(
        f"Written: {len(result.written)}, skipped: {result.skipped}, "
        f"failed: {len(result.failures)}"
    )
    for failure in result.failures[:10]:
        print(f"  - {failure}")
    return 0


def _cmd_synthetic(args: argparse.Namespace) -> int:
    sources = iter_images(args.sources)
    if not sources:
        print(f"No source images found in '{args.sources}'.", file=sys.stderr)
        return 1

    output = args.output or (args.processed_root / "synthetic_dark")
    written = generate_synthetic_darks(
        sources, output, count=args.count, seed=args.seed, overwrite=args.force
    )
    print(f"Synthetic images: {len(written)} in {output}")
    print(f"Free disk space: {free_disk_gb(output):.1f} GB")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    scans, validation = load_valid_scans(args.data_root, limit=args.limit)
    statistics = collect_statistics(scans, validation=validation)
    for path in statistics.write(args.report_dir):
        print(f"wrote {path}")
    return 0


def _cmd_all(args: argparse.Namespace) -> int:
    try:
        result = prepare(
            data_root=args.data_root,
            split_dir=args.split_dir,
            report_dir=args.report_dir,
            manifest_dir=args.manifest_dir,
            seed=args.seed,
            allow_count_mismatch=args.allow_count_mismatch,
            detect_leakage=not args.skip_leakage,
            limit=args.limit,
            write=not args.dry_run,
            overwrite=args.force,
        )
    except SplitSpecMismatch as exc:
        print(f"Split specification mismatch: {exc}", file=sys.stderr)
        return 1

    _print_validation_summary(result.validation)
    for pool, total in result.bundle.totals().items():
        print(f"{pool:<12} {total}")
    print(f"Duplicates removed: {len(result.bundle.duplicates)}")
    print(f"Leakage removed: {len(result.bundle.leakage)}")
    if args.dry_run:
        print("Dry run: nothing written.")
    else:
        print(f"Files written: {len(result.written)}")
    return 0


def _cmd_sources(_args: argparse.Namespace) -> int:
    print(describe_sources())
    return 0


COMMANDS = {
    "check": _cmd_check,
    "sources": _cmd_sources,
    "organise": _cmd_organise,
    "validate": _cmd_validate,
    "dedupe": _cmd_dedupe,
    "split": _cmd_split,
    "preprocess": _cmd_preprocess,
    "synthetic": _cmd_synthetic,
    "report": _cmd_report,
    "all": _cmd_all,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":  # pragma: no cover - exercised through the module entry point
    raise SystemExit(main())
