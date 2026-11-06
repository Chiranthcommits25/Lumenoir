"""Orchestration for dataset preparation.

Each step is a small function so the CLI, the tests and a notebook can call
exactly the part they need. Raw images are only ever read.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from .config import (
    CROP_SIZE,
    DEFAULT_DATA_ROOT,
    DEFAULT_MANIFEST_DIR,
    DEFAULT_PROCESSED_ROOT,
    DEFAULT_REPORT_DIR,
    DEFAULT_SPLIT_DIR,
    PROJECT_ROOT,
    SEED,
    SICE_LONG_EDGE,
    SplitRules,
)
from .organisation import DatasetScan, ImageRecord, scan_all, write_manifest
from .preprocessing import (
    PreprocessingError,
    apply_darkening,
    augment_views,
    load_rgb_float32,
    random_crop,
    random_crop_pair,
    resize_long_edge,
    save_png,
)
from .sources import describe_sources, format_presence, free_disk_gb, missing_datasets
from .splits import SplitBundle, build_splits, write_split_spec
from .statistics import DatasetStatistics, collect_statistics
from .validation import ValidationOutcome, validate_records


@dataclass(frozen=True)
class PreparationResult:
    scans: tuple[DatasetScan, ...]
    validation: ValidationOutcome
    bundle: SplitBundle
    statistics: DatasetStatistics
    written: tuple[Path, ...]


@dataclass(frozen=True)
class PreprocessResult:
    written: tuple[Path, ...]
    skipped: int
    failures: tuple[str, ...]


def load_valid_scans(
    data_root: Path = DEFAULT_DATA_ROOT,
    *,
    limit: int | None = None,
) -> tuple[tuple[DatasetScan, ...], ValidationOutcome]:
    """Scan every dataset and keep only records whose images decode."""
    scans = tuple(scan_all(data_root))
    all_records = [record for scan in scans for record in scan.records]
    validation = validate_records(all_records, limit=limit)

    valid_ids = validation.valid_ids
    filtered: list[DatasetScan] = []
    for scan in scans:
        records = tuple(
            record for record in scan.records if record.record_id in valid_ids
        )
        counts = Counter(record.group for record in records)
        filtered.append(
            replace(
                scan,
                records=records,
                group_counts={group.name: counts.get(group.name, 0) for group in scan.spec.groups},
            )
        )

    return tuple(filtered), validation


def prepare(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    split_dir: Path = DEFAULT_SPLIT_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
    manifest_dir: Path = DEFAULT_MANIFEST_DIR,
    seed: int = SEED,
    allow_count_mismatch: bool = False,
    detect_leakage: bool = True,
    limit: int | None = None,
    write: bool = True,
    overwrite: bool = False,
) -> PreparationResult:
    """Validate, deduplicate, split and report. Raises on count mismatches."""
    scans, validation = load_valid_scans(data_root, limit=limit)
    bundle = build_splits(
        scans,
        SplitRules(seed=seed),
        allow_count_mismatch=allow_count_mismatch,
        detect_leakage=detect_leakage,
    )
    statistics = collect_statistics(scans, validation=validation, bundle=bundle)

    written: list[Path] = []
    if write:
        written.append(write_manifest(scans, manifest_dir / "manifest.jsonl"))
        written.extend(bundle.write(split_dir, overwrite=overwrite))
        written.extend(statistics.write(report_dir))

    return PreparationResult(
        scans=scans,
        validation=validation,
        bundle=bundle,
        statistics=statistics,
        written=tuple(written),
    )


def split_spec(
    split_dir: Path = DEFAULT_SPLIT_DIR,
    *,
    seed: int = SEED,
    write: bool = True,
) -> Path | None:
    """Write the specification (counts and rules) without touching any data."""
    if not write:
        return None
    return write_split_spec(split_dir, SplitRules(seed=seed))


def dataset_check_report(
    data_root: Path = DEFAULT_DATA_ROOT,
    *,
    include_sources: bool = False,
) -> str:
    """Human-readable presence report; works when no dataset exists."""
    scans = scan_all(data_root)
    lines = [
        f"Dataset root: {data_root}",
        f"Free disk space: {free_disk_gb(data_root):.1f} GB",
        "",
        format_presence(scans),
    ]

    missing = missing_datasets(data_root)
    if missing:
        lines.extend(
            [
                "",
                f"Missing datasets ({len(missing)}): "
                + ", ".join(spec.label for spec in missing),
                "Place each dataset under datasets/raw/<dataset>/ and run this command again.",
            ]
        )
    if include_sources:
        lines.extend(["", describe_sources()])
    return "\n".join(lines)


def load_split_records(path: Path) -> tuple[str, tuple[ImageRecord, ...]]:
    """Read a pool file written by :meth:`SplitBundle.write`."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    pool = str(payload.get("pool") or path.stem)
    records: list[ImageRecord] = []

    for item in payload.get("records", []):
        low = Path(str(item["low"]))
        high_raw = item.get("high")
        high = Path(str(high_raw)) if high_raw else None
        records.append(
            ImageRecord(
                record_id=str(item["id"]),
                dataset=str(item["dataset"]),
                group=str(item["group"]),
                low_path=low if low.is_absolute() else PROJECT_ROOT / low,
                high_path=(
                    high if high is None or high.is_absolute() else PROJECT_ROOT / high
                ),
                scene=item.get("scene"),
                has_gt=bool(item.get("has_gt", high is not None)),
            )
        )
    return pool, tuple(records)


def _prepare_image(record: ImageRecord) -> tuple[np.ndarray, np.ndarray | None]:
    low = load_rgb_float32(record.low_path)
    if record.dataset == "sice":
        low = resize_long_edge(low, SICE_LONG_EDGE)

    high = None
    if record.high_path is not None:
        high = load_rgb_float32(record.high_path)
        if record.dataset == "sice":
            high = resize_long_edge(high, SICE_LONG_EDGE)
        if high.shape != low.shape:
            high = cv2.resize(high, (low.shape[1], low.shape[0]), interpolation=cv2.INTER_AREA)

    return low, high


def preprocess_pool(
    split_file: Path,
    output_dir: Path = DEFAULT_PROCESSED_ROOT,
    *,
    crop_size: int = CROP_SIZE,
    augment: bool = False,
    darken: bool = False,
    seed: int = SEED,
    limit: int | None = None,
    overwrite: bool = False,
) -> PreprocessResult:
    """Generate crops (and optional augmentation) for one split pool."""
    pool, records = load_split_records(split_file)
    written: list[Path] = []
    failures: list[str] = []
    skipped = 0

    for index, record in enumerate(records):
        if limit is not None and index >= limit:
            break

        rng = np.random.default_rng(seed + index)
        try:
            low, high = _prepare_image(record)
            if high is not None:
                low_crop, high_crop = random_crop_pair(low, high, crop_size, rng)
            else:
                low_crop, high_crop = random_crop(low, crop_size, rng), None

            if augment:
                view_index = int(rng.integers(0, 8))
                low_crop = augment_views(low_crop)[view_index]
                if high_crop is not None:
                    high_crop = augment_views(high_crop)[view_index]
            if darken:
                low_crop = apply_darkening(low_crop, rng)
        except (PreprocessingError, ValueError) as exc:
            failures.append(f"{record.record_id}: {exc}")
            continue

        stem = f"{index:06d}_{record.low_path.stem}"
        base = output_dir / pool / record.dataset
        low_target = base / "low" / f"{stem}.png"
        high_target = base / "high" / f"{stem}.png"

        if low_target.exists() and not overwrite:
            skipped += 1
            continue

        save_png(low_crop, low_target)
        written.append(low_target)
        if high_crop is not None:
            save_png(high_crop, high_target)
            written.append(high_target)

    return PreprocessResult(
        written=tuple(written), skipped=skipped, failures=tuple(failures)
    )
