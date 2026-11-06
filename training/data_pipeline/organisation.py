"""Dataset discovery, low/high pairing and manifest records.

Discovery is layout-tolerant but never invents a split. Each dataset is
expected at ``<data_root>/<dataset>`` (or one of its documented aliases), and
each group of a dataset at a documented relative path. A missing group is
reported, not silently skipped, and images are paired strictly by filename
stem across the low/high subdirectories.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import (
    DATASET_SPECS,
    IMAGE_EXTENSIONS,
    PROJECT_ROOT,
    DatasetGroup,
    DatasetSpec,
    Pairing,
)


@dataclass(frozen=True)
class ImageRecord:
    """One low-light image and, when available, its reference image."""

    record_id: str
    dataset: str
    group: str
    low_path: Path
    high_path: Path | None
    scene: str | None
    has_gt: bool

    @property
    def stem(self) -> str:
        return self.low_path.stem

    def relative_paths(self) -> dict[str, str | None]:
        return {
            "low": _relative(self.low_path),
            "high": _relative(self.high_path) if self.high_path is not None else None,
        }


@dataclass(frozen=True)
class DatasetScan:
    """What was found on disk for one dataset."""

    spec: DatasetSpec
    root: Path | None
    records: tuple[ImageRecord, ...]
    group_counts: dict[str, int]
    missing_groups: tuple[str, ...]
    problems: tuple[str, ...]

    @property
    def present(self) -> bool:
        return self.root is not None

    @property
    def expected_total(self) -> int:
        return sum(
            group.expected_count or 0 for group in self.spec.groups
        )

    @property
    def found_total(self) -> int:
        return len(self.records)


def _relative(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def iter_images(directory: Path) -> list[Path]:
    """Return every supported image below ``directory``, sorted for determinism."""
    if not directory.is_dir():
        return []

    files = [
        path
        for path in directory.rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files)


def _match_directory(parent: Path, names: Sequence[str]) -> Path | None:
    """Find a child directory by name, case-insensitively."""
    if not parent.is_dir():
        return None

    by_lower = {child.name.lower(): child for child in parent.iterdir() if child.is_dir()}
    for name in names:
        candidate = by_lower.get(name.lower())
        if candidate is not None:
            return candidate
    return None


def resolve_relative(root: Path, relative: str) -> Path | None:
    """Resolve a possibly nested relative path, tolerating name casing."""
    current = root
    if not relative:
        return current

    for part in Path(relative).parts:
        direct = current / part
        if direct.is_dir():
            current = direct
            continue
        matched = _match_directory(current, (part,))
        if matched is None:
            return None
        current = matched
    return current


def resolve_dataset_root(data_root: Path, spec: DatasetSpec) -> Path | None:
    """Locate a dataset directory by canonical key or documented alias."""
    names = (spec.key, *spec.aliases)
    return _match_directory(data_root, names)


def _index_by_stem(directory: Path) -> tuple[dict[str, Path], list[str]]:
    index: dict[str, Path] = {}
    problems: list[str] = []
    for path in iter_images(directory):
        stem = path.stem
        existing = index.get(stem)
        if existing is not None:
            problems.append(
                f"duplicate stem '{stem}' in {directory}: {existing.name} and {path.name}"
            )
            continue
        index[stem] = path
    return index, problems


def _paired_records(
    spec: DatasetSpec, group: DatasetGroup, group_dir: Path
) -> tuple[list[ImageRecord], list[str], int]:
    problems: list[str] = []
    low_dir = _match_directory(group_dir, group.low_names)
    high_dir = _match_directory(group_dir, group.high_names)

    if low_dir is None:
        problems.append(f"{spec.key}/{group.name}: no low-light directory found in {group_dir}")
        return [], problems, 0

    low_index, low_problems = _index_by_stem(low_dir)
    problems.extend(f"{spec.key}/{group.name}: {problem}" for problem in low_problems)

    high_index: dict[str, Path] = {}
    if high_dir is not None:
        high_index, high_problems = _index_by_stem(high_dir)
        problems.extend(f"{spec.key}/{group.name}: {problem}" for problem in high_problems)
    else:
        problems.append(f"{spec.key}/{group.name}: no reference directory found in {group_dir}")

    records: list[ImageRecord] = []
    missing_pairs = 0
    for stem in sorted(low_index):
        low_path = low_index[stem]
        high_path = high_index.get(stem)
        if high_path is None:
            missing_pairs += 1
        records.append(
            ImageRecord(
                record_id=f"{spec.key}:{group.name}:{_relative(low_path)}",
                dataset=spec.key,
                group=group.name,
                low_path=low_path,
                high_path=high_path,
                scene=low_path.parent.name if low_path.parent != low_dir else None,
                has_gt=high_path is not None,
            )
        )

    if missing_pairs:
        problems.append(
            f"{spec.key}/{group.name}: {missing_pairs} low-light image(s) have no reference"
        )

    orphan_high = sorted(set(high_index) - set(low_index))
    if orphan_high:
        problems.append(
            f"{spec.key}/{group.name}: {len(orphan_high)} reference image(s) "
            "have no low-light match"
        )

    return records, problems, missing_pairs


def _flat_records(
    spec: DatasetSpec, group: DatasetGroup, group_dir: Path
) -> tuple[list[ImageRecord], list[str], int]:
    records: list[ImageRecord] = []
    problems: list[str] = []

    for path in iter_images(group_dir):
        scene = path.parent.name if path.parent != group_dir else None
        records.append(
            ImageRecord(
                record_id=f"{spec.key}:{group.name}:{_relative(path)}",
                dataset=spec.key,
                group=group.name,
                low_path=path,
                high_path=None,
                scene=scene,
                has_gt=False,
            )
        )

    return records, problems, 0


def scan_dataset(
    spec: DatasetSpec,
    data_root: Path,
    *,
    require_root: bool = False,
) -> DatasetScan:
    """Discover one dataset. Never raises for a missing dataset."""
    root = resolve_dataset_root(data_root, spec)
    if root is None:
        if require_root:
            raise FileNotFoundError(f"Dataset '{spec.key}' not found below {data_root}.")
        return DatasetScan(
            spec=spec,
            root=None,
            records=(),
            group_counts={},
            missing_groups=tuple(group.name for group in spec.groups),
            problems=(f"dataset root not found below {data_root}",),
        )

    records: list[ImageRecord] = []
    problems: list[str] = []
    group_counts: dict[str, int] = {}
    missing: list[str] = []

    for group in spec.groups:
        group_dir = resolve_relative(root, group.relative)
        if group_dir is None:
            missing.append(group.name)
            group_counts[group.name] = 0
            problems.append(f"{spec.key}/{group.name}: directory '{group.relative}' not found")
            continue

        if group.pairing is Pairing.PAIRED_SUBDIRS:
            group_records, group_problems, _ = _paired_records(spec, group, group_dir)
        else:
            group_records, group_problems, _ = _flat_records(spec, group, group_dir)

        records.extend(group_records)
        problems.extend(group_problems)
        group_counts[group.name] = len(group_records)

    return DatasetScan(
        spec=spec,
        root=root,
        records=tuple(records),
        group_counts=group_counts,
        missing_groups=tuple(missing),
        problems=tuple(problems),
    )


def scan_all(data_root: Path, specs: Iterable[DatasetSpec] = DATASET_SPECS) -> list[DatasetScan]:
    return [scan_dataset(spec, data_root) for spec in specs]


def records_for(scans: Sequence[DatasetScan], dataset: str, group: str) -> tuple[ImageRecord, ...]:
    """Return the records of one dataset group, sorted for determinism."""
    for scan in scans:
        if scan.spec.key != dataset:
            continue
        return tuple(
            sorted(
                (record for record in scan.records if record.group == group),
                key=lambda record: record.record_id,
            )
        )
    return ()


def record_to_dict(record: ImageRecord) -> dict[str, object]:
    relative = record.relative_paths()
    return {
        "id": record.record_id,
        "dataset": record.dataset,
        "group": record.group,
        "low": relative["low"],
        "high": relative["high"],
        "scene": record.scene,
        "has_gt": record.has_gt,
    }


def write_manifest(scans: Sequence[DatasetScan], path: Path) -> Path:
    """Write one JSON object per record, plus a scan summary line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for scan in scans:
            summary = {
                "type": "dataset",
                "dataset": scan.spec.key,
                "root": _relative(scan.root),
                "group_counts": scan.group_counts,
                "missing_groups": list(scan.missing_groups),
                "problems": list(scan.problems),
            }
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
            for record in scan.records:
                payload = {"type": "record", **record_to_dict(record)}
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path
