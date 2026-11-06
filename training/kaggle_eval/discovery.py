"""Dataset discovery and deterministic manifest generation.

Auto-discovery is a convenience: it proposes roots and pair layouts under the
Kaggle input directory, prefers held-out/test splits, and never falls back to a
training split. Explicit configuration always wins. Every manifest is sorted so
a rerun produces identical files.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import (
    NPE_SUBGROUPS,
    DatasetKind,
    DatasetOverride,
    DatasetSpec,
    DatasetSubgroup,
    EvalConfig,
    PairLayout,
)

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
UNSUPPORTED_IMAGE_EXTENSIONS = frozenset({".webp", ".gif", ".heic", ".avif", ".jp2"})
LOW_DIR_NAMES = ("low", "lowlight", "low_light", "low-light", "input", "inputs", "dark")
HIGH_DIR_NAMES = (
    "high",
    "highlight",
    "high_light",
    "normal",
    "gt",
    "groundtruth",
    "reference",
    "target",
)
TRAINING_TOKENS = ("train", "training", "our485")


@dataclass(frozen=True)
class EvalRecord:
    dataset: str
    image_id: str
    low: Path
    high: Path | None
    split: str
    kind: DatasetKind
    notes: str = ""
    subgroup: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "image_id": self.image_id,
            "subgroup": self.subgroup,
            "low": str(self.low),
            "high": None if self.high is None else str(self.high),
            "split": self.split,
            "kind": self.kind.value,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class DatasetManifest:
    key: str
    label: str
    kind: DatasetKind
    root: Path | None
    records: tuple[EvalRecord, ...]
    warnings: tuple[str, ...] = ()
    skipped_reason: str | None = None
    notes: str = ""
    roots: tuple[Path, ...] = ()

    @property
    def present(self) -> bool:
        return self.root is not None and self.skipped_reason is None

    @property
    def image_count(self) -> int:
        return len(self.records)

    @property
    def format_counts(self) -> dict[str, int]:
        counts = Counter(record.low.suffix.lower().lstrip(".") for record in self.records)
        return dict(sorted(counts.items()))

    @property
    def subgroup_counts(self) -> dict[str, int]:
        counts = Counter(
            record.subgroup for record in self.records if record.subgroup is not None
        )
        return dict(sorted(counts.items()))

    @property
    def source_roots(self) -> tuple[Path, ...]:
        return self.roots or ((self.root,) if self.root is not None else ())

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.key,
            "label": self.label,
            "kind": self.kind.value,
            "root": None if self.root is None else str(self.root),
            "roots": [str(root) for root in self.source_roots],
            "record_count": len(self.records),
            "image_count": self.image_count,
            "format_counts": self.format_counts,
            "subgroup_counts": self.subgroup_counts,
            "skipped_reason": self.skipped_reason,
            "warnings": list(self.warnings),
            "notes": self.notes,
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(frozen=True)
class DiscoveryResult:
    manifests: tuple[DatasetManifest, ...]
    input_root: Path
    additional_roots: tuple[Path, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "input_root": str(self.input_root),
            "additional_roots": [str(root) for root in self.additional_roots],
            "datasets": [manifest.to_dict() for manifest in self.manifests],
        }

    def warnings(self) -> tuple[str, ...]:
        return tuple(
            warning
            for manifest in self.manifests
            for warning in manifest.warnings
        )


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _is_training_path(relative: str) -> bool:
    normalised = _normalise(relative)
    return any(token in normalised for token in TRAINING_TOKENS)


def _iter_images(directory: Path) -> list[Path]:
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


def _iter_unsupported_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in UNSUPPORTED_IMAGE_EXTENSIONS
    )


def _ignored_image_warning(prefix: str, directory: Path) -> str | None:
    ignored = _iter_unsupported_images(directory)
    if not ignored:
        return None
    examples = ", ".join(str(path.relative_to(directory)) for path in ignored[:5])
    return (
        f"{prefix}: {len(ignored)} image file(s) with unsupported extensions were "
        f"ignored: {examples}"
    )


def _named_dirs(root: Path, names: Sequence[str], max_depth: int = 3) -> list[Path]:
    wanted = {_normalise(name) for name in names}
    found: list[Path] = []
    root = root.resolve()

    for current, dirs, _files in os.walk(root):
        current_path = Path(current)
        depth = len(current_path.relative_to(root).parts)
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        for name in dirs:
            if _normalise(name) in wanted:
                found.append(current_path / name)
        if depth >= max_depth:
            dirs[:] = []

    return sorted(found)


def _directory_name_index(root: Path, max_depth: int = 2) -> set[str]:
    index: set[str] = set()
    root = root.resolve()
    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        depth = len(current_path.relative_to(root).parts)
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        index.update(_normalise(name) for name in dirs)
        index.update(_normalise(Path(name).stem) for name in files)
        if depth >= max_depth:
            dirs[:] = []
    return index


def _match_child(parent: Path, names: Sequence[str]) -> Path | None:
    if not parent.is_dir():
        return None
    wanted = {_normalise(name) for name in names}
    matches = sorted(
        child
        for child in parent.iterdir()
        if child.is_dir() and _normalise(child.name) in wanted
    )
    return matches[0] if matches else None


def _root_matches(spec: DatasetSpec, root: Path) -> bool:
    name = _normalise(root.name)
    index = _directory_name_index(root)
    aliases = tuple(_normalise(alias) for alias in spec.aliases)
    name_match = any(alias and alias in name for alias in aliases)

    if spec.key == "lol_v1":
        return (
            ("eval15" in index or "our485" in index)
            and "realcaptured" not in index
            and "synthetic" not in index
        )
    if spec.key == "lol_v2_real":
        return "realcaptured" in index or (name_match and "real" in name)
    if spec.key == "lol_v2_syn":
        return "synthetic" in index or (name_match and "syn" in name)
    if spec.key == "sice_p1":
        return "part1" in index or "sice" in name
    if spec.key == "exdark":
        return "exdark" in name or "exclusivelydark" in name
    return name_match


def _candidate_roots(input_root: Path) -> list[Path]:
    if not input_root.is_dir():
        return []
    candidates: list[Path] = []
    for child in sorted(input_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        candidates.append(child)
        for grandchild in sorted(child.iterdir()):
            if grandchild.is_dir() and not grandchild.name.startswith("."):
                candidates.append(grandchild)
    return candidates


def find_dataset_root(
    search_roots: Path | str | Sequence[Path],
    spec: DatasetSpec,
) -> Path | None:
    roots: Sequence[Path]
    if isinstance(search_roots, (str, Path)):
        roots = (Path(search_roots),)
    else:
        roots = search_roots
    for search_root in roots:
        for candidate in _candidate_roots(search_root):
            if _root_matches(spec, candidate):
                return candidate.resolve()
    return None


def _npe_subgroup_for_name(name: str) -> DatasetSubgroup | None:
    normalised = _normalise(name)
    for subgroup in NPE_SUBGROUPS:
        if normalised in {_normalise(alias) for alias in subgroup.aliases}:
            return subgroup
    return None


def _find_npe_roots(
    search_roots: Sequence[Path],
    override: DatasetOverride,
) -> tuple[tuple[tuple[DatasetSubgroup, Path], ...], tuple[str, ...]]:
    """Find NPE, NPE-ex1, NPE-ex2 and NPE-ex3 roots in deterministic order."""
    found: dict[str, Path] = {}
    warnings: list[str] = []

    if override.root:
        base = Path(override.root).resolve()
        if not base.is_dir():
            return (), (f"npe: root not found: {override.root}",)
        direct = _npe_subgroup_for_name(base.name)
        if direct is not None:
            found[direct.key] = base
        else:
            for candidate in _candidate_roots(base):
                subgroup = _npe_subgroup_for_name(candidate.name)
                if subgroup is not None and subgroup.key not in found:
                    found[subgroup.key] = candidate.resolve()
            if not found:
                direct_files = [
                    path
                    for path in base.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                ]
                if direct_files:
                    found["npe"] = base
    else:
        for search_root in search_roots:
            if not search_root.is_dir():
                continue
            direct = _npe_subgroup_for_name(search_root.name)
            if direct is not None and direct.key not in found:
                found[direct.key] = search_root.resolve()
            for candidate in _candidate_roots(search_root):
                subgroup = _npe_subgroup_for_name(candidate.name)
                if subgroup is not None and subgroup.key not in found:
                    found[subgroup.key] = candidate.resolve()

    if found:
        missing = [subgroup.label for subgroup in NPE_SUBGROUPS if subgroup.key not in found]
        if missing:
            warnings.append(
                "npe: missing subgroup root(s): " + ", ".join(missing)
            )

    roots = tuple(
        (subgroup, found[subgroup.key])
        for subgroup in NPE_SUBGROUPS
        if subgroup.key in found
    )
    return roots, tuple(warnings)


def _resolve_inside(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"'{relative}' escapes dataset root '{root}'.") from exc
    return candidate


def _deduplicate_layouts(layouts: Sequence[PairLayout]) -> tuple[PairLayout, ...]:
    seen: set[tuple[str, str]] = set()
    unique: list[PairLayout] = []
    for layout in layouts:
        key = (layout.low, layout.high)
        if key in seen:
            continue
        seen.add(key)
        unique.append(layout)
    return tuple(unique)


def _auto_pair_layouts(
    root: Path, spec: DatasetSpec
) -> tuple[tuple[PairLayout, ...], tuple[str, ...]]:
    warnings: list[str] = []
    lows = _named_dirs(root, LOW_DIR_NAMES, max_depth=3)
    layouts: list[PairLayout] = []

    for low in lows:
        high = _match_child(low.parent, HIGH_DIR_NAMES)
        if high is None:
            continue
        relative_low = low.relative_to(root).as_posix()
        relative_high = high.relative_to(root).as_posix()
        if _is_training_path(relative_low):
            continue
        layouts.append(
            PairLayout(
                low=relative_low,
                high=relative_high,
                split=low.parent.name or "test",
            )
        )

    unique = _deduplicate_layouts(layouts)
    if spec.split_preferences and unique:
        preferred = tuple(
            layout
            for layout in unique
            if any(
                preference in _normalise(layout.low)
                for preference in spec.split_preferences
            )
        )
        if preferred:
            unique = preferred
        else:
            warnings.append(
                f"{spec.key}: found pair layouts but none matched a held-out split "
                f"{spec.split_preferences}; refusing to evaluate training data."
            )
            unique = ()

    return unique, tuple(warnings)


def _explicit_layouts(
    root: Path, override: DatasetOverride, spec: DatasetSpec
) -> tuple[tuple[PairLayout, ...], tuple[str, ...]]:
    warnings: list[str] = []
    layouts: list[PairLayout] = []

    for layout in override.pairs:
        if _is_training_path(layout.low) or _is_training_path(layout.high):
            if not override.allow_training_split:
                warnings.append(
                    f"{spec.key}: explicit pair '{layout.low}' looks like a training "
                    "split; skipped (set allow_training_split to override)."
                )
                continue
            warnings.append(
                f"{spec.key}: explicit training split '{layout.low}' included; "
                "these results are not held out."
            )
        low_dir = _resolve_inside(root, layout.low)
        high_dir = _resolve_inside(root, layout.high)
        if not low_dir.is_dir() or not high_dir.is_dir():
            warnings.append(
                f"{spec.key}: explicit pair directories not found: "
                f"{layout.low} / {layout.high}"
            )
            continue
        layouts.append(layout)

    return tuple(layouts), tuple(warnings)


def _paired_records(
    root: Path,
    spec: DatasetSpec,
    layouts: Sequence[PairLayout],
    *,
    kind: DatasetKind,
) -> tuple[list[EvalRecord], list[str]]:
    records: list[EvalRecord] = []
    warnings: list[str] = []
    missing_high = 0

    for layout in layouts:
        low_dir = _resolve_inside(root, layout.low)
        high_dir = _resolve_inside(root, layout.high)
        low_files = _iter_images(low_dir)
        high_files = {path.stem: path for path in _iter_images(high_dir)}

        for low in low_files:
            high = high_files.get(low.stem)
            if high is None:
                missing_high += 1
                continue
            notes = spec.notes if spec.key == "sice_p1" else ""
            records.append(
                EvalRecord(
                    dataset=spec.key,
                    image_id=low.relative_to(root).as_posix(),
                    low=low,
                    high=high,
                    split=layout.split,
                    kind=kind,
                    notes=notes,
                )
            )

    if missing_high:
        warnings.append(
            f"{spec.key}: {missing_high} low-light image(s) had no matching "
            "reference and were excluded."
        )
    return records, warnings


def _unpaired_records(
    root: Path,
    spec: DatasetSpec,
    override: DatasetOverride,
    *,
    subgroup: str | None = None,
) -> tuple[list[EvalRecord], list[str]]:
    directories = override.images or ("",)
    records: list[EvalRecord] = []
    warnings: list[str] = []
    prefix = spec.key if subgroup is None else f"{spec.key}/{subgroup}"

    for relative in directories:
        directory = _resolve_inside(root, relative)
        if not directory.is_dir():
            warnings.append(f"{prefix}: image directory '{relative}' not found.")
            continue
        images = _iter_images(directory)
        if not images:
            warnings.append(f"{prefix}: no supported images found in '{relative}'.")
        ignored = _ignored_image_warning(prefix, directory)
        if ignored:
            warnings.append(ignored)
        for path in images:
            relative_path = path.relative_to(root).as_posix()
            if subgroup is not None:
                relative_path = f"{subgroup}/{relative_path}"
            records.append(
                EvalRecord(
                    dataset=spec.key,
                    image_id=relative_path,
                    low=path,
                    high=None,
                    split="unpaired",
                    kind=DatasetKind.UNPAIRED,
                    subgroup=subgroup,
                )
            )

    return records, warnings


def _sample_records(
    records: list[EvalRecord],
    limit: int | None,
    seed: int,
    key: str,
) -> tuple[list[EvalRecord], list[str]]:
    if limit is None or limit >= len(records):
        return sorted(records, key=lambda record: record.image_id), []
    if limit <= 0:
        raise ValueError(f"max_images must be positive, got {limit}.")

    indices = sorted(random.Random(seed).sample(range(len(records)), limit))
    selected = [records[index] for index in indices]
    warning = (
        f"{key}: sampled {limit} of {len(records)} records deterministically "
        f"with seed {seed}."
    )
    return sorted(selected, key=lambda record: record.image_id), [warning]


def _build_npe_manifest(
    spec: DatasetSpec,
    override: DatasetOverride,
    config: EvalConfig,
) -> DatasetManifest:
    subgroup_roots, initial_warnings = _find_npe_roots(
        config.search_roots(), override
    )
    warnings = list(initial_warnings)
    manifest_root = (
        Path(override.root).resolve()
        if override.root
        else (subgroup_roots[0][1] if subgroup_roots else None)
    )

    if not subgroup_roots:
        return DatasetManifest(
            key=spec.key,
            label=spec.label,
            kind=spec.kind,
            root=manifest_root,
            records=(),
            warnings=tuple(warnings),
            skipped_reason="NPE subgroup roots not found",
            notes=spec.notes,
        )

    records: list[EvalRecord] = []
    for subgroup, subgroup_root in subgroup_roots:
        subgroup_records, subgroup_warnings = _unpaired_records(
            subgroup_root,
            spec,
            override,
            subgroup=subgroup.key,
        )
        records.extend(subgroup_records)
        warnings.extend(subgroup_warnings)

    limit = override.max_images if override.max_images is not None else config.max_images
    records, sample_warnings = _sample_records(records, limit, config.seed, spec.key)
    warnings.extend(sample_warnings)

    skipped_reason = None
    if not records:
        skipped_reason = "no valid evaluation records discovered"

    return DatasetManifest(
        key=spec.key,
        label=spec.label,
        kind=spec.kind,
        root=manifest_root,
        records=tuple(records),
        warnings=tuple(warnings),
        skipped_reason=skipped_reason,
        notes=spec.notes,
        roots=tuple(root for _, root in subgroup_roots),
    )


def _build_manifest(
    spec: DatasetSpec,
    override: DatasetOverride,
    config: EvalConfig,
) -> DatasetManifest:
    if spec.key == "npe":
        return _build_npe_manifest(spec, override, config)

    root = Path(override.root).resolve() if override.root else find_dataset_root(
        config.search_roots(), spec
    )
    if root is None or not root.is_dir():
        return DatasetManifest(
            key=spec.key,
            label=spec.label,
            kind=spec.kind,
            root=root,
            records=(),
            skipped_reason="dataset root not found; provide an explicit root",
            notes=spec.notes,
        )

    warnings: list[str] = []
    if spec.kind is DatasetKind.PAIRED:
        if override.pairs:
            layouts, layout_warnings = _explicit_layouts(root, override, spec)
        else:
            layouts, layout_warnings = _auto_pair_layouts(root, spec)
        warnings.extend(layout_warnings)
        records, record_warnings = _paired_records(
            root, spec, layouts, kind=spec.kind
        )
        warnings.extend(record_warnings)
        if not layouts:
            warnings.append(
                f"{spec.key}: no valid low/high pair layout found; no paired "
                "metrics will be computed."
            )
    else:
        records, record_warnings = _unpaired_records(root, spec, override)
        warnings.extend(record_warnings)

    limit = override.max_images if override.max_images is not None else config.max_images
    records, sample_warnings = _sample_records(records, limit, config.seed, spec.key)
    warnings.extend(sample_warnings)

    if spec.key == "sice_p1" and records:
        warnings.append(
            "SICE Part 1 is Zero-DCE++ pre-training data; these paired results "
            "are not held out."
        )

    skipped_reason = None
    if not records:
        skipped_reason = "no valid evaluation records discovered"

    return DatasetManifest(
        key=spec.key,
        label=spec.label,
        kind=spec.kind,
        root=root,
        records=tuple(records),
        warnings=tuple(warnings),
        skipped_reason=skipped_reason,
        notes=spec.notes,
        roots=(root,),
    )


def discover(config: EvalConfig) -> DiscoveryResult:
    """Discover every selected dataset. Missing datasets are reported, not fatal."""
    manifests = tuple(
        _build_manifest(
            spec,
            config.datasets.get(spec.key, DatasetOverride()),
            config,
        )
        for spec in config.specs()
    )
    return DiscoveryResult(
        manifests=manifests,
        input_root=config.input_root,
        additional_roots=config.additional_roots,
    )


def write_manifests(
    result: DiscoveryResult,
    output_dir: Path,
    *,
    config: EvalConfig | None = None,
) -> list[Path]:
    """Write deterministic combined and per-dataset manifests."""
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "input_root": str(result.input_root),
        "additional_roots": [str(root) for root in result.additional_roots],
        "search_roots": [
            str(root)
            for root in (result.input_root, *result.additional_roots)
        ],
        "seed": None if config is None else config.seed,
        "datasets": [manifest.to_dict() for manifest in result.manifests],
    }
    written: list[Path] = []

    combined = manifest_dir / "manifest.json"
    combined.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    written.append(combined)

    for manifest in result.manifests:
        path = manifest_dir / f"{manifest.key}.json"
        path.write_text(
            json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        written.append(path)

    report = manifest_dir / "discovery_report.json"
    report.write_text(
        json.dumps(
            {
                "input_root": str(result.input_root),
                "additional_roots": [str(root) for root in result.additional_roots],
                "warnings": list(result.warnings()),
                "datasets": {
                    manifest.key: {
                        "present": manifest.present,
                        "records": len(manifest.records),
                        "image_count": manifest.image_count,
                        "format_counts": manifest.format_counts,
                        "subgroup_counts": manifest.subgroup_counts,
                        "roots": [str(root) for root in manifest.source_roots],
                        "skipped_reason": manifest.skipped_reason,
                    }
                    for manifest in result.manifests
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    written.append(report)
    return written
