"""Build deterministic 80:20 train/validation manifests from paired training data.

The builder never modifies source images and never places test images in either
output split. Existing pHash/SSIM leakage functions are used to remove
train/validation candidates that leak into test or duplicate other candidates.
Every exclusion is written to ``split_report.md``.

    python -m training.data_pipeline.validation_split \
      --data-root datasets/raw \
      --split-dir training/splits
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .leakage import HashFailure, Match, deduplicate, find_leakage, hash_records
from .organisation import ImageRecord

SEED = 42
VALIDATION_FRACTION = 0.20
PAIR_KEY_RE = re.compile(r"^(?:low|normal|high|gt|input)[-_]?(\d.*)$", re.I)


@dataclass(frozen=True)
class DatasetLayout:
    key: str
    label: str
    root: str
    low: str
    high: str
    expected_pairs: int
    test_low: str
    test_high: str
    expected_test_pairs: int


LAYOUTS: tuple[DatasetLayout, ...] = (
    DatasetLayout(
        key="lol_v1",
        label="LOL-v1",
        root="LOL-v1",
        low="our485/low",
        high="our485/high",
        expected_pairs=485,
        test_low="eval15/low",
        test_high="eval15/high",
        expected_test_pairs=15,
    ),
    DatasetLayout(
        key="lol_v2_real",
        label="LOL-v2 Real",
        root="LOL-v2-Real",
        low="Train/Low",
        high="Train/Normal",
        expected_pairs=689,
        test_low="Test/Low",
        test_high="Test/Normal",
        expected_test_pairs=100,
    ),
    DatasetLayout(
        key="lol_v2_syn",
        label="LOL-v2 Synthetic",
        root="LOL-v2-Synthetic",
        low="Train/Low",
        high="Train/Normal",
        expected_pairs=900,
        test_low="Test/Low",
        test_high="Test/Normal",
        expected_test_pairs=100,
    ),
)


@dataclass(frozen=True)
class Pair:
    dataset: str
    image_id: str
    low: Path
    high: Path

    @property
    def record_id(self) -> str:
        return f"{self.dataset}:{self.image_id}"

    def as_record(self, split: str, data_root: Path) -> dict[str, str]:
        return {
            "dataset": self.dataset,
            "image_id": self.image_id,
            "low": _relative(self.low, data_root),
            "high": _relative(self.high, data_root),
            "split": split,
            "kind": "paired",
        }


@dataclass(frozen=True)
class SplitOutcome:
    train: tuple[Pair, ...]
    validation: tuple[Pair, ...]
    counts: dict[str, dict[str, int]]
    exclusions: tuple[dict[str, str], ...]
    leakage: tuple[Match, ...]
    duplicates: tuple[Match, ...]
    hash_failures: tuple[HashFailure, ...]
    test_pairs: int
    test_overlap: int
    train_validation_overlap: int
    seed: int


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return str(path.resolve())


def _pair_key(path: Path) -> str:
    match = PAIR_KEY_RE.match(path.stem)
    return match.group(1) if match else path.stem


def _images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and not path.name.startswith(".")
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )


def _paired_pairs(
    layout: DatasetLayout,
    data_root: Path,
    *,
    low_relative: str,
    high_relative: str,
    expected: int,
    split_name: str,
) -> list[Pair]:
    root = data_root / layout.root
    low_dir = root / low_relative
    high_dir = root / high_relative
    low_files = _images(low_dir)
    high_files = {_pair_key(path): path for path in _images(high_dir)}
    pairs: list[Pair] = []
    missing = 0
    for low in low_files:
        high = high_files.get(_pair_key(low))
        if high is None:
            missing += 1
            continue
        pairs.append(
            Pair(
                dataset=layout.key,
                image_id=low.relative_to(root).as_posix(),
                low=low,
                high=high,
            )
        )
    if missing:
        raise ValueError(
            f"{layout.key}/{split_name}: {missing} low-light image(s) lack a reference."
        )
    if len(pairs) != expected:
        raise ValueError(
            f"{layout.key}/{split_name}: expected {expected} pairs, found {len(pairs)}."
        )
    return sorted(pairs, key=lambda pair: pair.image_id)


def _image_records(pairs: list[Pair], split: str) -> list[ImageRecord]:
    return [
        ImageRecord(
            record_id=pair.record_id,
            dataset=pair.dataset,
            group=split,
            low_path=pair.low,
            high_path=pair.high,
            scene=None,
            has_gt=True,
        )
        for pair in pairs
    ]


def _split_pairs(
    pairs: list[Pair],
    seed: int,
) -> tuple[list[Pair], list[Pair]]:
    ordered = sorted(pairs, key=lambda pair: pair.image_id)
    n_validation = round(len(ordered) * VALIDATION_FRACTION)
    indices = set(random.Random(seed).sample(range(len(ordered)), n_validation))
    validation = [pair for index, pair in enumerate(ordered) if index in indices]
    train = [pair for index, pair in enumerate(ordered) if index not in indices]
    return train, validation


def _duplicate_components(
    pairs: list[Pair],
    matches: tuple[Match, ...],
) -> dict[str, list[Pair]]:
    """Union duplicate matches so all near-duplicates stay in one split."""
    parent = {pair.record_id: pair.record_id for pair in pairs}
    by_id = {pair.record_id: pair for pair in pairs}

    def find(item: str) -> str:
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != item:
            next_item = parent[item]
            parent[item] = root
            item = next_item
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for match in matches:
        if match.kept_id in parent and match.dropped_id in parent:
            union(match.kept_id, match.dropped_id)

    groups: dict[str, list[Pair]] = {}
    for record_id, pair in by_id.items():
        groups.setdefault(find(record_id), []).append(pair)
    return groups


def build_split(
    data_root: Path = PROJECT_ROOT / "datasets" / "raw",
    *,
    seed: int = SEED,
    layouts: tuple[DatasetLayout, ...] = LAYOUTS,
) -> SplitOutcome:
    """Build the split. Test images are used only for leakage detection.

    Duplicate-connected training/validation components are kept together. Any
    component containing an image that leaks into test is excluded as a unit and
    documented; images are never silently removed.
    """
    pool: list[Pair] = []
    tests: list[Pair] = []
    for layout in layouts:
        pool.extend(
            _paired_pairs(
                layout,
                data_root,
                low_relative=layout.low,
                high_relative=layout.high,
                expected=layout.expected_pairs,
                split_name="train",
            )
        )
        tests.extend(
            _paired_pairs(
                layout,
                data_root,
                low_relative=layout.test_low,
                high_relative=layout.test_high,
                expected=layout.expected_test_pairs,
                split_name="test",
            )
        )

    pool_records = _image_records(pool, "train")
    test_records = _image_records(tests, "test")
    pool_hashes = hash_records(pool_records)
    test_hashes = hash_records(test_records)

    test_leakage = find_leakage(pool_hashes.hashed, test_hashes.hashed)
    dedupe = deduplicate(
        pool_hashes.hashed,
        priority={"lol_v1": 0, "lol_v2_real": 1, "lol_v2_syn": 2},
    )
    components = _duplicate_components(pool, dedupe.matches)

    hash_failures = (*pool_hashes.failures, *test_hashes.failures)
    leak_ids = {match.dropped_id for match in test_leakage.matches}
    failure_ids = {failure.record_id for failure in hash_failures}
    excluded_ids = set(leak_ids) | failure_ids
    # A duplicate component is contaminated if any member leaks into test or
    # could not be hashed. Excluding the whole component avoids retaining a
    # near-identical clean image from the same scene.
    contaminated_components: list[list[Pair]] = []
    for component in components.values():
        ids = {pair.record_id for pair in component}
        if ids & excluded_ids:
            contaminated_components.append(component)
            excluded_ids.update(ids)

    excluded: dict[str, str] = {}
    for match in test_leakage.matches:
        excluded[match.dropped_id] = (
            f"test leakage: {match.kept_id} hamming={match.hamming} "
            f"ssim={match.ssim:.4f}"
        )
    for failure in hash_failures:
        excluded[failure.record_id] = f"hash failure: {failure.detail}"
    for component in contaminated_components:
        for pair in component:
            excluded.setdefault(
                pair.record_id,
                "duplicate component contains a test-leaking or unhashable image",
            )

    clean_pairs = [pair for pair in pool if pair.record_id not in excluded_ids]
    clean_groups = [
        component
        for component in components.values()
        if not ({pair.record_id for pair in component} & excluded_ids)
    ]

    targets = {
        layout.key: round(
            sum(1 for pair in clean_pairs if pair.dataset == layout.key)
            * VALIDATION_FRACTION
        )
        for layout in layouts
    }
    validation_counts: dict[str, int] = {layout.key: 0 for layout in layouts}
    validation: list[Pair] = []
    train: list[Pair] = []

    ordered_groups = sorted(
        clean_groups,
        key=lambda group: min(pair.record_id for pair in group),
    )
    random.Random(seed).shuffle(ordered_groups)
    for group in ordered_groups:
        group_counts: dict[str, int] = {}
        for pair in group:
            group_counts[pair.dataset] = group_counts.get(pair.dataset, 0) + 1
        if all(
            validation_counts[dataset] + count <= targets[dataset]
            for dataset, count in group_counts.items()
        ):
            validation.extend(group)
            for dataset, count in group_counts.items():
                validation_counts[dataset] += count
        else:
            train.extend(group)

    train.sort(key=lambda pair: (pair.dataset, pair.image_id))
    validation.sort(key=lambda pair: (pair.dataset, pair.image_id))
    counts: dict[str, dict[str, int]] = {}
    for layout in layouts:
        counts[layout.key] = {
            "source_pairs": layout.expected_pairs,
            "eligible_pairs": sum(
                1 for pair in clean_pairs if pair.dataset == layout.key
            ),
            "train": sum(1 for pair in train if pair.dataset == layout.key),
            "validation": validation_counts[layout.key],
        }

    train_ids = {pair.record_id for pair in train}
    validation_ids = {pair.record_id for pair in validation}
    test_ids = {pair.record_id for pair in tests}
    if train_ids & validation_ids:
        raise ValueError("train and validation contain overlapping records")
    if (train_ids | validation_ids) & test_ids:
        raise ValueError("train/validation contains a test record")

    validation_hashes = hash_records(_image_records(validation, "validation"))
    train_hashes = hash_records(_image_records(train, "train"))
    cross = find_leakage(validation_hashes.hashed, train_hashes.hashed)

    return SplitOutcome(
        train=tuple(train),
        validation=tuple(validation),
        counts=counts,
        exclusions=tuple(
            {"record_id": record_id, "reason": reason}
            for record_id, reason in sorted(excluded.items())
        ),
        leakage=tuple(test_leakage.matches),
        duplicates=tuple(dedupe.matches),
        hash_failures=tuple(hash_failures),
        test_pairs=len(tests),
        test_overlap=len(split_overlap(train, validation, tests)),
        train_validation_overlap=len(cross.matches),
        seed=seed,
    )


def split_overlap(train: list[Pair], validation: list[Pair], test: list[Pair]) -> set[str]:
    train_ids = {pair.record_id for pair in train}
    validation_ids = {pair.record_id for pair in validation}
    test_ids = {pair.record_id for pair in test}
    return (train_ids & test_ids) | (validation_ids & test_ids) | (train_ids & validation_ids)


def write_outputs(
    outcome: SplitOutcome,
    output_dir: Path,
    *,
    data_root: Path,
    overwrite: bool = False,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "train.json",
        output_dir / "validation.json",
        output_dir / "split_report.md",
    ]
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing split files: "
            + ", ".join(str(path) for path in existing)
            + ". Pass --force only after explicit approval."
        )

    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    for name, split, pairs in (
        ("train.json", "train", outcome.train),
        ("validation.json", "validation", outcome.validation),
    ):
        payload: dict[str, Any] = {
            "pool": split,
            "seed": outcome.seed,
            "created_at": created_at,
            "data_root": _relative(data_root, PROJECT_ROOT),
            "records": [pair.as_record(split, data_root) for pair in pairs],
        }
        (output_dir / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    lines = [
        "# Train/validation split report",
        "",
        f"- Seed: {outcome.seed}",
        f"- Validation fraction: {VALIDATION_FRACTION:.0%}",
        f"- Test pairs used for leakage checking only: {outcome.test_pairs}",
        f"- Train/validation/test exact ID overlap: {outcome.test_overlap}",
        f"- Train/validation match count: {outcome.train_validation_overlap}",
        "",
        "| Dataset | Source pairs | Eligible | Train | Validation |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, count in outcome.counts.items():
        lines.append(
            f"| {key} | {count['source_pairs']} | {count['eligible_pairs']} | "
            f"{count['train']} | {count['validation']} |"
        )
    targets = ", ".join(
        f"{key} {round(count['source_pairs'] * VALIDATION_FRACTION)}/"
        f"{count['source_pairs'] - round(count['source_pairs'] * VALIDATION_FRACTION)}"
        for key, count in outcome.counts.items()
    )
    lines.extend(
        [
            "",
            f"Source-level validation/train targets: {targets}. "
            f"{len(outcome.exclusions)} candidate pair(s) were excluded by the "
            "test-leakage audit; the actual split is 20% of eligible data per dataset.",
            "",
            "## Leakage and duplicate candidates",
            "",
            f"- Test-leakage pairs: {len(outcome.leakage)}",
            f"- Duplicate pairs inside train/validation pool: {len(outcome.duplicates)}",
            f"- Excluded records: {len(outcome.exclusions)}",
            f"- Hash failures: {len(outcome.hash_failures)}",
            "",
            "No test images were placed in train or validation. Duplicate-connected "
            "candidates were kept together in one split and were not deleted; only "
            "test-leaking components were excluded. Excluded records are listed below.",
            "",
        ]
    )
    if outcome.exclusions:
        lines.append("| Record | Reason |")
        lines.append("| --- | --- |")
        for item in outcome.exclusions:
            lines.append(f"| `{item['record_id']}` | {item['reason']} |")
    else:
        lines.append("No exclusions.")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `train.json`",
            "- `validation.json`",
            "- `split_report.md`",
            "",
        ]
    )
    (output_dir / "split_report.md").write_text("\n".join(lines), encoding="utf-8")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "datasets" / "raw")
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "training" / "splits")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true", help="Overwrite existing split files.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outcome = build_split(args.data_root, seed=args.seed)
    except (OSError, ValueError) as exc:
        print(f"Split error: {exc}", file=sys.stderr)
        return 2

    for key, count in outcome.counts.items():
        print(
            f"{key}: source={count['source_pairs']} eligible={count['eligible_pairs']} "
            f"train={count['train']} validation={count['validation']}"
        )
    print(
        f"test_leakage={len(outcome.leakage)} duplicates={len(outcome.duplicates)} "
        f"excluded={len(outcome.exclusions)} test_overlap={outcome.test_overlap}"
    )
    if args.dry_run:
        return 0
    try:
        paths = write_outputs(
            outcome,
            args.split_dir,
            data_root=args.data_root,
            overwrite=args.force,
        )
    except OSError as exc:
        print(f"Write error: {exc}", file=sys.stderr)
        return 1
    print("written: " + ", ".join(str(path) for path in paths))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
