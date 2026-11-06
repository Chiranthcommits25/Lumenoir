"""Deterministic split generation with leakage filtering.

The arithmetic is fixed by ``final_decision.md`` §1.3 and encoded in
:class:`SplitRules`. This module does not choose any count: it samples the
documented subsets, asserts the dataset sizes, removes duplicates and leakage,
and writes one file per pool. A mismatch is an error unless the operator
explicitly allows it, in which case the deviation is recorded in the report.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import (
    DATASET_SPECS,
    HAMMING_THRESHOLD,
    PROJECT_ROOT,
    DatasetSpec,
    SplitRules,
)
from .leakage import (
    HashFailure,
    Match,
    deduplicate,
    find_leakage,
    hash_records,
)
from .organisation import DatasetScan, ImageRecord, record_to_dict, records_for

POOLS = ("fine_tune", "validation", "test", "unseen", "do_no_harm")

# Which duplicate wins inside the fine-tune pool.
FINE_TUNE_PRIORITY: Mapping[str, int] = {"lol_v1": 0, "sice": 1, "lol_v2_real": 2}


class SplitError(RuntimeError):
    """Base class for split generation failures."""


class SplitSpecMismatch(SplitError):
    """The data on disk does not match the specified counts."""


@dataclass(frozen=True)
class CountMismatch:
    dataset: str
    group: str
    expected: int | None
    found: int

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "group": self.group,
            "expected": self.expected,
            "found": self.found,
        }


@dataclass(frozen=True)
class SplitBundle:
    """All generated pools plus everything that was removed and why."""

    seed: int
    rules: SplitRules
    splits: dict[str, tuple[ImageRecord, ...]]
    count_mismatches: tuple[CountMismatch, ...] = ()
    duplicates: tuple[Match, ...] = ()
    leakage: tuple[Match, ...] = ()
    hash_failures: tuple[HashFailure, ...] = ()
    notes: tuple[str, ...] = ()

    def counts(self) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for pool, records in self.splits.items():
            per_dataset: dict[str, int] = defaultdict(int)
            for record in records:
                per_dataset[record.dataset] += 1
            result[pool] = dict(sorted(per_dataset.items()))
        return result

    def totals(self) -> dict[str, int]:
        return {pool: len(records) for pool, records in self.splits.items()}

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "seed": self.seed,
            "rules": asdict(self.rules),
            "counts": self.counts(),
            "totals": self.totals(),
            "count_mismatches": [item.as_dict() for item in self.count_mismatches],
            "duplicates": [match.__dict__ for match in self.duplicates],
            "leakage": [match.__dict__ for match in self.leakage],
            "hash_failures": [
                {"record_id": item.record_id, "path": str(item.path), "detail": item.detail}
                for item in self.hash_failures
            ],
            "notes": list(self.notes),
        }

    def write(self, directory: Path, *, overwrite: bool = True) -> list[Path]:
        """Write one JSON file per pool plus ``split_report.json``."""
        directory.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        for pool in POOLS:
            records = self.splits.get(pool, ())
            path = directory / f"{pool}.json"
            if path.exists() and not overwrite:
                raise SplitError(f"'{path}' already exists; pass --force to overwrite.")
            payload = {
                "pool": pool,
                "seed": self.seed,
                "count": len(records),
                "records": [record_to_dict(record) for record in records],
            }
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            written.append(path)

        report_path = directory / "split_report.json"
        report_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        written.append(report_path)
        return written


def _sample_records(
    records: Sequence[ImageRecord],
    count: int,
    seed: int,
    *,
    label: str,
) -> tuple[tuple[ImageRecord, ...], tuple[ImageRecord, ...]]:
    ordered = sorted(records, key=lambda record: record.record_id)
    if count < 0 or count > len(ordered):
        raise SplitSpecMismatch(
            f"{label}: cannot select {count} records from a pool of {len(ordered)}."
        )

    indices = sorted(random.Random(seed).sample(range(len(ordered)), count))
    chosen = tuple(ordered[index] for index in indices)
    chosen_ids = {record.record_id for record in chosen}
    remainder = tuple(record for record in ordered if record.record_id not in chosen_ids)
    return chosen, remainder


def _check_count(
    found: Sequence[ImageRecord],
    expected: int | None,
    dataset: str,
    group: str,
    mismatches: list[CountMismatch],
) -> None:
    if expected is not None and len(found) != expected:
        mismatches.append(
            CountMismatch(dataset=dataset, group=group, expected=expected, found=len(found))
        )


def _stratify(
    records: Sequence[ImageRecord],
    per_class: int,
    classes: int,
    mismatches: list[CountMismatch],
) -> tuple[ImageRecord, ...]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.scene or "unknown"].append(record)

    if len(grouped) != classes:
        mismatches.append(
            CountMismatch(
                dataset="exdark",
                group="all",
                expected=classes,
                found=len(grouped),
            )
        )

    selected: list[ImageRecord] = []
    for scene in sorted(grouped):
        ordered = sorted(grouped[scene], key=lambda record: record.record_id)
        if len(ordered) < per_class:
            mismatches.append(
                CountMismatch(
                    dataset="exdark",
                    group=scene,
                    expected=per_class,
                    found=len(ordered),
                )
            )
        selected.extend(ordered[:per_class])
    return tuple(selected)


def _assert_disjoint(splits: Mapping[str, Sequence[ImageRecord]]) -> None:
    seen: dict[str, str] = {}
    for pool, records in splits.items():
        for record in records:
            previous = seen.get(record.record_id)
            if previous is not None and previous != pool:
                raise SplitError(
                    f"Record '{record.record_id}' appears in both '{previous}' and '{pool}'."
                )
            seen[record.record_id] = pool


def build_splits(
    scans: Sequence[DatasetScan],
    rules: SplitRules | None = None,
    *,
    allow_count_mismatch: bool = False,
    detect_leakage: bool = True,
) -> SplitBundle:
    """Build every pool. Raises :class:`SplitSpecMismatch` on wrong counts."""
    rules = rules or SplitRules()
    mismatches: list[CountMismatch] = []
    notes: list[str] = []

    def take(dataset: str, group: str, expected: int | None) -> tuple[ImageRecord, ...]:
        found = records_for(scans, dataset, group)
        _check_count(found, expected, dataset, group, mismatches)
        return found

    sice_part1 = take("sice", "part1", rules.sice_part1_total)
    sice_validation, sice_fine_tune = _sample_records(
        sice_part1, rules.sice_part1_validation, rules.seed, label="sice part1"
    )
    _check_count(sice_fine_tune, rules.sice_part1_fine_tune, "sice", "part1/train", mismatches)

    lol_v1_train = take("lol_v1", "train", rules.lol_v1_train)
    lol_v1_test = take("lol_v1", "test", rules.lol_v1_test)

    real_train_all = take("lol_v2_real", "train", rules.lol_v2_real_train)
    real_validation, real_fine_tune = _sample_records(
        real_train_all,
        rules.lol_v2_real_validation,
        rules.seed,
        label="lol_v2_real validation",
    )
    real_test = take("lol_v2_real", "test", rules.lol_v2_real_test)

    syn_train_all = take("lol_v2_syn", "train", rules.lol_v2_syn_train)
    syn_validation, syn_unused = _sample_records(
        syn_train_all,
        rules.lol_v2_syn_validation,
        rules.seed,
        label="lol_v2_syn validation",
    )
    syn_test = take("lol_v2_syn", "test", rules.lol_v2_syn_test)

    sice_part2 = take("sice", "part2_lowlight", rules.sice_part2_lowlight_total)
    sice_test, _ = _sample_records(
        sice_part2, rules.sice_part2_test, rules.seed, label="sice part2 test"
    )

    sice_do_no_harm = take("sice", "do_no_harm", rules.do_no_harm)

    unseen: list[ImageRecord] = []
    for dataset, expected in rules.unseen_counts.items():
        unseen.extend(take(dataset, "all", expected))

    exdark = take("exdark", "all", rules.exdark_total)
    exdark_subset = _stratify(exdark, rules.exdark_per_class, rules.exdark_classes, mismatches)
    _check_count(exdark_subset, rules.exdark_unseen, "exdark", "all/subset", mismatches)
    unseen.extend(exdark_subset)

    for group in rules.own_groups:
        unseen.extend(take("own", group, rules.own_per_group))

    own_paired = records_for(scans, "own", "paired")
    if own_paired:
        notes.append(
            f"Own paired captures ({len(own_paired)}) are held out for pseudo-reference "
            "evaluation and are not part of the unseen pool."
        )

    if mismatches and not allow_count_mismatch:
        details = "; ".join(
            f"{m.dataset}/{m.group}: expected {m.expected}, found {m.found}"
            for m in mismatches
        )
        raise SplitSpecMismatch(
            "Dataset counts do not match final_decision.md. "
            f"{details}. Re-run with --allow-count-mismatch to record the deviation instead."
        )
    if mismatches:
        notes.append(
            "Count mismatches were explicitly allowed; see the report and do not treat "
            "the results as the specified protocol."
        )

    fine_tune = (*sice_fine_tune, *lol_v1_train, *real_fine_tune)
    validation = (*sice_validation, *real_validation, *syn_validation)
    test = (*lol_v1_test, *real_test, *syn_test, *sice_test)
    unseen_records = tuple(sorted(unseen, key=lambda record: record.record_id))

    if syn_unused:
        notes.append(
            f"{len(syn_unused)} LOL-v2-Synthetic training pairs are outside the specified "
            "fine-tune pool and are left unused."
        )

    notes.append(
        "The extreme-dark training set is generated on the fly with probability "
        "0.5 (final_decision.md §1.5); no synthetic file count is fixed here."
    )

    duplicates: tuple[Match, ...] = ()
    leakage: tuple[Match, ...] = ()
    hash_failures: tuple[HashFailure, ...] = ()

    if detect_leakage:
        test_hashes = hash_records(test)
        fine_hashes = hash_records(fine_tune)
        validation_hashes = hash_records(validation)
        hash_failures = (
            *test_hashes.failures,
            *fine_hashes.failures,
            *validation_hashes.failures,
        )

        failed_ids = {item.record_id for item in hash_failures}
        if failed_ids:
            notes.append(
                f"{len(failed_ids)} record(s) could not be hashed and were excluded from "
                "the fine-tune/validation pools."
            )

        fine_hashes_clean = tuple(
            item for item in fine_hashes.hashed if item.record.record_id not in failed_ids
        )
        validation_hashes_clean = tuple(
            item for item in validation_hashes.hashed if item.record.record_id not in failed_ids
        )

        dedupe_report = deduplicate(fine_hashes_clean, priority=FINE_TUNE_PRIORITY)
        duplicates = dedupe_report.matches

        leakage_report = find_leakage(
            (*fine_hashes_clean, *validation_hashes_clean),
            test_hashes.hashed,
        )
        leakage = leakage_report.matches
        leaked_ids = leakage_report.dropped_ids | failed_ids

        fine_tune = tuple(
            record for record in dedupe_report.kept if record.record_id not in leaked_ids
        )

        validation = tuple(
            record for record in validation if record.record_id not in leaked_ids
        )

        notes.append(
            f"Duplicate/leakage check: {len(duplicates)} duplicate(s) removed within the "
            f"fine-tune pool, {len(leakage)} train/validation image(s) removed because they "
            f"were within Hamming {HAMMING_THRESHOLD} of a test image and SSIM > 0.9."
        )

    splits = {
        "fine_tune": tuple(sorted(fine_tune, key=lambda record: record.record_id)),
        "validation": tuple(sorted(validation, key=lambda record: record.record_id)),
        "test": tuple(sorted(test, key=lambda record: record.record_id)),
        "unseen": unseen_records,
        "do_no_harm": tuple(sorted(sice_do_no_harm, key=lambda record: record.record_id)),
    }
    _assert_disjoint(splits)

    return SplitBundle(
        seed=rules.seed,
        rules=rules,
        splits=splits,
        count_mismatches=tuple(mismatches),
        duplicates=duplicates,
        leakage=leakage,
        hash_failures=hash_failures,
        notes=tuple(notes),
    )


def _dataset_specs_as_dicts(
    specs: Iterable[DatasetSpec] = DATASET_SPECS,
) -> list[dict[str, object]]:
    return [
        {
            "key": spec.key,
            "label": spec.label,
            "role": spec.role.value,
            "gt_available": spec.gt_available,
            "aliases": list(spec.aliases),
            "groups": [
                {
                    "name": group.name,
                    "relative": group.relative,
                    "pairing": group.pairing.value,
                    "expected_count": group.expected_count,
                }
                for group in spec.groups
            ],
        }
        for spec in specs
    ]


def build_split_spec(rules: SplitRules | None = None) -> dict[str, object]:
    """Describe every pool and count without needing any dataset on disk."""
    rules = rules or SplitRules()
    return {
        "version": 1,
        "generated_by": "training.data_pipeline.splits",
        "seed": rules.seed,
        "rules": asdict(rules),
        "datasets": _dataset_specs_as_dicts(),
        "pools": {
            "fine_tune": {
                "seed": rules.seed,
                "composition": {
                    "sice/part1/train": rules.sice_part1_fine_tune,
                    "lol_v1/train": rules.lol_v1_train,
                    "lol_v2_real/train (excluding validation)": (
                        rules.lol_v2_real_train - rules.lol_v2_real_validation
                    ),
                },
                "base_total": rules.fine_tune_base_total,
                "augmentation": "synth_dark applied on the fly with probability 0.5",
            },
            "validation": {
                "seed": rules.seed,
                "composition": {
                    "sice/part1/validation": rules.sice_part1_validation,
                    "lol_v2_real/train (validation sample)": rules.lol_v2_real_validation,
                    "lol_v2_syn/train (validation sample)": rules.lol_v2_syn_validation,
                },
                "total": (
                    rules.sice_part1_validation
                    + rules.lol_v2_real_validation
                    + rules.lol_v2_syn_validation
                ),
            },
            "test": {
                "seed": rules.seed,
                "composition": {
                    "lol_v1/test": rules.lol_v1_test,
                    "lol_v2_real/test": rules.lol_v2_real_test,
                    "lol_v2_syn/test": rules.lol_v2_syn_test,
                    "sice/part2_lowlight (sample)": rules.sice_part2_test,
                },
                "total": (
                    rules.lol_v1_test
                    + rules.lol_v2_real_test
                    + rules.lol_v2_syn_test
                    + rules.sice_part2_test
                ),
            },
            "unseen": {
                "seed": rules.seed,
                "composition": {
                    **rules.unseen_counts,
                    "exdark (stratified)": rules.exdark_unseen,
                    **{
                        f"own/{group}": rules.own_per_group
                        for group in rules.own_groups
                    },
                },
                "total": rules.unseen_total,
            },
            "do_no_harm": {"count": rules.do_no_harm},
        },
        "duplicate_rules": {
            "hash_bits": 64,
            "phash_hamming_threshold": 8,
            "ssim_threshold": 0.9,
            "ssim_size": 64,
        },
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def write_split_spec(directory: Path, rules: SplitRules | None = None) -> Path:
    """Write ``split_spec.json``; requires no dataset on disk."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "split_spec.json"
    path.write_text(
        json.dumps(build_split_spec(rules), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def split_path(pool: str, directory: Path | None = None) -> Path:
    directory = directory or PROJECT_ROOT / "training" / "splits"
    if pool not in POOLS:
        raise SplitError(f"Unknown pool '{pool}'. Choose one of: {', '.join(POOLS)}.")
    return directory / f"{pool}.json"
