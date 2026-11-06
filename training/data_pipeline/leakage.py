"""Duplicate and cross-dataset leakage detection.

Two passes, exactly as ``final_decision.md`` §1.4 specifies:

1. 64-bit perceptual hash on every train/validation/test image; any
   train/validation image within Hamming distance <= 8 of a test image is a
   candidate for removal.
2. For those candidates, compare 64x64 grayscale copies with SSIM; a pair above
   0.9 is a duplicate and the train/validation image is dropped.

Test images are never dropped. The same two-pass rule is applied to duplicates
within the fine-tune pool, with a deterministic keep order so a rerun produces
the same result.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from .config import HAMMING_THRESHOLD, SSIM_SIZE, SSIM_THRESHOLD
from .hashing import HammingIndex, hamming, phash64
from .organisation import ImageRecord


@dataclass(frozen=True)
class HashedRecord:
    record: ImageRecord
    phash: int


@dataclass(frozen=True)
class HashFailure:
    record_id: str
    path: Path
    detail: str


@dataclass(frozen=True)
class HashOutcome:
    hashed: tuple[HashedRecord, ...]
    failures: tuple[HashFailure, ...]

    @property
    def values(self) -> tuple[tuple[str, int], ...]:
        return tuple((item.record.record_id, item.phash) for item in self.hashed)


@dataclass(frozen=True)
class Match:
    """A duplicate pair. ``dropped_id`` is the train/validation side."""

    kept_id: str
    dropped_id: str
    hamming: int
    ssim: float


@dataclass(frozen=True)
class LeakageReport:
    matches: tuple[Match, ...]
    candidates_reviewed: int

    @property
    def dropped_ids(self) -> frozenset[str]:
        return frozenset(match.dropped_id for match in self.matches)

    def dropped_by_dataset(self, records: Sequence[ImageRecord]) -> dict[str, int]:
        datasets = {record.record_id: record.dataset for record in records}
        counts: dict[str, int] = defaultdict(int)
        for match in self.matches:
            counts[datasets.get(match.dropped_id, "unknown")] += 1
        return dict(sorted(counts.items()))


@dataclass(frozen=True)
class DedupeReport:
    matches: tuple[Match, ...]
    kept: tuple[ImageRecord, ...]
    candidates_reviewed: int

    @property
    def dropped_ids(self) -> frozenset[str]:
        return frozenset(match.dropped_id for match in self.matches)


def hash_records(
    records: Iterable[ImageRecord],
    *,
    limit: int | None = None,
) -> HashOutcome:
    hashed: list[HashedRecord] = []
    failures: list[HashFailure] = []

    for index, record in enumerate(records):
        if limit is not None and index >= limit:
            break
        try:
            value = phash64(record.low_path)
        except Exception as exc:
            failures.append(HashFailure(record.record_id, record.low_path, str(exc)))
            continue
        hashed.append(HashedRecord(record=record, phash=value))

    return HashOutcome(hashed=tuple(hashed), failures=tuple(failures))


def grayscale_copy(path: Path, size: int = SSIM_SIZE) -> np.ndarray:
    """Read an image as a ``size x size`` float32 grayscale copy in [0, 1]."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image '{path}'.")
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def ssim_similarity(path_a: Path, path_b: Path, size: int = SSIM_SIZE) -> float:
    left = grayscale_copy(path_a, size)
    right = grayscale_copy(path_b, size)
    return float(structural_similarity(left, right, data_range=1.0))


def _review(
    candidate: HashedRecord,
    reference: HashedRecord,
    cache: dict[tuple[Path, Path], float],
) -> float:
    key = (candidate.record.low_path, reference.record.low_path)
    cached = cache.get(key)
    if cached is not None:
        return cached
    value = ssim_similarity(candidate.record.low_path, reference.record.low_path)
    cache[key] = value
    return value


def find_leakage(
    train_val: Sequence[HashedRecord],
    test: Sequence[HashedRecord],
    *,
    max_distance: int = HAMMING_THRESHOLD,
    ssim_threshold: float = SSIM_THRESHOLD,
) -> LeakageReport:
    """Return every train/validation record that leaks into the test pool."""
    index = HammingIndex()
    index.add_many((item.phash, item.record.record_id) for item in test)
    by_id = {item.record.record_id: item for item in test}

    ssim_cache: dict[tuple[Path, Path], float] = {}
    matches: list[Match] = []
    dropped: set[str] = set()
    reviewed = 0

    for candidate in train_val:
        if candidate.record.record_id in dropped:
            continue
        for test_id, distance in index.query(candidate.phash, max_distance):
            reviewed += 1
            reference = by_id[test_id]
            similarity = _review(candidate, reference, ssim_cache)
            if similarity > ssim_threshold:
                matches.append(
                    Match(
                        kept_id=test_id,
                        dropped_id=candidate.record.record_id,
                        hamming=distance,
                        ssim=similarity,
                    )
                )
                dropped.add(candidate.record.record_id)
                break

    return LeakageReport(matches=tuple(matches), candidates_reviewed=reviewed)


def deduplicate(
    hashed: Sequence[HashedRecord],
    *,
    priority: Mapping[str, int] | None = None,
    max_distance: int = HAMMING_THRESHOLD,
    ssim_threshold: float = SSIM_THRESHOLD,
) -> DedupeReport:
    """Remove near-duplicates inside one pool, keeping the priority dataset."""
    priority = priority or {}

    ordered = sorted(
        hashed,
        key=lambda item: (priority.get(item.record.dataset, 99), item.record.record_id),
    )

    index = HammingIndex()
    by_id: dict[str, HashedRecord] = {}
    ssim_cache: dict[tuple[Path, Path], float] = {}
    kept: list[ImageRecord] = []
    matches: list[Match] = []
    reviewed = 0

    for item in ordered:
        duplicate_of: Match | None = None
        for other_id, distance in index.query(item.phash, max_distance):
            reviewed += 1
            similarity = _review(item, by_id[other_id], ssim_cache)
            if similarity > ssim_threshold:
                duplicate_of = Match(
                    kept_id=other_id,
                    dropped_id=item.record.record_id,
                    hamming=distance,
                    ssim=similarity,
                )
                break

        if duplicate_of is not None:
            matches.append(duplicate_of)
            continue

        kept.append(item.record)
        by_id[item.record.record_id] = item
        index.add(item.phash, item.record.record_id)

    return DedupeReport(
        matches=tuple(matches),
        kept=tuple(kept),
        candidates_reviewed=reviewed,
    )


def exact_duplicate_groups(hashed: Sequence[HashedRecord]) -> dict[int, list[str]]:
    """Group records that share an identical 64-bit hash."""
    groups: dict[int, list[str]] = defaultdict(list)
    for item in hashed:
        groups[item.phash].append(item.record.record_id)
    return {value: ids for value, ids in groups.items() if len(ids) > 1}


def closest_distance(left: int, right: int) -> int:
    """Small helper used by tests and reports."""
    return hamming(left, right)
