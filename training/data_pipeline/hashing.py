"""64-bit perceptual hashing and exact near-duplicate search.

``imagehash.phash`` produces a 64-bit hash (8x8 low-frequency DCT signs). Finding
every pair within a Hamming radius by comparing all pairs is quadratic, so the
index uses a pigeonhole split: with four 16-bit blocks and a radius of 8, at
least one block is within distance 2 of the query. Candidates from those
buckets are then verified with an exact popcount, so the index never misses a
real match.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Iterable, Sequence
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image

from .config import HAMMING_THRESHOLD, PHASH_BITS, PHASH_HASH_SIZE

try:  # imagehash is an evaluation dependency, not a runtime one
    import imagehash
except ImportError:  # pragma: no cover - reported by the caller
    imagehash = None  # type: ignore[assignment]

DEFAULT_BLOCKS = 4


class HashingUnavailableError(RuntimeError):
    """Raised when ``imagehash`` is not installed."""


def _require_imagehash() -> None:
    if imagehash is None:
        raise HashingUnavailableError(
            "Perceptual hashing requires the 'imagehash' package. "
            "Install it with: pip install -r requirements-eval.txt"
        )


def _pack(hash_value: imagehash.ImageHash) -> int:
    bits = np.asarray(hash_value.hash, dtype=bool).flatten()
    if bits.size != PHASH_BITS:
        raise ValueError(
            f"Expected a {PHASH_BITS}-bit hash, got {bits.size} bits "
            f"(hash_size={PHASH_HASH_SIZE})."
        )

    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return value


def phash64_from_image(image: Image.Image) -> int:
    """Compute the 64-bit perceptual hash of a PIL image."""
    _require_imagehash()
    with image.convert("L") as gray:
        return _pack(imagehash.phash(gray, hash_size=PHASH_HASH_SIZE))


def phash64(path: Path) -> int:
    """Compute the 64-bit perceptual hash of an image file."""
    _require_imagehash()
    with Image.open(path) as image:
        image.load()
        return phash64_from_image(image)


def phash64_from_array(array: np.ndarray) -> int:
    """Compute the hash of an HxWx3 uint8/float array."""
    rgb = np.asarray(array)
    if rgb.dtype != np.uint8:
        rgb = np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)
    return phash64_from_image(Image.fromarray(rgb, mode="RGB"))


def hamming(left: int, right: int) -> int:
    return (int(left) ^ int(right)).bit_count()


@cache
def _offset_masks(bits: int, radius: int) -> tuple[int, ...]:
    """Every integer below ``2**bits`` whose popcount is at most ``radius``."""
    masks: set[int] = {0}
    for weight in range(1, radius + 1):
        for combination in itertools.combinations(range(bits), weight):
            value = 0
            for position in combination:
                value |= 1 << position
            masks.add(value)
    return tuple(sorted(masks))


class HammingIndex:
    """Exact nearest-neighbour search over 64-bit hashes."""

    def __init__(self, *, bits: int = PHASH_BITS, blocks: int = DEFAULT_BLOCKS) -> None:
        if bits <= 0 or blocks <= 0 or bits % blocks != 0:
            raise ValueError("bits must be positive and divisible by blocks.")
        self.bits = bits
        self.blocks = blocks
        self.block_bits = bits // blocks
        self.block_mask = (1 << self.block_bits) - 1
        self._buckets: list[dict[int, list[str]]] = [
            defaultdict(list) for _ in range(blocks)
        ]
        self._hashes: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._hashes)

    def add(self, value: int, item_id: str) -> None:
        value = int(value)
        if value < 0 or value >= (1 << self.bits):
            raise ValueError(f"Hash value {value} does not fit in {self.bits} bits.")
        existing = self._hashes.get(item_id)
        if existing is not None and existing != value:
            raise ValueError(f"Item '{item_id}' was already added with a different hash.")

        self._hashes[item_id] = value
        for block in range(self.blocks):
            self._buckets[block][self._block(value, block)].append(item_id)

    def add_many(self, items: Iterable[tuple[int, str]]) -> None:
        for value, item_id in items:
            self.add(value, item_id)

    def query(self, value: int, max_distance: int = HAMMING_THRESHOLD) -> list[tuple[str, int]]:
        """Return ``(item_id, distance)`` for every item within ``max_distance``."""
        if max_distance < 0:
            raise ValueError("max_distance must be non-negative.")
        if max_distance > self.bits:
            max_distance = self.bits

        radius = min(self.block_bits, max_distance // self.blocks)
        offsets = _offset_masks(self.block_bits, radius)

        candidates: set[str] = set()
        for block in range(self.blocks):
            base = self._block(value, block)
            bucket = self._buckets[block]
            for offset in offsets:
                candidates.update(bucket.get(base ^ offset, ()))

        matches = [
            (item_id, hamming(value, self._hashes[item_id]))
            for item_id in candidates
            if hamming(value, self._hashes[item_id]) <= max_distance
        ]
        return sorted(matches, key=lambda item: (item[1], item[0]))

    def _block(self, value: int, block: int) -> int:
        return (int(value) >> (block * self.block_bits)) & self.block_mask


def brute_force_pairs(
    values: Sequence[int], max_distance: int = HAMMING_THRESHOLD
) -> list[tuple[int, int, int]]:
    """Reference implementation used to verify the index in tests."""
    pairs: list[tuple[int, int, int]] = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            distance = hamming(values[left], values[right])
            if distance <= max_distance:
                pairs.append((left, right, distance))
    return pairs


def pairs_within(
    items: Sequence[tuple[str, int]], max_distance: int = HAMMING_THRESHOLD
) -> list[tuple[str, str, int]]:
    """Find every pair of items within ``max_distance`` using one index."""
    index = HammingIndex()
    results: list[tuple[str, str, int]] = []

    for item_id, value in items:
        for other_id, distance in index.query(value, max_distance):
            results.append((other_id, item_id, distance))
        index.add(value, item_id)

    return results
