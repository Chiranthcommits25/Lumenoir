"""Paired data loading for Zero-DCE++ fine-tuning.

Training and validation records come from the existing materialized split
manifests. Test paths are rejected before a dataset is built, and train and
validation path sets must be disjoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from training.data_pipeline.config import PROJECT_ROOT
from training.data_pipeline.preprocessing import (
    augment_views,
    load_rgb_float32,
    random_crop_pair,
)

TEST_PATH_PARTS = frozenset({"eval15", "test"})


@dataclass(frozen=True)
class PairRecord:
    dataset: str
    image_id: str
    low: Path
    high: Path


def _resolve(raw: str, base: Path, data_root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    for candidate in (base / path, data_root / path, PROJECT_ROOT / path, Path.cwd() / path):
        if candidate.is_file():
            return candidate.resolve()
    return (base / path).resolve()


def _records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("records"), list):
        return [dict(item) for item in payload["records"]]
    if isinstance(payload.get("datasets"), list):
        records: list[dict[str, Any]] = []
        for dataset in payload["datasets"]:
            key = dataset.get("dataset") or dataset.get("key") or "validation"
            for item in dataset.get("records", []):
                record = dict(item)
                record.setdefault("dataset", key)
                records.append(record)
        return records
    raise ValueError("Manifest has no 'records' or 'datasets' list.")


def is_test_path(path: Path, data_root: Path) -> bool:
    try:
        parts = path.resolve().relative_to(data_root.resolve()).parts
    except ValueError:
        parts = path.resolve().parts
    return any(part.lower() in TEST_PATH_PARTS for part in parts)


def load_pairs(
    manifest_path: Path,
    *,
    expected_split: str,
    data_root: Path,
) -> list[PairRecord]:
    """Load a split manifest and validate every paired record."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    pool = payload.get("pool")
    pairs: list[PairRecord] = []
    seen: set[tuple[str, str]] = set()
    for index, record in enumerate(_records(payload)):
        split = record.get("split", pool)
        if split != expected_split:
            raise ValueError(
                f"{manifest_path}: record {index} has split {split!r}; "
                f"expected {expected_split!r}."
            )
        if record.get("kind") not in (None, "paired"):
            raise ValueError(f"{manifest_path}: record {index} is not paired.")
        low_raw = record.get("low")
        high_raw = record.get("high")
        if not low_raw or not high_raw:
            raise ValueError(f"{manifest_path}: record {index} is missing low/high.")
        base = manifest_path.parent
        low = _resolve(str(low_raw), base, data_root)
        high = _resolve(str(high_raw), base, data_root)
        if not low.is_file() or not high.is_file():
            raise ValueError(
                f"{manifest_path}: record {index} points to a missing file: {low} / {high}"
            )
        if is_test_path(low, data_root) or is_test_path(high, data_root):
            raise ValueError(f"{manifest_path}: test path in {expected_split}: {low}")
        with Image.open(low) as low_image, Image.open(high) as high_image:
            if low_image.size != high_image.size:
                raise ValueError(
                    f"{manifest_path}: size mismatch at record {index}: "
                    f"{low_image.size} vs {high_image.size}"
                )
        dataset = str(record.get("dataset") or "unknown")
        image_id = str(record.get("image_id") or record.get("id") or low.name)
        key = (dataset, image_id)
        if key in seen:
            raise ValueError(f"{manifest_path}: duplicate record {key!r}.")
        seen.add(key)
        pairs.append(PairRecord(dataset, image_id, low, high))
    if not pairs:
        raise ValueError(f"{manifest_path}: no usable paired records.")
    return sorted(pairs, key=lambda item: (item.dataset, item.image_id))


def load_train_validation_pairs(
    train_manifest: Path,
    validation_manifest: Path,
    data_root: Path,
) -> tuple[list[PairRecord], list[PairRecord]]:
    train = load_pairs(train_manifest, expected_split="train", data_root=data_root)
    validation = load_pairs(validation_manifest, expected_split="validation", data_root=data_root)
    train_paths = {(pair.low.resolve(), pair.high.resolve()) for pair in train}
    validation_paths = {(pair.low.resolve(), pair.high.resolve()) for pair in validation}
    overlap = train_paths & validation_paths
    if overlap:
        raise ValueError(f"Train/validation path leakage: {len(overlap)} pair(s).")
    return train, validation


def _ensure_crop_size(
    low: np.ndarray, high: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray]:
    height, width = low.shape[:2]
    if min(height, width) >= size:
        return low, high
    scale = size / min(height, width)
    target = (max(size, round(width * scale)), max(size, round(height * scale)))
    return (
        cv2.resize(low, target, interpolation=cv2.INTER_LINEAR),
        cv2.resize(high, target, interpolation=cv2.INTER_LINEAR),
    )


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


class PairedDataset(Dataset):
    """Lazy paired images; training uses one shared 384x384 crop and view."""

    def __init__(
        self,
        pairs: list[PairRecord],
        *,
        crop_size: int = 384,
        train: bool,
        seed: int = 42,
        augment: bool = True,
    ) -> None:
        self.pairs = pairs
        self.crop_size = crop_size
        self.train = train
        self.seed = seed
        self.augment = augment and train
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        low = load_rgb_float32(pair.low)
        high = load_rgb_float32(pair.high)
        if low.shape != high.shape:
            raise ValueError(f"Pair shape mismatch for {pair.image_id}.")
        if self.train:
            low, high = _ensure_crop_size(low, high, self.crop_size)
            rng = np.random.default_rng(self.seed + self.epoch * len(self.pairs) + index)
            low, high = random_crop_pair(low, high, self.crop_size, rng)
            if self.augment:
                view = int(rng.integers(0, 8))
                low = augment_views(low)[view]
                high = augment_views(high)[view]
        return {
            "low": _to_tensor(low),
            "high": _to_tensor(high),
            "dataset": pair.dataset,
            "image_id": pair.image_id,
        }


def make_loader(
    dataset: PairedDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    epoch: int,
    pin_memory: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
    )
