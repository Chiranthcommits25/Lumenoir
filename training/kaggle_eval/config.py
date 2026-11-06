"""Configuration and dataset catalogue for the Kaggle evaluation runner.

The catalogue describes the supported datasets and their evaluation roles. It
does not assume a directory layout: discovery proposes a layout, and an explicit
JSON configuration always wins. Paths default to Kaggle mounts but can be
overridden for local smoke tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
DEFAULT_ADDITIONAL_ROOT = KAGGLE_WORKING / "lumenoir_additional" / "extracted"
DEFAULT_WEIGHTS = PROJECT_ROOT / "backend" / "weights" / "zero_dce_plus_plus.pth"

MODES = ("original", "classical", "deep_learning", "hybrid")
PIPELINE_MODES = ("classical", "deep_learning", "hybrid")

PAIRED_METRICS = ("psnr", "ssim", "lpips")
UNPAIRED_METRICS = (
    "niqe",
    "loe",
    "noise_relative",
    "edge_preservation",
    "texture_retention",
    "hue_shift_deg",
    "new_clipping",
    "highlight_push",
    "black_crushing",
    "halo_rate",
)


class DatasetKind(StrEnum):
    PAIRED = "paired"
    UNPAIRED = "unpaired"


@dataclass(frozen=True)
class DatasetSubgroup:
    key: str
    label: str
    aliases: tuple[str, ...]


NPE_SUBGROUPS: tuple[DatasetSubgroup, ...] = (
    DatasetSubgroup(key="npe", label="NPE", aliases=("npe",)),
    DatasetSubgroup(
        key="npe_ex1",
        label="NPE-ex1",
        aliases=("npe-ex1", "npe_ex1", "npeex1"),
    ),
    DatasetSubgroup(
        key="npe_ex2",
        label="NPE-ex2",
        aliases=("npe-ex2", "npe_ex2", "npeex2"),
    ),
    DatasetSubgroup(
        key="npe_ex3",
        label="NPE-ex3",
        aliases=("npe-ex3", "npe_ex3", "npeex3"),
    ),
)
NPE_SUBGROUP_BY_KEY = {subgroup.key: subgroup for subgroup in NPE_SUBGROUPS}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    kind: DatasetKind
    aliases: tuple[str, ...]
    split_preferences: tuple[str, ...] = ()
    notes: str = ""


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="lol_v1",
        label="LOL-v1",
        kind=DatasetKind.PAIRED,
        aliases=("lol-v1", "lolv1", "lol_dataset", "lol-dataset", "lol"),
        split_preferences=("eval15", "test", "eval"),
        notes="Uses eval15/test only; our485/train is excluded by default.",
    ),
    DatasetSpec(
        key="lol_v2_real",
        label="LOL-v2 Real",
        kind=DatasetKind.PAIRED,
        aliases=("lol-v2-real", "lolv2-real", "lolv2real", "lolv2_real", "real_captured"),
        split_preferences=("test", "eval"),
        notes="Uses Test only; train pairs are excluded by default.",
    ),
    DatasetSpec(
        key="lol_v2_syn",
        label="LOL-v2 Synthetic",
        kind=DatasetKind.PAIRED,
        aliases=("lol-v2-syn", "lolv2-syn", "lolv2syn", "lolv2_syn", "synthetic"),
        split_preferences=("test", "eval"),
        notes="Uses Test only; train pairs are excluded by default.",
    ),
    DatasetSpec(
        key="sice_p1",
        label="SICE Part 1",
        kind=DatasetKind.PAIRED,
        aliases=("sice", "sice-part1", "sice_part1", "part1"),
        split_preferences=(),
        notes=(
            "Paired metrics only when explicit low/high pairs exist. SICE Part 1 "
            "is Zero-DCE++ pre-training data, so results are not held out."
        ),
    ),
    DatasetSpec(
        key="exdark",
        label="ExDark",
        kind=DatasetKind.UNPAIRED,
        aliases=("exdark", "exclusively-dark", "exclusively_dark"),
        notes="No-reference evaluation only.",
    ),
    DatasetSpec(
        key="dicm",
        label="DICM",
        kind=DatasetKind.UNPAIRED,
        aliases=("dicm",),
        notes="No-reference evaluation only.",
    ),
    DatasetSpec(
        key="npe",
        label="NPE",
        kind=DatasetKind.UNPAIRED,
        aliases=("npe",),
        notes=(
            "No-reference evaluation only. NPE, NPE-ex1, NPE-ex2 and NPE-ex3 "
            "are grouped as one logical dataset; subgroup names are retained "
            "on every record."
        ),
    ),
    DatasetSpec(
        key="lime",
        label="LIME",
        kind=DatasetKind.UNPAIRED,
        aliases=("lime",),
        notes="No-reference evaluation only.",
    ),
    DatasetSpec(
        key="mef",
        label="MEF",
        kind=DatasetKind.UNPAIRED,
        aliases=("mef",),
        notes="No-reference evaluation only.",
    ),
    DatasetSpec(
        key="vv",
        label="VV",
        kind=DatasetKind.UNPAIRED,
        aliases=("vv",),
        notes="No-reference evaluation only.",
    ),
)

SPECS_BY_KEY = {spec.key: spec for spec in DATASET_SPECS}
PAIRED_KEYS = tuple(spec.key for spec in DATASET_SPECS if spec.kind is DatasetKind.PAIRED)
UNPAIRED_KEYS = tuple(
    spec.key for spec in DATASET_SPECS if spec.kind is DatasetKind.UNPAIRED
)


def _default_input_root() -> Path:
    return KAGGLE_INPUT if KAGGLE_INPUT.is_dir() else PROJECT_ROOT / "datasets" / "raw"


def _default_additional_roots() -> tuple[Path, ...]:
    if DEFAULT_ADDITIONAL_ROOT.is_dir():
        return (DEFAULT_ADDITIONAL_ROOT,)
    return ()


def _default_output_dir() -> Path:
    if KAGGLE_WORKING.is_dir():
        return KAGGLE_WORKING / "lumenoir-evaluation"
    return PROJECT_ROOT / "training" / "reports" / "kaggle_eval"


@dataclass(frozen=True)
class PairLayout:
    """Relative low/high directories under a dataset root."""

    low: str
    high: str
    split: str = "test"


@dataclass(frozen=True)
class DatasetOverride:
    """Explicit dataset location; overrides auto-discovery."""

    root: str | None = None
    pairs: tuple[PairLayout, ...] = ()
    images: tuple[str, ...] = ()
    max_images: int | None = None
    allow_training_split: bool = False
    notes: str = ""


@dataclass(frozen=True)
class EvalConfig:
    """Everything needed for one reproducible evaluation run."""

    input_root: Path = field(default_factory=_default_input_root)
    output_dir: Path = field(default_factory=_default_output_dir)
    datasets: dict[str, DatasetOverride] = field(default_factory=dict)
    selected_datasets: tuple[str, ...] = ()
    modes: tuple[str, ...] = MODES
    seed: int = 42
    max_images: int | None = None
    weights: Path | None = None
    model_id: str = "zero_dce_plus_plus"
    device: str = "auto"
    save_visuals: bool = True
    visual_limit: int = 10
    skip_metrics: tuple[str, ...] = ()
    allow_missing_model: bool = True
    notes: tuple[str, ...] = ()
    additional_roots: tuple[Path, ...] = field(
        default_factory=_default_additional_roots
    )

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if self.max_images is not None and self.max_images <= 0:
            raise ValueError("max_images must be positive when set.")
        if self.visual_limit < 0:
            raise ValueError("visual_limit must be non-negative.")
        unknown_modes = set(self.modes) - set(MODES)
        if unknown_modes:
            raise ValueError(
                f"Unknown modes {sorted(unknown_modes)}; choose from {MODES}."
            )
        unknown_datasets = set(self.selected_datasets) - set(SPECS_BY_KEY)
        if unknown_datasets:
            raise ValueError(f"Unknown datasets {sorted(unknown_datasets)}.")
        unknown_metrics = set(self.skip_metrics) - set(PAIRED_METRICS) - set(UNPAIRED_METRICS)
        if unknown_metrics:
            raise ValueError(f"Unknown metrics to skip {sorted(unknown_metrics)}.")
        unknown_overrides = set(self.datasets) - set(SPECS_BY_KEY)
        if unknown_overrides:
            raise ValueError(f"Unknown dataset overrides {sorted(unknown_overrides)}.")
        if not self.modes:
            raise ValueError("At least one evaluation mode is required.")

    def specs(self) -> tuple[DatasetSpec, ...]:
        if not self.selected_datasets:
            return DATASET_SPECS
        selected = set(self.selected_datasets)
        return tuple(spec for spec in DATASET_SPECS if spec.key in selected)

    def search_roots(self) -> tuple[Path, ...]:
        roots: list[Path] = []
        for root in (self.input_root, *self.additional_roots):
            if root not in roots:
                roots.append(root)
        return tuple(roots)

    def paired_metrics(self) -> tuple[str, ...]:
        return tuple(
            key for key in PAIRED_METRICS if key not in set(self.skip_metrics)
        )

    def unpaired_metrics(self) -> tuple[str, ...]:
        return tuple(
            key for key in UNPAIRED_METRICS if key not in set(self.skip_metrics)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_root": str(self.input_root),
            "additional_roots": [str(root) for root in self.additional_roots],
            "output_dir": str(self.output_dir),
            "selected_datasets": list(self.selected_datasets),
            "modes": list(self.modes),
            "seed": self.seed,
            "max_images": self.max_images,
            "weights": None if self.weights is None else str(self.weights),
            "model_id": self.model_id,
            "device": self.device,
            "save_visuals": self.save_visuals,
            "visual_limit": self.visual_limit,
            "skip_metrics": list(self.skip_metrics),
            "allow_missing_model": self.allow_missing_model,
            "notes": list(self.notes),
            "datasets": {
                key: {
                    "root": override.root,
                    "pairs": [
                        {"low": pair.low, "high": pair.high, "split": pair.split}
                        for pair in override.pairs
                    ],
                    "images": list(override.images),
                    "max_images": override.max_images,
                    "allow_training_split": override.allow_training_split,
                    "notes": override.notes,
                }
                for key, override in self.datasets.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalConfig:
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a JSON object.")

        overrides: dict[str, DatasetOverride] = {}
        for key, raw in (data.get("datasets") or {}).items():
            if key not in SPECS_BY_KEY:
                raise ValueError(f"Unknown dataset override '{key}'.")
            if not isinstance(raw, dict):
                raise ValueError(f"Dataset override '{key}' must be an object.")
            pairs = tuple(
                PairLayout(
                    low=str(item["low"]),
                    high=str(item["high"]),
                    split=str(item.get("split", "test")),
                )
                for item in (raw.get("pairs") or [])
            )
            overrides[key] = DatasetOverride(
                root=raw.get("root"),
                pairs=pairs,
                images=tuple(str(item) for item in (raw.get("images") or [])),
                max_images=(
                    int(raw["max_images"]) if raw.get("max_images") is not None else None
                ),
                allow_training_split=bool(raw.get("allow_training_split", False)),
                notes=str(raw.get("notes", "")),
            )

        input_root = Path(data.get("input_root", _default_input_root()))
        additional_roots_raw = data.get("additional_roots")
        additional_roots = (
            tuple(Path(item) for item in additional_roots_raw)
            if additional_roots_raw is not None
            else _default_additional_roots()
        )
        output_dir = Path(data.get("output_dir", _default_output_dir()))
        weights_raw = data.get("weights")
        weights = Path(weights_raw) if weights_raw else None
        max_images_raw = data.get("max_images")

        return cls(
            input_root=input_root,
            additional_roots=additional_roots,
            output_dir=output_dir,
            datasets=overrides,
            selected_datasets=tuple(str(item) for item in (data.get("selected_datasets") or [])),
            modes=tuple(str(item) for item in (data.get("modes") or MODES)),
            seed=int(data.get("seed", 42)),
            max_images=int(max_images_raw) if max_images_raw is not None else None,
            weights=weights,
            model_id=str(data.get("model_id", "zero_dce_plus_plus")),
            device=str(data.get("device", "auto")),
            save_visuals=bool(data.get("save_visuals", True)),
            visual_limit=int(data.get("visual_limit", 10)),
            skip_metrics=tuple(str(item) for item in (data.get("skip_metrics") or [])),
            allow_missing_model=bool(data.get("allow_missing_model", True)),
            notes=tuple(str(item) for item in (data.get("notes") or [])),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> EvalConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def replace(self, **changes: Any) -> EvalConfig:
        from dataclasses import replace

        return replace(self, **changes)
