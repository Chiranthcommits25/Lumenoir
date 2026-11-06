"""Constants, dataset layouts and the exact split rules.

Every count here comes from ``final_decision.md`` §§1.1-1.6. They are data, not
defaults: a mismatch between the repository on disk and these numbers is an
error unless the operator explicitly allows it, and the deviation is then
recorded in the split report. Nothing in this module invents a count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Processing constants (final_decision.md §1.5 and §1.4)
# ---------------------------------------------------------------------------

SEED = 42

#: Training crops are 384x384 for LOL and SICE-derived samples.
CROP_SIZE = 384
#: SICE images are resized so the long edge is 1200 before cropping.
SICE_LONG_EDGE = 1200
#: Synthetic LOL images are already 384x384 and are not cropped further.
SYNTHETIC_LOL_SIZE = 384
#: Inference cap for the application (not used by dataset preparation).
APP_INFERENCE_MAX_EDGE = 1920

#: Probability of applying the darkening augmentation during fine-tuning.
DARKEN_PROBABILITY = 0.5

# ---------------------------------------------------------------------------
# Duplicate and leakage constants (final_decision.md §1.4)
# ---------------------------------------------------------------------------

PHASH_HASH_SIZE = 8
PHASH_BITS = PHASH_HASH_SIZE * PHASH_HASH_SIZE  # 64 bits
HAMMING_THRESHOLD = 8
SSIM_SIZE = 64
SSIM_THRESHOLD = 0.9

# ---------------------------------------------------------------------------
# Filesystem conventions
# ---------------------------------------------------------------------------

DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "raw"
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "datasets" / "processed"
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "datasets" / "manifests"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "training" / "splits"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "training" / "reports"

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})

DEFAULT_LOW_DIRS = ("low", "Low", "LOW", "input", "Input")
DEFAULT_HIGH_DIRS = (
    "high",
    "High",
    "HIGH",
    "target",
    "Target",
    "gt",
    "GT",
    "normal",
    "Normal",
)


class DatasetRole(StrEnum):
    """Why a dataset is used. Roles never mix within one split file."""

    FINE_TUNE = "fine_tune"
    VALIDATION = "validation"
    TEST = "test"
    UNSEEN = "unseen"
    DO_NO_HARM = "do_no_harm"


class Pairing(StrEnum):
    """How a dataset group exposes low/high images."""

    PAIRED_SUBDIRS = "paired_subdirs"
    FLAT = "flat"


class Sampling(StrEnum):
    """How a group contributes to a split pool."""

    ALL = "all"
    RANDOM = "random"
    STRATIFIED = "stratified"


@dataclass(frozen=True)
class DatasetGroup:
    """A directory inside a dataset that contains one logical split."""

    name: str
    relative: str
    pairing: Pairing = Pairing.PAIRED_SUBDIRS
    expected_count: int | None = None
    low_names: tuple[str, ...] = DEFAULT_LOW_DIRS
    high_names: tuple[str, ...] = DEFAULT_HIGH_DIRS
    description: str = ""

    @property
    def is_paired(self) -> bool:
        return self.pairing is Pairing.PAIRED_SUBDIRS


@dataclass(frozen=True)
class DatasetSpec:
    """A dataset, its expected layout and its role in the study."""

    key: str
    label: str
    groups: tuple[DatasetGroup, ...]
    role: DatasetRole
    gt_available: bool = True
    priority: int = 0
    aliases: tuple[str, ...] = ()
    notes: str = ""

    def group(self, name: str) -> DatasetGroup:
        for group in self.groups:
            if group.name == name:
                return group
        raise KeyError(f"Dataset '{self.key}' has no group '{name}'.")


# ---------------------------------------------------------------------------
# Dataset layouts
#
# The directory names are the ones the released archives use. Discovery is
# tolerant about low/high casing and about unrelated extra folders, but it
# never invents a split: if a group does not exist, it is reported missing.
# ---------------------------------------------------------------------------

DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="lol_v1",
        label="LOL-v1",
        role=DatasetRole.FINE_TUNE,
        priority=0,
        aliases=("lol", "lol-v1", "lol_v1", "lol_dataset"),
        groups=(
            DatasetGroup(
                "train",
                "our485",
                expected_count=485,
                description="485 paired training scenes (our485).",
            ),
            DatasetGroup(
                "test",
                "eval15",
                expected_count=15,
                description="15 paired evaluation scenes (eval15).",
            ),
        ),
        notes="500 pairs in total: 485 train / 15 test at 400x600.",
    ),
    DatasetSpec(
        key="lol_v2_real",
        label="LOL-v2-Real",
        role=DatasetRole.FINE_TUNE,
        priority=2,
        aliases=("lol_v2_real", "lol-v2-real", "lolv2_real", "lol_v2_real_captured"),
        groups=(
            DatasetGroup(
                "train",
                "Real_captured/Train",
                expected_count=689,
                description="689 paired training scenes from the real capture set.",
            ),
            DatasetGroup(
                "test",
                "Real_captured/Test",
                expected_count=100,
                description="100 paired test scenes from the real capture set.",
            ),
        ),
        notes="789 pairs in total: 689 train / 100 test.",
    ),
    DatasetSpec(
        key="lol_v2_syn",
        label="LOL-v2-Synthetic",
        role=DatasetRole.VALIDATION,
        priority=3,
        aliases=("lol_v2_syn", "lol-v2-syn", "lolv2_syn", "lol_v2_synthetic"),
        groups=(
            DatasetGroup(
                "train",
                "Synthetic/Train",
                expected_count=900,
                description="900 paired synthetic training scenes at 384x384.",
            ),
            DatasetGroup(
                "test",
                "Synthetic/Test",
                expected_count=100,
                description="100 paired synthetic test scenes at 384x384.",
            ),
        ),
        notes="1000 pairs in total: 900 train / 100 test.",
    ),
    DatasetSpec(
        key="sice",
        label="SICE",
        role=DatasetRole.FINE_TUNE,
        priority=1,
        aliases=("sice", "sice_dataset"),
        groups=(
            DatasetGroup(
                "part1",
                "part1",
                pairing=Pairing.FLAT,
                expected_count=3022,
                description=(
                    "SICE Part 1 images used by the Zero-DCE training split "
                    "(360 sequences, 3022 images)."
                ),
            ),
            DatasetGroup(
                "part2_lowlight",
                "part2_lowlight",
                pairing=Pairing.FLAT,
                expected_count=767,
                description=(
                    "SICE Part 2 low-light images only (the Zero-DCE test protocol). "
                    "Place the curated low-light selection in this folder."
                ),
            ),
            DatasetGroup(
                "do_no_harm",
                "do_no_harm",
                pairing=Pairing.FLAT,
                expected_count=100,
                description="100 well-lit / over-exposed SICE frames (0EV and +1EV).",
            ),
        ),
        notes=(
            "SICE itself has 589 multi-exposure sequences and 4413 images (Part 1: "
            "360 sequences, Part 2: 229). The pipeline consumes the curated "
            "subsets described by final_decision.md §1.3 rather than guessing "
            "which exposure index is low-light."
        ),
    ),
    DatasetSpec(
        key="dicm",
        label="DICM",
        role=DatasetRole.UNSEEN,
        gt_available=False,
        priority=10,
        groups=(
            DatasetGroup(
                "all",
                "",
                pairing=Pairing.FLAT,
                expected_count=64,
                description="64 no-reference images.",
            ),
        ),
    ),
    DatasetSpec(
        key="lime",
        label="LIME",
        role=DatasetRole.UNSEEN,
        gt_available=False,
        priority=10,
        groups=(
            DatasetGroup(
                "all",
                "",
                pairing=Pairing.FLAT,
                expected_count=10,
                description="10 no-reference images.",
            ),
        ),
    ),
    DatasetSpec(
        key="mef",
        label="MEF",
        role=DatasetRole.UNSEEN,
        gt_available=False,
        priority=10,
        groups=(
            DatasetGroup(
                "all",
                "",
                pairing=Pairing.FLAT,
                expected_count=17,
                description="17 no-reference images.",
            ),
        ),
    ),
    DatasetSpec(
        key="npe",
        label="NPE",
        role=DatasetRole.UNSEEN,
        gt_available=False,
        priority=10,
        groups=(
            DatasetGroup(
                "all",
                "",
                pairing=Pairing.FLAT,
                expected_count=84,
                description="84 no-reference images.",
            ),
        ),
    ),
    DatasetSpec(
        key="vv",
        label="VV",
        role=DatasetRole.UNSEEN,
        gt_available=False,
        priority=10,
        groups=(
            DatasetGroup(
                "all",
                "",
                pairing=Pairing.FLAT,
                expected_count=24,
                description="24 no-reference images.",
            ),
        ),
    ),
    DatasetSpec(
        key="exdark",
        label="ExDark",
        role=DatasetRole.UNSEEN,
        gt_available=False,
        priority=10,
        groups=(
            DatasetGroup(
                "all",
                "",
                pairing=Pairing.FLAT,
                expected_count=7363,
                description=(
                    "7363 images across 10 lighting conditions; the class is the "
                    "immediate parent folder name."
                ),
            ),
        ),
        notes="The study uses a stratified 20 images per lighting condition (200).",
    ),
    DatasetSpec(
        key="own",
        label="Own captures",
        role=DatasetRole.UNSEEN,
        gt_available=False,
        priority=20,
        groups=(
            DatasetGroup(
                "extremely_dark",
                "extremely_dark",
                pairing=Pairing.FLAT,
                expected_count=20,
                description="Phone night captures below about 5 lux.",
            ),
            DatasetGroup(
                "non_uniform",
                "non_uniform",
                pairing=Pairing.FLAT,
                expected_count=20,
                description="Backlit, street-lamp and mixed indoor/outdoor scenes.",
            ),
            DatasetGroup(
                "normal",
                "normal",
                pairing=Pairing.FLAT,
                expected_count=20,
                description="Normal or bright do-no-harm captures.",
            ),
            DatasetGroup(
                "paired",
                "paired",
                expected_count=20,
                description=(
                    "About 20 tripod scenes with a high-ISO short exposure (low) "
                    "and a low-ISO long exposure (pseudo-reference)."
                ),
            ),
        ),
    ),
)

DATASET_SPECS_BY_KEY: dict[str, DatasetSpec] = {spec.key: spec for spec in DATASET_SPECS}


@dataclass(frozen=True)
class SplitRules:
    """The exact split arithmetic from final_decision.md §1.3-§1.4."""

    seed: int = SEED

    sice_part1_total: int = 3022
    sice_part1_fine_tune: int = 2422
    sice_part1_validation: int = 600

    lol_v1_train: int = 485
    lol_v1_test: int = 15

    lol_v2_real_train: int = 689
    lol_v2_real_test: int = 100
    lol_v2_real_validation: int = 100

    lol_v2_syn_train: int = 900
    lol_v2_syn_test: int = 100
    lol_v2_syn_validation: int = 100

    sice_part2_lowlight_total: int = 767
    sice_part2_test: int = 200

    exdark_total: int = 7363
    exdark_per_class: int = 20
    exdark_classes: int = 10
    exdark_unseen: int = 200

    own_groups: tuple[str, ...] = ("extremely_dark", "non_uniform", "normal")
    own_per_group: int = 20

    do_no_harm: int = 100

    unseen_counts: dict[str, int] = field(
        default_factory=lambda: {
            "dicm": 64,
            "lime": 10,
            "mef": 17,
            "npe": 84,
            "vv": 24,
        }
    )

    @property
    def sice_part2_total(self) -> int:
        return self.sice_part2_lowlight_total

    @property
    def fine_tune_base_total(self) -> int:
        """Base fine-tune images before on-the-fly darkening.

        SICE P1 train (2422) + LOL-v1 train (485) + LOL-v2-Real train minus the
        100 validation pairs (589). The §1.3 synthetic set is produced on the
        fly with probability 0.5, so it has no fixed file count here.
        """
        return (
            self.sice_part1_fine_tune
            + self.lol_v1_train
            + (self.lol_v2_real_train - self.lol_v2_real_validation)
        )

    @property
    def unseen_total(self) -> int:
        return (
            sum(self.unseen_counts.values())
            + self.exdark_unseen
            + self.own_per_group * len(self.own_groups)
        )
