"""Official dataset sources, licences and download guidance.

No dataset is downloaded automatically. Every URL here was verified against the
project page during implementation; entries that could not be confirmed are
marked ``verified=False`` and carry no download URL. Registration, licence and
redistribution terms stay with the dataset authors.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import DATASET_SPECS, DatasetSpec
from .organisation import DatasetScan, resolve_dataset_root


@dataclass(frozen=True)
class DatasetSource:
    """Where a dataset comes from and what the operator must do to obtain it."""

    key: str
    label: str
    access: str
    licence: str
    approx_size: str
    expected_layout: str
    official_page: str | None = None
    verified: bool = False
    mirrors: tuple[str, ...] = ()
    notes: str = ""

    @property
    def manual_download(self) -> bool:
        return self.official_page is None or not self.verified


DATASET_SOURCES: tuple[DatasetSource, ...] = (
    DatasetSource(
        key="lol_v1",
        label="LOL-v1",
        official_page="https://daooshee.github.io/BMVC2018website/",
        verified=True,
        access=(
            "Direct download links on the project page (Google Drive and Baidu Pan). "
            "No registration. Download 'LOw Light paired dataset (LOL)'."
        ),
        licence="Research use; check the project page and cite Wei et al., BMVC 2018.",
        approx_size="Tens of MB (500 pairs, 400x600).",
        expected_layout=(
            "datasets/raw/lol_v1/our485/{low,high}/ and "
            "datasets/raw/lol_v1/eval15/{low,high}/"
        ),
        notes="The archive extracts as our485/ and eval15/ with low/ and high/ inside.",
    ),
    DatasetSource(
        key="lol_v2_real",
        label="LOL-v2-Real",
        access=(
            "Distributed by the LOL-v2 benchmark authors; obtain the Real_captured "
            "archive from the dataset page referenced by the paper or a mirror you trust. "
            "This project ships no downloader."
        ),
        licence="Research use; cite the LOL-v2 paper. Verify before redistributing.",
        approx_size="Hundreds of MB (789 pairs).",
        expected_layout=(
            "datasets/raw/lol_v2_real/Real_captured/{Train,Test}/{low,high}/"
        ),
        official_page=None,
        verified=False,
        mirrors=(
            "https://huggingface.co/datasets/okhater/lolv2-real",
        ),
        notes="Mirror listed for convenience only; it is not the authors' page.",
    ),
    DatasetSource(
        key="lol_v2_syn",
        label="LOL-v2-Synthetic",
        access=(
            "Distributed with LOL-v2; obtain the Synthetic archive from the dataset "
            "page or a mirror you trust. This project ships no downloader."
        ),
        licence="Research use; cite the LOL-v2 paper. Verify before redistributing.",
        approx_size="Hundreds of MB (1000 pairs, 384x384).",
        expected_layout=(
            "datasets/raw/lol_v2_syn/Synthetic/{Train,Test}/{low,high}/"
        ),
        official_page=None,
        verified=False,
        mirrors=(
            "https://huggingface.co/datasets/okhater/lolv2-synthetic",
        ),
        notes="Mirror listed for convenience only; it is not the authors' page.",
    ),
    DatasetSource(
        key="sice",
        label="SICE",
        official_page="https://github.com/csjcai/SICE",
        verified=True,
        access=(
            "Download links are in the repository README (Baidu Pan and Google Drive). "
            "No registration. Extract Part 1 and Part 2."
        ),
        licence="Research use; cite Cai et al., TIP 2018.",
        approx_size="Large, order of 1-2 GB (589 sequences, 4413 images).",
        expected_layout=(
            "datasets/raw/sice/part1/ (SICE Part 1 images), "
            "datasets/raw/sice/part2_lowlight/ (the curated 767 low-light images), "
            "datasets/raw/sice/do_no_harm/ (100 0EV and +1EV frames)"
        ),
        notes=(
            "The pipeline does not guess which exposure index is low-light. Put the "
            "curated selections in the named folders and the counts are checked."
        ),
    ),
    DatasetSource(
        key="dicm",
        label="DICM",
        official_page="https://github.com/baidut/BIMEF",
        verified=True,
        access=(
            "The BIMEF repository collects VV, LIME, NPE, DICM and MEF and links a "
            "Google Drive archive in its README."
        ),
        licence="Research use; cite the original dataset papers.",
        approx_size="A few MB (64 images).",
        expected_layout="datasets/raw/dicm/ (flat image folder)",
    ),
    DatasetSource(
        key="lime",
        label="LIME",
        official_page="https://github.com/baidut/BIMEF",
        verified=True,
        access="Same collection as DICM (BIMEF repository, Google Drive archive).",
        licence="Research use; cite Guo et al., TIP 2017.",
        approx_size="A few MB (10 images).",
        expected_layout="datasets/raw/lime/ (flat image folder)",
    ),
    DatasetSource(
        key="mef",
        label="MEF",
        official_page="https://github.com/baidut/BIMEF",
        verified=True,
        access="Same collection as DICM (BIMEF repository, Google Drive archive).",
        licence="Research use; cite Ma et al., TIP 2015.",
        approx_size="A few MB (17 images).",
        expected_layout="datasets/raw/mef/ (flat image folder)",
    ),
    DatasetSource(
        key="npe",
        label="NPE",
        official_page="https://github.com/baidut/BIMEF",
        verified=True,
        access="Same collection as DICM (BIMEF repository, Google Drive archive).",
        licence="Research use; cite Wang et al., TIP 2013.",
        approx_size="Tens of MB (84 images).",
        expected_layout="datasets/raw/npe/ (flat image folder)",
    ),
    DatasetSource(
        key="vv",
        label="VV",
        official_page="https://github.com/baidut/BIMEF",
        verified=True,
        access="Same collection as DICM (BIMEF repository, Google Drive archive).",
        licence="Research use; cite the original dataset paper.",
        approx_size="A few MB (24 images).",
        expected_layout="datasets/raw/vv/ (flat image folder)",
    ),
    DatasetSource(
        key="exdark",
        label="ExDark",
        official_page="https://github.com/cs-chan/Exclusively-Dark-Image-Dataset",
        verified=True,
        access="Direct download from the repository. No registration.",
        licence="BSD-3-Clause per the repository; keep the licence notice with the data.",
        approx_size="Large, order of 1 GB (7363 images).",
        expected_layout=(
            "datasets/raw/exdark/<lighting-condition>/*.jpg (10 condition folders)"
        ),
        notes="The lighting condition is the immediate parent folder name.",
    ),
    DatasetSource(
        key="own",
        label="Own captures",
        access=(
            "Capture with a phone: 20 extremely dark (below about 5 lux), 20 non-uniform "
            "(backlit, street lamps, mixed indoor/outdoor), 20 normal/bright, and about "
            "20 tripod scenes with a high-ISO short exposure plus a low-ISO long exposure."
        ),
        licence="You own these images; obtain consent for any identifiable people.",
        approx_size="A few hundred MB depending on phone resolution.",
        expected_layout=(
            "datasets/raw/own/{extremely_dark,non_uniform,normal}/*.jpg and "
            "datasets/raw/own/paired/{low,high}/"
        ),
        notes="Do not tune on these images; they are an unseen set.",
    ),
)

SOURCES_BY_KEY: dict[str, DatasetSource] = {source.key: source for source in DATASET_SOURCES}


def get_source(key: str) -> DatasetSource:
    try:
        return SOURCES_BY_KEY[key]
    except KeyError as exc:
        raise KeyError(f"No source registered for dataset '{key}'.") from exc


def free_disk_gb(path: Path) -> float:
    """Free space on the volume that holds ``path``."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return usage.free / (1024**3)


def describe_source(source: DatasetSource) -> str:
    lines = [f"{source.label} ({source.key})"]
    lines.append(f"  access       : {source.access}")
    lines.append(f"  licence      : {source.licence}")
    lines.append(f"  size         : {source.approx_size}")
    lines.append(f"  layout       : {source.expected_layout}")
    if source.official_page:
        lines.append(f"  official page: {source.official_page}")
    else:
        lines.append("  official page: not verified; obtain from the dataset authors")
    if source.mirrors:
        lines.append(f"  mirrors      : {', '.join(source.mirrors)}")
    if source.notes:
        lines.append(f"  notes        : {source.notes}")
    return "\n".join(lines)


def describe_sources(keys: Iterable[str] | None = None) -> str:
    if keys is None:
        return "\n\n".join(describe_source(source) for source in DATASET_SOURCES)
    return "\n\n".join(describe_source(get_source(key)) for key in keys)


def missing_datasets(
    data_root: Path, specs: Sequence[DatasetSpec] = DATASET_SPECS
) -> tuple[DatasetSpec, ...]:
    """Datasets whose root directory is not present below ``data_root``."""
    return tuple(
        spec for spec in specs if resolve_dataset_root(data_root, spec) is None
    )


def format_presence(scans: Sequence[DatasetScan]) -> str:
    lines: list[str] = []
    for scan in scans:
        status = "present" if scan.present else "MISSING"
        expected = sum(group.expected_count or 0 for group in scan.spec.groups)
        lines.append(
            f"{scan.spec.label:<18} {status:<8} found={scan.found_total:<5} "
            f"expected={expected:<5} groups={scan.group_counts}"
        )
        for problem in scan.problems:
            lines.append(f"  - {problem}")
    return "\n".join(lines)
