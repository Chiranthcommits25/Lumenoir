"""Dataset statistics and cleaning reports.

The reports describe what was found and what was removed, per dataset and per
pool. They are written as JSON (machine readable) and Markdown (for the
project documentation). Nothing here deletes or rewrites an image.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

from .organisation import DatasetScan
from .splits import SplitBundle
from .validation import ImageProbe, ValidationOutcome


@dataclass(frozen=True)
class GroupStatistics:
    dataset: str
    group: str
    count: int
    expected: int | None
    missing_references: int
    min_width: int | None
    median_width: float | None
    max_width: int | None
    min_height: int | None
    median_height: float | None
    max_height: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "group": self.group,
            "count": self.count,
            "expected": self.expected,
            "missing_references": self.missing_references,
            "width": [self.min_width, self.median_width, self.max_width],
            "height": [self.min_height, self.median_height, self.max_height],
        }


@dataclass(frozen=True)
class DatasetStatistics:
    generated_at: str
    groups: tuple[GroupStatistics, ...]
    validation_valid: int
    validation_problems: dict[str, int]
    validation_examples: tuple[dict[str, str], ...]
    pool_counts: dict[str, dict[str, int]]
    pool_totals: dict[str, int]
    duplicates_removed: int
    leakage_removed: int
    count_mismatches: tuple[dict[str, object], ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "groups": [group.as_dict() for group in self.groups],
            "validation": {
                "valid_records": self.validation_valid,
                "problems_by_code": self.validation_problems,
                "examples": list(self.validation_examples),
            },
            "pools": {"counts": self.pool_counts, "totals": self.pool_totals},
            "cleaning": {
                "duplicates_removed": self.duplicates_removed,
                "leakage_removed": self.leakage_removed,
                "count_mismatches": list(self.count_mismatches),
            },
            "notes": list(self.notes),
        }

    def write(self, directory: Path) -> list[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / "cleaning_report.json"
        markdown_path = directory / "cleaning_report.md"
        json_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        markdown_path.write_text(self.to_markdown(), encoding="utf-8")
        return [json_path, markdown_path]

    def to_markdown(self) -> str:
        lines = [
            "# Dataset cleaning report",
            "",
            f"Generated: {self.generated_at}",
            "",
            "## Per-group inventory",
            "",
            "| Dataset | Group | Found | Expected | Missing references | Width min/med/max "
            "| Height min/med/max |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
        for group in self.groups:
            width = (
                f"{group.min_width}/{group.median_width}/{group.max_width}"
                if group.min_width is not None
                else "-"
            )
            height = (
                f"{group.min_height}/{group.median_height}/{group.max_height}"
                if group.min_height is not None
                else "-"
            )
            lines.append(
                f"| {group.dataset} | {group.group} | {group.count} | "
                f"{group.expected if group.expected is not None else '-'} | "
                f"{group.missing_references} | {width} | {height} |"
            )

        lines.extend(
            [
                "",
                "## Validation",
                "",
                f"Valid records: {self.validation_valid}",
                "",
            ]
        )
        if self.validation_problems:
            lines.append("| Problem code | Count |")
            lines.append("| --- | ---: |")
            for code, count in sorted(self.validation_problems.items()):
                lines.append(f"| {code} | {count} |")
            lines.append("")
            if self.validation_examples:
                lines.append("First rejected files:")
                lines.append("")
                for example in self.validation_examples:
                    lines.append(
                        f"- `{example['path']}` — {example['code']}: {example['detail']}"
                    )
                lines.append("")
        else:
            lines.extend(["No invalid files were detected.", ""])

        lines.extend(
            [
                "## Split pools",
                "",
                "| Pool | Total | Per dataset |",
                "| --- | ---: | --- |",
            ]
        )
        for pool, total in self.pool_totals.items():
            per_dataset = ", ".join(
                f"{dataset}={count}" for dataset, count in self.pool_counts.get(pool, {}).items()
            )
            lines.append(f"| {pool} | {total} | {per_dataset} |")

        lines.extend(
            [
                "",
                "## Cleaning",
                "",
                f"- Duplicate images removed from the fine-tune pool: {self.duplicates_removed}",
                f"- Train/validation images removed for test leakage: {self.leakage_removed}",
                f"- Count mismatches: {len(self.count_mismatches)}",
            ]
        )
        for mismatch in self.count_mismatches:
            lines.append(f"  - {mismatch}")

        if self.notes:
            lines.extend(["", "## Notes", ""])
            lines.extend(f"- {note}" for note in self.notes)

        lines.append("")
        return "\n".join(lines)


def _group_statistics(
    scans: tuple[DatasetScan, ...],
    probes: dict[Path, ImageProbe],
) -> tuple[GroupStatistics, ...]:
    results: list[GroupStatistics] = []

    for scan in scans:
        by_group: dict[str, list] = defaultdict(list)
        for record in scan.records:
            by_group[record.group].append(record)

        for group in scan.spec.groups:
            records = by_group.get(group.name, [])
            widths: list[int] = []
            heights: list[int] = []
            missing_references = 0

            for record in records:
                if group.is_paired and record.high_path is None:
                    missing_references += 1
                probe = probes.get(record.low_path)
                if probe is not None and probe.ok and probe.width and probe.height:
                    widths.append(probe.width)
                    heights.append(probe.height)

            results.append(
                GroupStatistics(
                    dataset=scan.spec.key,
                    group=group.name,
                    count=len(records),
                    expected=group.expected_count,
                    missing_references=missing_references,
                    min_width=min(widths) if widths else None,
                    median_width=median(widths) if widths else None,
                    max_width=max(widths) if widths else None,
                    min_height=min(heights) if heights else None,
                    median_height=median(heights) if heights else None,
                    max_height=max(heights) if heights else None,
                )
            )

    return tuple(results)


def collect_statistics(
    scans: tuple[DatasetScan, ...],
    *,
    validation: ValidationOutcome | None = None,
    bundle: SplitBundle | None = None,
) -> DatasetStatistics:
    probes = {probe.path: probe for probe in validation.probes} if validation else {}
    problems = Counter(problem.code for problem in validation.problems) if validation else {}
    examples = (
        tuple(
            {
                "record_id": problem.record_id,
                "path": str(problem.path),
                "code": problem.code,
                "detail": problem.detail,
            }
            for problem in validation.problems[:10]
        )
        if validation
        else ()
    )

    return DatasetStatistics(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        groups=_group_statistics(scans, probes),
        validation_valid=len(validation.valid) if validation else 0,
        validation_problems=dict(sorted(problems.items())),
        validation_examples=examples,
        pool_counts=bundle.counts() if bundle else {},
        pool_totals=bundle.totals() if bundle else {},
        duplicates_removed=len(bundle.duplicates) if bundle else 0,
        leakage_removed=len(bundle.leakage) if bundle else 0,
        count_mismatches=tuple(item.as_dict() for item in bundle.count_mismatches)
        if bundle
        else (),
        notes=bundle.notes if bundle else (),
    )
