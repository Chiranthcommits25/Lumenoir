"""Report writers for evaluation results.

JSON keeps every value and note, CSV is for spreadsheets, Markdown is for the
project documentation. Mean/std/median/P10/P90/worst-five follow §6.6.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .aggregate import Summary, summarise, worst
from .registry import all_metrics


def metric_columns(rows: Sequence[Mapping[str, object]]) -> list[str]:
    """Metric keys present in the rows, in registry order."""
    known = [definition.key for definition in all_metrics()]
    present = {key for row in rows for key in row if key != "name"}
    ordered = [key for key in known if key in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def summarise_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, Summary]:
    summaries: dict[str, Summary] = {}
    for key in metric_columns(rows):
        values = [
            float(row[key])
            for row in rows
            if row.get(key) is not None and isinstance(row[key], (int, float))
        ]
        summary = summarise(values)
        if summary is not None:
            summaries[key] = summary
    return summaries


def worst_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    n: int = 5,
) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    definitions = {definition.key: definition for definition in all_metrics()}

    for key in metric_columns(rows):
        definition = definitions.get(key)
        values = [row.get(key) for row in rows]
        numeric = [float(value) if isinstance(value, (int, float)) else None for value in values]
        worst_items = worst(
            numeric,
            n=n,
            higher_is_better=bool(definition and definition.higher_is_better),
        )
        result[key] = [
            {"name": rows[index].get("name", str(index)), "value": value}
            for index, value in worst_items
        ]
    return result


def write_json(
    rows: Sequence[Mapping[str, object]],
    path: Path,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": dict(metadata or {}),
        "rows": list(rows),
        "summary": {
            key: summary.as_dict()
            for key, summary in summarise_rows(rows).items()
        },
        "worst": worst_rows(rows),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["name", *metric_columns(rows)]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})
    return path


def write_markdown(
    rows: Sequence[Mapping[str, object]],
    path: Path,
    *,
    title: str = "LUMENOIR evaluation",
    metadata: Mapping[str, object] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    definitions = {definition.key: definition for definition in all_metrics()}
    summaries = summarise_rows(rows)

    lines = [f"# {title}", ""]
    if metadata:
        for key, value in metadata.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")

    lines.extend(
        [
            f"Images evaluated: {len(rows)}",
            "",
            "| Metric | Reference | n | Mean | Std | Median | P10 | P90 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, summary in summaries.items():
        definition = definitions.get(key)
        label = definition.label if definition else key
        requires_reference = "yes" if definition and definition.requires_reference else "no"
        lines.append(
            f"| {label} | {requires_reference} | {summary.count} | "
            f"{summary.mean:.4f} | {summary.std:.4f} | {summary.median:.4f} | "
            f"{summary.p10:.4f} | {summary.p90:.4f} |"
        )

    worst = worst_rows(rows)
    if worst:
        lines.extend(["", "## Worst five per metric", ""])
        for key, items in worst.items():
            definition = definitions.get(key)
            label = definition.label if definition else key
            rendered = ", ".join(f"{item['name']} ({item['value']:.4f})" for item in items)
            lines.append(f"- **{label}**: {rendered}")

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
