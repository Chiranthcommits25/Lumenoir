"""Command line entry point for the evaluation harness.

    python -m backend.evaluation.cli --pairs pairs.json --output reports/eval
    python -m backend.evaluation.cli --predictions out/ --reference gt/

A pairs file is a JSON list of objects with ``name``, ``prediction`` and
optional ``reference``/``original`` paths. Directory mode matches files by their
relative path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from .harness import evaluate_set
from .report import summarise_rows, write_csv, write_json, write_markdown
from .types import ImagePair

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})


def _pairs_from_json(path: Path) -> list[ImagePair]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("A pairs file must contain a JSON list.")

    pairs: list[ImagePair] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or "prediction" not in item:
            raise ValueError(f"Pair {index} must be an object with a 'prediction' key.")
        pairs.append(
            ImagePair(
                prediction=str(item["prediction"]),
                reference=str(item["reference"]) if item.get("reference") else None,
                original=str(item["original"]) if item.get("original") else None,
                name=str(item["name"]) if item.get("name") else None,
            )
        )
    return pairs


def _pairs_from_directories(
    predictions: Path,
    reference: Path | None,
    originals: Path | None,
) -> list[ImagePair]:
    if not predictions.is_dir():
        raise ValueError(f"Predictions directory '{predictions}' does not exist.")

    files = sorted(
        path
        for path in predictions.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if not files:
        raise ValueError(f"No images found below '{predictions}'.")

    pairs: list[ImagePair] = []
    for path in files:
        relative = path.relative_to(predictions)
        reference_path = reference / relative if reference else None
        original_path = originals / relative if originals else None
        has_reference = reference_path is not None and reference_path.is_file()
        has_original = original_path is not None and original_path.is_file()
        pairs.append(
            ImagePair(
                prediction=str(path),
                reference=str(reference_path) if has_reference else None,
                original=str(original_path) if has_original else None,
                name=str(relative),
            )
        )
    return pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backend.evaluation.cli",
        description="Compute LUMENOIR metrics for a set of output images.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pairs", type=Path, help="JSON list of prediction/reference pairs.")
    source.add_argument("--predictions", type=Path, help="Directory of output images.")

    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--original", type=Path, default=None, dest="originals")
    parser.add_argument("--output", type=Path, default=Path("evaluation"))
    parser.add_argument("--name", default="LUMENOIR evaluation")
    parser.add_argument("--device", default=None, help="cpu or cuda (LPIPS/NIQE).")
    parser.add_argument("--keys", default=None, help="Comma-separated metric keys.")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.device:
        os.environ["LUMENOIR_EVAL_DEVICE"] = args.device

    try:
        if args.pairs:
            pairs = _pairs_from_json(args.pairs)
        else:
            pairs = _pairs_from_directories(args.predictions, args.reference, args.originals)
    except (OSError, ValueError) as exc:
        print(f"Could not load pairs: {exc}", file=sys.stderr)
        return 1

    keys = (
        [key.strip() for key in args.keys.split(",") if key.strip()]
        if args.keys
        else None
    )

    rows = evaluate_set(pairs, keys=keys, limit=args.limit)
    if not rows:
        print("No images were evaluated.", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"device": os.getenv("LUMENOIR_EVAL_DEVICE", "cpu"), "images": len(rows)}
    json_path = write_json(rows, args.output / "evaluation.json", metadata=metadata)
    csv_path = write_csv(rows, args.output / "evaluation.csv")
    markdown_path = write_markdown(rows, args.output / "evaluation.md", title=args.name)

    summaries = summarise_rows(rows)
    print(f"Images: {len(rows)}")
    for key, summary in summaries.items():
        print(
            f"{key:<22} mean={summary.mean:.4f} std={summary.std:.4f} "
            f"median={summary.median:.4f} n={summary.count}"
        )
    print(f"Wrote {json_path}, {csv_path}, {markdown_path}")

    missing = [
        key
        for key, summary in summaries.items()
        if summary.count < len(rows)
    ]
    if missing:
        print(f"Metrics with missing values: {', '.join(missing)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
