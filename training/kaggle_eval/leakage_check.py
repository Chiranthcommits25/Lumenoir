"""Read-only pHash + SSIM leakage check for evaluation manifests.

This consumes a manifest written by ``training.kaggle_eval`` and reports
train/validation images that are near-duplicates of test images. It never
writes split files or modifies dataset images.

    python -m training.kaggle_eval.leakage_check \
      --manifest /kaggle/working/lumenoir-evaluation/manifests/manifest.json \
      --output /kaggle/working/lumenoir-leakage.json
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.data_pipeline.leakage import find_leakage, hash_records
from training.data_pipeline.organisation import ImageRecord


def _records_from_manifest(payload: dict[str, Any]) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for dataset in payload.get("datasets", []):
        dataset_key = str(dataset.get("dataset") or "")
        for item in dataset.get("records", []):
            split = str(item.get("split") or "")
            if split not in {"train", "validation", "test"}:
                continue
            low = Path(str(item["low"]))
            high_raw = item.get("high")
            high = Path(str(high_raw)) if high_raw else None
            image_id = str(item.get("image_id") or low.name)
            records.append(
                ImageRecord(
                    record_id=f"{dataset_key}:{image_id}",
                    dataset=dataset_key,
                    group=split,
                    low_path=low,
                    high_path=high,
                    scene=item.get("subgroup"),
                    has_gt=high is not None,
                )
            )
    return records


def run_check(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = _records_from_manifest(payload)
    train_val = [record for record in records if record.group in {"train", "validation"}]
    test = [record for record in records if record.group == "test"]

    train_hashes = hash_records(train_val)
    test_hashes = hash_records(test)
    report = find_leakage(train_hashes.hashed, test_hashes.hashed)

    train_paths = {record.low_path.resolve() for record in train_val}
    test_paths = {record.low_path.resolve() for record in test}
    path_overlap = sorted(str(path) for path in train_paths & test_paths)

    result: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "manifest": str(manifest_path),
        "images_checked": len(train_hashes.hashed) + len(test_hashes.hashed),
        "training_validation_images": len(train_val),
        "test_images": len(test),
        "duplicate_pairs_found": len(report.matches),
        "images_excluded": len(report.dropped_ids),
        "train_test_overlap": path_overlap,
        "hash_failures": [
            {"record_id": item.record_id, "path": str(item.path), "detail": item.detail}
            for item in (*train_hashes.failures, *test_hashes.failures)
        ],
        "matches": [
            {
                "kept_test_id": match.kept_id,
                "dropped_train_validation_id": match.dropped_id,
                "hamming": match.hamming,
                "ssim": match.ssim,
            }
            for match in report.matches
        ],
        "note": (
            "Read-only check: no split files or images were written or modified. "
            "Thresholds are pHash Hamming <= 8 followed by 64x64 grayscale SSIM > 0.9."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    result = run_check(args.manifest, args.output)
    print(
        f"checked={result['images_checked']} "
        f"duplicates={result['duplicate_pairs_found']} "
        f"excluded={result['images_excluded']} "
        f"path_overlap={len(result['train_test_overlap'])} "
        f"report={args.output}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
