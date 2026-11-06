"""Read-only leakage-check tests for evaluation manifests."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from training.kaggle_eval.leakage_check import run_check


def _image(path, *, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((64, 64, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


def test_leakage_check_reports_duplicate_and_path_overlap(tmp_path) -> None:
    train = tmp_path / "train.png"
    test_duplicate = tmp_path / "test_duplicate.png"
    test_unique = tmp_path / "test_unique.png"
    _image(train, value=40)
    _image(test_duplicate, value=40)
    _image(test_unique, value=210)

    manifest = {
        "datasets": [
            {
                "dataset": "example",
                "records": [
                    {
                        "dataset": "example",
                        "image_id": "train.png",
                        "low": str(train),
                        "high": None,
                        "split": "train",
                        "kind": "paired",
                    },
                    {
                        "dataset": "example",
                        "image_id": "test_duplicate.png",
                        "low": str(test_duplicate),
                        "high": None,
                        "split": "test",
                        "kind": "paired",
                    },
                    {
                        "dataset": "example",
                        "image_id": "test_unique.png",
                        "low": str(test_unique),
                        "high": None,
                        "split": "test",
                        "kind": "paired",
                    },
                ],
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "leakage.json"

    result = run_check(manifest_path, output)

    assert result["images_checked"] == 3
    assert result["duplicate_pairs_found"] >= 1
    assert result["images_excluded"] >= 1
    assert result["matches"]
    assert output.is_file()
    assert not (tmp_path / "train.json").exists()
