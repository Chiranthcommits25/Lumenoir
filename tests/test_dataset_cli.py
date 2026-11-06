"""Dataset pipeline CLI: works before any dataset is downloaded."""

from __future__ import annotations

import json
from pathlib import Path

from training.data_pipeline.cli import main


def test_check_reports_missing_datasets(tmp_path: Path, capsys) -> None:
    exit_code = main(["check", "--data-root", str(tmp_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "MISSING" in output
    assert "Missing datasets" in output
    assert "Free disk space" in output


def test_check_strict_fails_when_data_is_missing(tmp_path: Path) -> None:
    assert main(["check", "--data-root", str(tmp_path), "--strict"]) == 1


def test_split_spec_only_writes_counts(tmp_path: Path) -> None:
    exit_code = main(
        ["split", "--spec-only", "--split-dir", str(tmp_path), "--data-root", str(tmp_path)]
    )

    assert exit_code == 0
    payload = json.loads((tmp_path / "split_spec.json").read_text(encoding="utf-8"))
    assert payload["pools"]["fine_tune"]["base_total"] == 3496
    assert payload["duplicate_rules"]["phash_hamming_threshold"] == 8


def test_sources_command_lists_every_dataset(capsys) -> None:
    assert main(["sources"]) == 0

    output = capsys.readouterr().out
    for label in ("LOL-v1", "LOL-v2-Real", "SICE", "ExDark", "DICM", "Own captures"):
        assert label in output


def test_all_fails_cleanly_without_data(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "all",
            "--data-root",
            str(tmp_path),
            "--split-dir",
            str(tmp_path / "splits"),
            "--report-dir",
            str(tmp_path / "reports"),
            "--manifest-dir",
            str(tmp_path / "manifests"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "mismatch" in captured.err.lower()


def test_organise_writes_a_manifest(tmp_path: Path) -> None:
    exit_code = main(
        [
            "organise",
            "--data-root",
            str(tmp_path / "raw"),
            "--manifest-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "manifest.jsonl").is_file()
