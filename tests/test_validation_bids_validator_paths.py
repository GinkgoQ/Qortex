"""Local BIDS validation path and artifact-boundary contracts."""

from pathlib import Path

import pytest

from qortex.console.local_validation import resolve_validation_artifact
from qortex.validation.bids_validator import _parse_location


def test_validator_location_prefers_dataset_relative_path(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    absolute = dataset / "participants.tsv"

    path, line, column = _parse_location(
        {
            "path": str(absolute),
            "relativePath": "/participants.tsv",
            "line": 3,
            "column": 7,
        },
        dataset_root=dataset,
    )

    assert (path, line, column) == ("participants.tsv", 3, 7)


def test_validation_artifact_resolution_rejects_escape(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    report = run / "validation_report.json"
    report.write_text("{}", encoding="utf-8")
    (tmp_path / "outside.json").write_text("{}", encoding="utf-8")

    assert resolve_validation_artifact(run, report.name) == report
    with pytest.raises(FileNotFoundError):
        resolve_validation_artifact(run, "../outside.json")

