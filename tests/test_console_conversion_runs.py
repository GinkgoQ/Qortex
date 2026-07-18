"""Safety and capability contracts for Atlas-backed conversion runs."""

from pathlib import Path

import pytest

from qortex.console.conversion_runs import (
    conversion_capabilities,
    resolve_conversion_artifact,
)
from qortex.console.run_inventory import persistent_run_inventory
from qortex.convert.formats import get_writer


def test_available_conversion_capabilities_have_registered_writers() -> None:
    report = conversion_capabilities()

    assert report["formats"]
    for capability in report["formats"]:
        if capability["available"]:
            assert get_writer(capability["name"]) is not None
        else:
            assert capability["missing_packages"]


def test_conversion_artifact_resolution_is_bounded_to_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    assert resolve_conversion_artifact(run_dir, "artifact.json") == artifact
    with pytest.raises(FileNotFoundError):
        resolve_conversion_artifact(run_dir, "../outside.json")


def test_persistent_run_inventory_has_an_evidence_contract() -> None:
    report = persistent_run_inventory(limit=1)

    assert isinstance(report["runs"], list)
    assert isinstance(report["scan_errors"], list)
    assert report["evidence"].startswith("Only persisted run directories")
