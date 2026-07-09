# tests/test_neuroai_zoo_cli.py
from __future__ import annotations

import pytest
from typer.testing import CliRunner

from qortex.cli.app import app
from qortex.neuroai.models.zoo import seed_examples
from qortex.neuroai.models.zoo.registry import clear_registry

runner = CliRunner()


@pytest.fixture(autouse=True)
def _seeded_registry():
    # Other zoo test modules clear the shared in-memory registry in their own
    # autouse fixtures without restoring it, so re-seed deterministically here
    # regardless of what ran before this module.
    clear_registry()
    seed_examples.register_all()
    yield


def test_zoo_list_shows_seed_entries():
    result = runner.invoke(app, ["neuroai", "zoo", "list"])
    assert result.exit_code == 0
    assert "monai.brats_mri_segmentation" in result.stdout
    assert "braindecode.EEGNet" in result.stdout
    assert "external.totalsegmentator" in result.stdout


def test_zoo_list_filters_by_provider():
    result = runner.invoke(app, ["neuroai", "zoo", "list", "--provider", "braindecode"])
    assert result.exit_code == 0
    assert "braindecode.EEGNet" in result.stdout
    assert "monai.brats_mri_segmentation" not in result.stdout


def test_zoo_show_prints_entry_detail():
    result = runner.invoke(app, ["neuroai", "zoo", "show", "braindecode.EEGNet"])
    assert result.exit_code == 0
    assert "EEGNet" in result.stdout
    assert "braindecode" in result.stdout


def test_zoo_show_unknown_id_exits_nonzero():
    result = runner.invoke(app, ["neuroai", "zoo", "show", "nonexistent.model"])
    assert result.exit_code != 0


def test_zoo_validate_passes_on_seed_registry():
    result = runner.invoke(app, ["neuroai", "zoo", "validate"])
    assert result.exit_code == 0
    assert "0 issue" in result.stdout.lower() or "no issues" in result.stdout.lower()
