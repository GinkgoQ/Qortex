# tests/test_neuroai_zoo_cli.py
from __future__ import annotations

from typer.testing import CliRunner

from qortex.cli.app import app

runner = CliRunner()


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
    # Exit code 0 means no errors (warnings are allowed and expected from braindecode entries)
    assert "[error]" not in result.stdout.lower()


def test_zoo_show_includes_synthetic_data_notice_for_generative_entry():
    result = runner.invoke(app, ["neuroai", "zoo", "show", "monai.mednist_gan"])
    assert result.exit_code == 0
    assert "clinical_use" in result.output
    assert "prohibited" in result.output


def test_zoo_show_omits_synthetic_data_notice_for_non_generative_entry():
    result = runner.invoke(app, ["neuroai", "zoo", "show", "braindecode.EEGNet"])
    assert result.exit_code == 0
    assert "clinical_use" not in result.output
