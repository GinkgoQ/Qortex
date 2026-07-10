from __future__ import annotations

from typer.testing import CliRunner

from qortex.cli.app import app

runner = CliRunner()


def test_prompt_predict_rejects_unknown_model_id():
    result = runner.invoke(
        app,
        ["neuroai", "prompt-predict", "input.nii.gz", "--model", "nonexistent.model", "--point", "1,2,3"],
    )
    assert result.exit_code != 0
    assert "nonexistent.model" in result.output


def test_prompt_predict_rejects_non_promptable_model():
    # braindecode.EEGNet exists in the zoo (Phase 1 seed) but is not promptable.
    result = runner.invoke(
        app,
        ["neuroai", "prompt-predict", "input.edf", "--model", "braindecode.EEGNet", "--point", "1,2,3"],
    )
    assert result.exit_code != 0
    assert "promptable" in result.output.lower()


def test_prompt_predict_parses_point_and_box_flags():
    from qortex.cli.app import _parse_prompt_points, _parse_prompt_boxes

    points = _parse_prompt_points(["1,2,3", "4,5,6"])
    assert points == [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]

    boxes = _parse_prompt_boxes(["0,0,0,10,10,10"])
    assert boxes == [(0.0, 0.0, 0.0, 10.0, 10.0, 10.0)]


def test_prompt_predict_rejects_malformed_point():
    result = runner.invoke(
        app,
        ["neuroai", "prompt-predict", "input.nii.gz", "--model", "monai.vista3d", "--point", "not-a-point"],
    )
    assert result.exit_code != 0
