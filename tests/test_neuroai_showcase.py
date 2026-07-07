from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qortex.neuroai.showcase import (
    ShowcaseInput,
    render_segmentation_showcase,
    render_segmentation_showcase_from_files,
    segmentation_metrics,
)


def _case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z, y, x = np.mgrid[0:18, 0:72, 0:80]
    image = (
        np.exp(-(((x - 38) / 24) ** 2 + ((y - 36) / 21) ** 2 + ((z - 9) / 7) ** 2))
        + 0.25 * np.exp(-(((x - 52) / 6) ** 2 + ((y - 31) / 5) ** 2 + ((z - 10) / 3) ** 2))
    ).astype(np.float32)
    truth = ((((x - 52) / 7) ** 2 + ((y - 31) / 6) ** 2 + ((z - 10) / 3) ** 2) <= 1).astype(np.int16)
    pred = ((((x - 51) / 8) ** 2 + ((y - 32) / 6) ** 2 + ((z - 10) / 3) ** 2) <= 1).astype(np.int16)
    return image, pred, truth


def test_segmentation_metrics_with_truth():
    _, pred, truth = _case()
    metrics = segmentation_metrics(pred, truth, {0: "background", 1: "target"})
    cls = metrics["per_class"]["1"]
    assert metrics["has_ground_truth"] is True
    assert 0.6 < cls["dice"] <= 1.0
    assert cls["tp"] > 0
    assert cls["fp"] > 0


def test_render_segmentation_showcase_writes_artifacts(tmp_path: Path):
    image, pred, truth = _case()
    artifacts = render_segmentation_showcase(
        ShowcaseInput(
            image=image,
            prediction_mask=pred,
            truth_mask=truth,
            output_dir=tmp_path,
            case_id="unit-test-case",
            model_id="local-test-segmenter",
            source_id="local-test-volume",
            class_labels={0: "background", 1: "target"},
            voxel_sizes=(1.0, 1.0, 1.5),
            metadata={"purpose": "test"},
        )
    )

    for path in (
        artifacts.board,
        artifacts.overlay,
        artifacts.mask,
        artifacts.metrics,
        artifacts.manifest,
        artifacts.area_plot,
        artifacts.source_slice,
        artifacts.error_map,
    ):
        assert path is not None
        assert path.exists(), path
        assert path.stat().st_size > 0

    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    assert manifest["case_id"] == "unit-test-case"
    assert manifest["model_id"] == "local-test-segmenter"
    assert manifest["artifacts"]["board"] == "segmentation-board.png"


def test_render_segmentation_showcase_rejects_shape_mismatch(tmp_path: Path):
    image, pred, _truth = _case()
    with pytest.raises(ValueError, match="does not match"):
        render_segmentation_showcase(
            ShowcaseInput(
                image=image,
                prediction_mask=pred[:, :-1, :],
                output_dir=tmp_path,
                case_id="bad-shape",
                model_id="local-test-segmenter",
                source_id="local-test-volume",
            )
        )


def test_render_segmentation_showcase_from_nifti_files(tmp_path: Path):
    nib = pytest.importorskip("nibabel")
    image, pred, truth = _case()
    affine = np.diag([1.0, 1.0, 1.5, 1.0])
    image_path = tmp_path / "image.nii.gz"
    pred_path = tmp_path / "prediction.nii.gz"
    truth_path = tmp_path / "truth.nii.gz"
    nib.Nifti1Image(image, affine).to_filename(image_path)
    nib.Nifti1Image(pred, affine).to_filename(pred_path)
    nib.Nifti1Image(truth, affine).to_filename(truth_path)

    artifacts = render_segmentation_showcase_from_files(
        image_path=image_path,
        prediction_mask_path=pred_path,
        truth_mask_path=truth_path,
        output_dir=tmp_path / "showcase",
        case_id="nifti-case",
        model_id="nifti-test-model",
        class_labels={0: "background", 1: "target"},
    )

    assert artifacts.board.exists()
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    assert manifest["metadata"]["image_path"] == str(image_path)
    assert manifest["voxel_sizes"] == [1.0, 1.0, 1.5]


def test_cli_render_segmentation_showcase(tmp_path: Path):
    nib = pytest.importorskip("nibabel")
    from typer.testing import CliRunner
    from qortex.cli.app import app

    image, pred, _truth = _case()
    affine = np.eye(4)
    image_path = tmp_path / "image.nii.gz"
    pred_path = tmp_path / "prediction.nii.gz"
    out_dir = tmp_path / "cli-showcase"
    nib.Nifti1Image(image, affine).to_filename(image_path)
    nib.Nifti1Image(pred, affine).to_filename(pred_path)

    result = CliRunner().invoke(
        app,
        [
            "neuroai",
            "render-segmentation-showcase",
            str(image_path),
            str(pred_path),
            str(out_dir),
            "--case-id",
            "cli-case",
            "--model-id",
            "cli-model",
            "--class-labels-json",
            '{"0":"background","1":"target"}',
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert Path(payload["board"]).exists()
    assert Path(payload["manifest"]).exists()
