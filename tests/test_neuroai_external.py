from __future__ import annotations

import os
from pathlib import Path

import pytest

from qortex.neuroai import (
    ExternalSegmentationError,
    ExternalSegmentationRequest,
    available_external_segmentation_engines,
    build_external_segmentation_command,
    run_external_segmentation,
)


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def test_external_segmentation_reports_available_engines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_executable(tmp_path / "TotalSegmentator", "#!/usr/bin/env bash\nexit 0\n")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")

    engines = available_external_segmentation_engines()

    assert engines["totalsegmentator"] is True
    assert "nnunet" in engines


def test_run_totalsegmentator_external_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_executable(
        tmp_path / "TotalSegmentator",
        """#!/usr/bin/env bash
set -euo pipefail
out=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    *) shift ;;
  esac
done
printf 'mask' > "$out"
""",
    )
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    image = tmp_path / "image.nii.gz"
    image.write_text("image", encoding="utf-8")
    output = tmp_path / "mask.nii.gz"

    result = run_external_segmentation(
        ExternalSegmentationRequest(
            engine="totalsegmentator",
            image_path=image,
            output_path=output,
            task="total",
            device="cpu",
        )
    )

    assert result.success is True
    assert output.read_text(encoding="utf-8") == "mask"
    assert result.metadata_path.exists()
    assert "TotalSegmentator" in result.command[0]


def test_build_nnunet_command_uses_results_folder_environment_not_plan_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _write_executable(tmp_path / "nnUNetv2_predict", "#!/usr/bin/env bash\nexit 0\n")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    image = tmp_path / "case_0000.nii.gz"
    image.write_text("image", encoding="utf-8")

    command = build_external_segmentation_command(
        ExternalSegmentationRequest(
            engine="nnunet",
            image_path=image,
            output_path=tmp_path / "predictions",
            model_folder=tmp_path / "nnunet-results",
            dataset_id=501,
            configuration="3d_fullres",
            trainer="nnUNetTrainer",
            plans="nnUNetPlans",
            folds=(0, 1),
        )
    )

    assert "-d" in command
    assert "501" in command
    assert "-c" in command
    assert "3d_fullres" in command
    assert command.count("-p") == 1
    assert "nnUNetPlans" in command
    assert str(tmp_path / "nnunet-results") not in command


def test_external_segmentation_missing_executable_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    image = tmp_path / "image.nii.gz"
    image.write_text("image", encoding="utf-8")

    with pytest.raises(ExternalSegmentationError, match="Required executable"):
        build_external_segmentation_command(
            ExternalSegmentationRequest(
                engine="totalsegmentator",
                image_path=image,
                output_path=tmp_path / "mask.nii.gz",
            )
        )


def test_cli_run_external_segmentation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from typer.testing import CliRunner
    from qortex.cli.app import app

    _write_executable(
        tmp_path / "TotalSegmentator",
        """#!/usr/bin/env bash
set -euo pipefail
out=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    *) shift ;;
  esac
done
printf 'mask' > "$out"
""",
    )
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    image = tmp_path / "image.nii.gz"
    output = tmp_path / "mask.nii.gz"
    image.write_text("image", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "neuroai",
            "run-external-segmentation",
            "totalsegmentator",
            str(image),
            str(output),
            "--task",
            "total",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert '"success": true' in result.output


def test_external_segmentation_reports_all_7_engines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PATH", str(tmp_path))

    engines = available_external_segmentation_engines()

    assert set(engines.keys()) == {
        "totalsegmentator", "nnunet", "synthseg", "synthstrip",
        "hdbet", "fastsurfer", "tractseg",
    }
    assert all(v is False for v in engines.values())


def test_build_synthseg_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_executable(tmp_path / "mri_synthseg", "#!/usr/bin/env bash\nexit 0\n")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    image = tmp_path / "t1.nii.gz"
    image.write_text("image", encoding="utf-8")

    command = build_external_segmentation_command(
        ExternalSegmentationRequest(
            engine="synthseg",
            image_path=image,
            output_path=tmp_path / "seg.nii.gz",
            device="cpu",
        )
    )

    assert command[0].endswith("mri_synthseg")
    assert "--i" in command and str(image) in command
    assert "--o" in command
    assert "--cpu" in command


def test_build_synthstrip_command_gpu_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_executable(tmp_path / "mri_synthstrip", "#!/usr/bin/env bash\nexit 0\n")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    image = tmp_path / "t1.nii.gz"
    image.write_text("image", encoding="utf-8")

    command = build_external_segmentation_command(
        ExternalSegmentationRequest(
            engine="synthstrip",
            image_path=image,
            output_path=tmp_path / "brain.nii.gz",
            device="cuda",
        )
    )

    assert command[0].endswith("mri_synthstrip")
    assert "-i" in command and str(image) in command
    assert "-o" in command
    assert "--gpu" in command


def test_build_hdbet_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_executable(tmp_path / "hd-bet", "#!/usr/bin/env bash\nexit 0\n")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    image = tmp_path / "t1.nii.gz"
    image.write_text("image", encoding="utf-8")

    command = build_external_segmentation_command(
        ExternalSegmentationRequest(
            engine="hdbet",
            image_path=image,
            output_path=tmp_path / "brain.nii.gz",
            device="cpu",
        )
    )

    assert command[0].endswith("hd-bet")
    assert "-i" in command and str(image) in command
    assert "-o" in command
    assert "-device" in command and "cpu" in command


def test_build_fastsurfer_command_requires_subject_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_executable(tmp_path / "run_fastsurfer.sh", "#!/usr/bin/env bash\nexit 0\n")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    image = tmp_path / "t1.nii.gz"
    image.write_text("image", encoding="utf-8")

    with pytest.raises(ExternalSegmentationError, match="subject_id"):
        build_external_segmentation_command(
            ExternalSegmentationRequest(
                engine="fastsurfer",
                image_path=image,
                output_path=tmp_path / "subjects",
            )
        )


def test_build_fastsurfer_command_with_subject_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_executable(tmp_path / "run_fastsurfer.sh", "#!/usr/bin/env bash\nexit 0\n")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    image = tmp_path / "t1.nii.gz"
    image.write_text("image", encoding="utf-8")

    command = build_external_segmentation_command(
        ExternalSegmentationRequest(
            engine="fastsurfer",
            image_path=image,
            output_path=tmp_path / "subjects",
            subject_id="sub-01",
        )
    )

    assert command[0].endswith("run_fastsurfer.sh")
    assert "--t1" in command and str(image) in command
    assert "--sid" in command and "sub-01" in command
    assert "--sd" in command


def test_build_tractseg_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_executable(tmp_path / "TractSeg", "#!/usr/bin/env bash\nexit 0\n")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    image = tmp_path / "dwi.nii.gz"
    image.write_text("image", encoding="utf-8")

    command = build_external_segmentation_command(
        ExternalSegmentationRequest(
            engine="tractseg",
            image_path=image,
            output_path=tmp_path / "bundles",
        )
    )

    assert command[0].endswith("TractSeg")
    assert "-i" in command and str(image) in command
    assert "-o" in command


def test_unsupported_engine_still_raises():
    with pytest.raises(ExternalSegmentationError):
        build_external_segmentation_command(
            ExternalSegmentationRequest(
                engine="not_a_real_engine",  # type: ignore[arg-type]
                image_path="x.nii.gz",
                output_path="y.nii.gz",
            )
        )
