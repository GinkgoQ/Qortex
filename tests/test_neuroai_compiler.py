from __future__ import annotations

import json

from typer.testing import CliRunner

from qortex.cli.app import app
from qortex.neuroai.compiler import CapabilityState, CompilationRequest, compile_neuroai, profile_source

runner = CliRunner()


def test_profile_source_hashes_local_file_and_infers_nifti_modality(tmp_path):
    source = tmp_path / "sub-01_T1w.nii.gz"
    source.write_bytes(b"qortex-neuroai-compiler")

    profile = profile_source(str(source))

    assert profile.source_type == "local_file"
    assert profile.exists is True
    assert profile.size_bytes == len(b"qortex-neuroai-compiler")
    assert profile.sha256 is not None
    assert profile.modality == "mri"
    assert profile.available_suffixes == ["nii.gz"]


def test_compile_marks_checkpoint_unresolved_promptable_models_unavailable(tmp_path):
    source = tmp_path / "image.nii.gz"
    source.write_bytes(b"not-a-real-volume-but-a-real-local-source")

    result = compile_neuroai(CompilationRequest(
        source=str(source),
        task="foundation_segmentation",
        accept_unknown_license_risk=True,
    ))

    vista = next(candidate for candidate in result.candidates if candidate.id == "monai.vista3d")
    assert vista.runtime_status == "checkpoint_unresolved"
    assert vista.capability_state == CapabilityState.unavailable
    assert vista.runnable is False
    assert any("unresolved" in blocker for blocker in vista.blockers)


def test_compile_blocks_unknown_license_by_default(tmp_path):
    source = tmp_path / "image.nii.gz"
    source.write_bytes(b"source")

    result = compile_neuroai(CompilationRequest(source=str(source), task="whole_brain_segmentation"))

    candidate = next(candidate for candidate in result.candidates if candidate.id == "monai.wholeBrainSeg_Large_UNEST_segmentation")
    assert candidate.runnable is False
    assert candidate.license_report.status == "unknown"
    assert any("License evidence is unknown" in blocker for blocker in candidate.blockers)
    assert any(option.code == "accept_unknown_license_risk" for option in candidate.repair_options)


def test_compile_external_engine_records_missing_executable_requirement(tmp_path):
    source = tmp_path / "image.nii.gz"
    source.write_bytes(b"source")

    result = compile_neuroai(CompilationRequest(
        source=str(source),
        task="whole_brain_segmentation",
        accept_unknown_license_risk=True,
    ))

    candidate = next(candidate for candidate in result.candidates if candidate.id == "external.synthseg")
    assert candidate.execution_mode == "external_cli"
    if candidate.security_report.resolved_executable is None:
        assert candidate.capability_state == CapabilityState.requires_local_executable
        assert any("Required executable" in blocker for blocker in candidate.blockers)
        assert any(option.code == "install_external_executable" for option in candidate.repair_options)
    else:
        assert candidate.capability_state in {CapabilityState.executable, CapabilityState.blocked}


def test_compile_plan_hash_is_stable_and_saved_json_contains_required_sections(tmp_path):
    source = tmp_path / "image.nii.gz"
    source.write_bytes(b"source")
    request = CompilationRequest(
        source=str(source),
        task="segmentation",
        accept_unknown_license_risk=True,
    )

    first = compile_neuroai(request)
    second = compile_neuroai(request)
    output = tmp_path / "execution-plan.json"
    first.save(output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert first.plan_hash == second.plan_hash
    assert payload["plan_hash"] == first.plan_hash
    assert "source_profile" in payload
    assert "evidence_graph" in payload
    assert "acquisition_plan" in payload
    assert "candidates" in payload
    assert {"license_report", "security_report", "artifact_contract", "resource_plan"} <= set(payload["candidates"][0])


def test_compile_cli_writes_execution_plan(tmp_path):
    source = tmp_path / "image.nii.gz"
    source.write_bytes(b"source")
    output = tmp_path / "plan.json"

    result = runner.invoke(app, [
        "compile",
        str(source),
        "--task",
        "segmentation",
        "--accept-unknown-license-risk",
        "--output",
        str(output),
    ])

    assert result.exit_code == 0
    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["request"]["task"] == "segmentation"
    assert "plan_hash=" in result.stdout
