from __future__ import annotations

from qortex.neuroai.compiler import CompilationRequest, compile_neuroai
from qortex.neuroai.compiler.acquisition import AcquisitionPlan
from qortex.neuroai.compiler.candidates import _fit_score
from qortex.neuroai.compiler.evidence import EvidenceGraph
from qortex.neuroai.compiler.result import (
    CapabilityState,
    CompatibilityProof,
    CompilationResult,
    GeometryPlan,
    LicenseReport,
    ModelCandidate,
    SecurityReport,
    SourceProfileSummary,
)
from qortex.neuroai.compiler.resources import ResourcePlan
from qortex.neuroai.contracts import ArtifactContract, PreprocessPlan


def _make_candidate(
    *,
    id: str,
    capability_state: CapabilityState,
    compatibility_status: str,
    blockers: list[str],
    geometry_known: bool,
    runnable: bool = False,
) -> ModelCandidate:
    return ModelCandidate(
        id=id,
        display_name=id,
        provider="test",
        execution_mode="in_process",
        entry_type="segmentation",
        runtime_status="runnable",
        capability_state=capability_state,
        runnable=runnable,
        compatibility=CompatibilityProof(status=compatibility_status),
        preprocess_plan=PreprocessPlan(),
        geometry_plan=GeometryPlan(
            source_coordinate_frame="RAS" if geometry_known else None,
            model_axis_convention="RAS" if geometry_known else None,
        ),
        resource_plan=ResourcePlan(device="cpu"),
        license_report=LicenseReport(status="open", evidence_status="confirmed"),
        security_report=SecurityReport(),
        artifact_contract=ArtifactContract(qortex_version="unknown"),
        blockers=blockers,
    )


def test_fit_score_exact_formula_executable_compatible_with_geometry_and_blocker():
    candidate = _make_candidate(
        id="m1",
        capability_state=CapabilityState.executable,
        compatibility_status="compatible",
        blockers=["one blocker"],
        geometry_known=True,
    )
    score, reasons = _fit_score(candidate)
    # base 70 + compat +20 - 8*1 blocker + geometry +5 = 87
    assert score == 87.0
    assert any("base tier for capability_state=executable: 70" in r for r in reasons)
    assert any("+20" in r for r in reasons)
    assert any("blocker penalty" in r for r in reasons)
    assert any("geometry bonus" in r for r in reasons)


def test_fit_score_clamped_to_zero_when_heavily_penalized():
    candidate = _make_candidate(
        id="m2",
        capability_state=CapabilityState.blocked,
        compatibility_status="incompatible",
        blockers=["a", "b", "c"],
        geometry_known=False,
    )
    score, _ = _fit_score(candidate)
    # base 0 - 40 - 8*3 = -64 -> clamped to 0
    assert score == 0.0


def test_selected_model_is_highest_scoring_runnable_candidate():
    low = _make_candidate(
        id="z_low",
        capability_state=CapabilityState.plan_only,
        compatibility_status="uncertain",
        blockers=[],
        geometry_known=False,
        runnable=True,
    )
    high = _make_candidate(
        id="a_high",
        capability_state=CapabilityState.executable,
        compatibility_status="compatible",
        blockers=[],
        geometry_known=True,
        runnable=True,
    )
    low_score, _ = _fit_score(low)
    high_score, _ = _fit_score(high)
    low = low.model_copy(update={"fit_score": low_score})
    high = high.model_copy(update={"fit_score": high_score})

    result = CompilationResult.build(
        request={},
        source_profile=SourceProfileSummary(source="x", source_type="local_file", exists=True),
        evidence_graph=EvidenceGraph(),
        acquisition_plan=AcquisitionPlan(source="x", source_type="local_file", required_download=False),
        candidates=[low, high],
    )

    assert result.selected_model == "a_high"


def test_compile_neuroai_selected_model_is_none_when_nothing_runnable(tmp_path):
    source = tmp_path / "image.nii.gz"
    source.write_bytes(b"source")

    result = compile_neuroai(CompilationRequest(source=str(source), task="whole_brain_segmentation"))

    assert result.selected_model is None
