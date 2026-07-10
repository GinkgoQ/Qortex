"""Candidate construction for the Qortex NeuroAI compiler."""

from __future__ import annotations

import shutil
from typing import Any

from qortex.neuroai.compiler.repair import RepairOption
from qortex.neuroai.compiler.resources import estimate_resource_plan
from qortex.neuroai.compiler.result import (
    CapabilityState,
    CompatibilityProof,
    GeometryPlan,
    LicenseReport,
    ModelCandidate,
    SecurityReport,
    SourceProfileSummary,
)
from qortex.neuroai.contracts import ArtifactContract, CompatibilityStatus, EvidenceStatus, PreprocessPlan
from qortex.neuroai.models.license import LicenseStatus, evaluate_license
from qortex.neuroai.models.zoo.schema import ExecutionMode, ZooEntry
from qortex.neuroai.models.zoo.status import RuntimeStatus, is_runtime_executable, runtime_status


def build_candidates(
    *,
    entries: list[ZooEntry],
    source_profile: SourceProfileSummary,
    task: str,
    device: str,
    max_vram_gb: float | None,
    accept_unknown_license_risk: bool,
    allow_remote_code: bool,
    require_open_license: bool,
) -> list[ModelCandidate]:
    candidates: list[ModelCandidate] = []
    for entry in entries:
        if task not in entry.task:
            continue
        candidates.append(_candidate(
            entry=entry,
            source_profile=source_profile,
            device=device,
            max_vram_gb=max_vram_gb,
            accept_unknown_license_risk=accept_unknown_license_risk,
            allow_remote_code=allow_remote_code,
            require_open_license=require_open_license,
        ))
    return sorted(candidates, key=lambda candidate: (-candidate.fit_score, candidate.id))


def _candidate(
    *,
    entry: ZooEntry,
    source_profile: SourceProfileSummary,
    device: str,
    max_vram_gb: float | None,
    accept_unknown_license_risk: bool,
    allow_remote_code: bool,
    require_open_license: bool,
) -> ModelCandidate:
    blockers: list[str] = []
    warnings: list[str] = []
    repairs: list[RepairOption] = []
    evidence_ids = [
        f"zoo:{entry.id}",
        f"license:{entry.id}",
        f"security:{entry.id}",
        f"runtime:{entry.id}",
    ]

    license_report = _license_report(
        entry,
        accept_unknown_license_risk=accept_unknown_license_risk,
        require_open_license=require_open_license,
        repairs=repairs,
    )
    security_report = _security_report(
        entry,
        allow_remote_code=allow_remote_code,
        repairs=repairs,
    )
    compatibility = _compatibility(entry, source_profile)
    resource_plan = estimate_resource_plan(
        device=device,
        input_contract=entry.input_contract,
        source_size_bytes=source_profile.size_bytes,
        max_vram_gb=max_vram_gb,
    )
    geometry_plan = _geometry_plan(entry, source_profile)
    preprocess_plan = PreprocessPlan(
        unknowns=[
            "Model preprocessing contract is unconfirmed."
        ] if getattr(entry.input_contract, "evidence_status", None) == EvidenceStatus.unknown else []
    )

    runtime = runtime_status(entry)
    runtime_blocker = _runtime_blocker(entry, runtime)
    if runtime_blocker is not None:
        blockers.append(runtime_blocker)
        repairs.extend(_runtime_repairs(entry, runtime))

    blockers.extend(license_report.blockers)
    blockers.extend(security_report.blockers)
    blockers.extend(compatibility.blockers)
    blockers.extend(resource_plan.blockers)
    blockers.extend(geometry_plan.blockers)
    warnings.extend(license_report.warnings)
    warnings.extend(security_report.warnings)
    warnings.extend(compatibility.warnings)
    warnings.extend(resource_plan.notes)
    warnings.extend(geometry_plan.notes)

    capability_state = _capability_state(
        entry=entry,
        runtime=runtime,
        blockers=blockers,
        security_report=security_report,
    )
    runnable = (
        capability_state == CapabilityState.executable
        and compatibility.status == CompatibilityStatus.compatible.value
        and not blockers
    )

    candidate = ModelCandidate(
        id=entry.id,
        display_name=entry.display_name,
        provider=entry.provider,
        execution_mode=entry.execution_mode.value,
        entry_type=entry.entry_type.value,
        tasks=list(entry.task),
        modalities=list(entry.modality),
        runtime_status=runtime.value,
        capability_state=capability_state,
        runnable=runnable,
        compatibility=compatibility,
        preprocess_plan=preprocess_plan,
        geometry_plan=geometry_plan,
        resource_plan=resource_plan,
        license_report=license_report,
        security_report=security_report,
        artifact_contract=ArtifactContract(
            qortex_version="unknown",
            created_at="not_created_by_compile",
            source_id=source_profile.source,
            model_id=entry.id,
            runtime_backend=entry.execution_mode.value,
            device=device,
            output_type=getattr(entry.output_contract, "output_type", None),
            compatibility_status=compatibility.status,
            warnings=[{"message": warning} for warning in warnings],
            unknowns=list(preprocess_plan.unknowns),
        ),
        repair_options=repairs,
        blockers=blockers,
        warnings=warnings,
        evidence_ids=evidence_ids,
    )
    fit_score, fit_reasons = _fit_score(candidate)
    return candidate.model_copy(update={"fit_score": fit_score, "fit_reasons": fit_reasons})


def _license_report(
    entry: ZooEntry,
    *,
    accept_unknown_license_risk: bool,
    require_open_license: bool,
    repairs: list[RepairOption],
) -> LicenseReport:
    status = evaluate_license(entry.license)
    blockers: list[str] = []
    warnings: list[str] = []
    if status == LicenseStatus.blocked:
        blockers.append("License evidence is blocked; execution is not allowed.")
    elif status == LicenseStatus.unknown and not accept_unknown_license_risk:
        blockers.append("License evidence is unknown and explicit risk acceptance was not provided.")
        repairs.append(RepairOption(
            code="accept_unknown_license_risk",
            severity="blocking",
            title="Explicitly accept unknown license risk",
            detail="Use only after manual legal review of the model license.",
            command=["qortex", "compile", "<source>", "--task", "<task>", "--accept-unknown-license-risk"],
            affects=[entry.id],
        ))
    elif status == LicenseStatus.unknown:
        warnings.append("License evidence remains unknown; risk was explicitly accepted.")

    if require_open_license and status in {
        LicenseStatus.non_commercial_only,
        LicenseStatus.registration_required,
        LicenseStatus.research_only,
    }:
        blockers.append(f"License status {status.value!r} is not open-use compatible.")

    return LicenseReport(
        status=status.value,
        evidence_status=entry.license.evidence_status,
        name=entry.license.name,
        url=entry.license.url,
        blockers=blockers,
        warnings=warnings,
    )


def _security_report(
    entry: ZooEntry,
    *,
    allow_remote_code: bool,
    repairs: list[RepairOption],
) -> SecurityReport:
    blockers: list[str] = []
    warnings: list[str] = []
    if entry.security.trust_remote_code_required and not (allow_remote_code or entry.security.allow_remote_code):
        blockers.append("Remote Python code is required but not allowed.")
        repairs.append(RepairOption(
            code="allow_remote_code",
            severity="blocking",
            title="Allow remote code in a trusted sandbox",
            detail="Only enable this inside a reviewed, isolated runtime environment.",
            command=["qortex", "compile", "<source>", "--task", "<task>", "--allow-remote-code"],
            affects=[entry.id],
        ))

    resolved: str | None = None
    executable = getattr(entry.external_engine_contract, "executable", None)
    if entry.execution_mode == ExecutionMode.external_cli and executable:
        resolved = shutil.which(executable)
        if resolved is None:
            blockers.append(f"Required executable {executable!r} was not found on PATH.")
            repairs.append(RepairOption(
                code="install_external_executable",
                severity="blocking",
                title=f"Install {executable}",
                detail="Install the external engine and ensure the declared executable is on PATH.",
                affects=[entry.id],
            ))

    return SecurityReport(
        remote_code_required=entry.security.trust_remote_code_required,
        remote_code_allowed=allow_remote_code or entry.security.allow_remote_code,
        sandbox_required=entry.security.requires_sandbox,
        executable_names=list(entry.security.executable_names),
        resolved_executable=resolved,
        blockers=blockers,
        warnings=warnings,
    )


def _compatibility(entry: ZooEntry, source_profile: SourceProfileSummary) -> CompatibilityProof:
    evidence: list[dict[str, Any]] = []
    blockers: list[str] = []
    warnings: list[str] = []
    source_modality = source_profile.modality
    model_modalities = list(entry.modality)
    evidence.append({
        "check": "modality",
        "source_modality": source_modality,
        "model_modalities": model_modalities,
    })
    if source_profile.spatial_shape is not None or source_profile.n_channels is not None:
        evidence.append({
            "check": "header_geometry",
            "spatial_shape": source_profile.spatial_shape,
            "voxel_sizes_mm": source_profile.voxel_sizes_mm,
            "orientation": source_profile.orientation,
            "n_channels": source_profile.n_channels,
            "sampling_rate_hz": source_profile.sampling_rate_hz,
            "duration_s": source_profile.duration_s,
        })

    if source_modality is None:
        warnings.append("Source modality is unknown; compatibility cannot be proven offline.")
        status = "uncertain"
    elif source_modality not in model_modalities:
        blockers.append(
            f"Source modality {source_modality!r} is not declared by model modalities {model_modalities!r}."
        )
        status = "incompatible"
    else:
        status = "compatible"

    # Deep evidence-vs-contract evaluation: the header/signal facts collected
    # from the real source (channel count, sampling rate, spatial voxel
    # spacing, orientation) are compared against the model's InputContract
    # requirements and turned into a real verdict plus SPECIFIC required
    # transforms -- not left as an unused evidence dict. A hard mismatch that
    # no transform can bridge (e.g. the source has fewer channels than the
    # model requires) is a blocker; a soft mismatch a known transform bridges
    # (resample, reorient, channel-select, resample_spatial) downgrades a
    # "compatible" verdict to "compatible_with_transforms" and records exactly
    # what preprocessing would be needed.
    required_transforms: list[dict[str, Any]] = []
    ic = entry.input_contract
    if status == "compatible" and ic is not None:
        # Channel count
        model_ch = getattr(ic, "n_channels", None)
        src_ch = source_profile.n_channels
        if model_ch is not None and src_ch is not None and src_ch != model_ch:
            if src_ch < model_ch:
                blockers.append(
                    f"Source has {src_ch} channel(s) but model requires {model_ch}; "
                    "missing channels cannot be synthesized."
                )
                status = "incompatible"
            else:
                required_transforms.append({
                    "transform": "select_channels",
                    "reason": f"source {src_ch} channels > model {model_ch} required",
                    "from": src_ch,
                    "to": model_ch,
                })
        # Sampling rate
        model_sr = getattr(ic, "sampling_rate_hz", None)
        src_sr = source_profile.sampling_rate_hz
        if model_sr is not None and src_sr is not None and abs(src_sr - model_sr) > 1e-6:
            required_transforms.append({
                "transform": "resample",
                "reason": f"source {src_sr} Hz != model {model_sr} Hz",
                "from": src_sr,
                "to": model_sr,
            })
        # Orientation (only when the model axis convention is a 3-letter
        # anatomical orientation code like RAS/LAS/LPS -- not channels_first etc.)
        model_axis = str(getattr(ic, "axis_convention", "") or "")
        model_axis = model_axis.split(".")[-1]  # AxisConvention enum -> bare value
        src_orient = source_profile.orientation
        if (
            src_orient is not None
            and len(model_axis) == 3
            and model_axis.isalpha()
            and model_axis.upper() != src_orient.upper()
        ):
            required_transforms.append({
                "transform": "reorient",
                "reason": f"source {src_orient} != model {model_axis}",
                "from": src_orient,
                "to": model_axis,
            })
        # Voxel spacing
        model_vox = getattr(ic, "voxel_sizes_mm", None)
        src_vox = source_profile.voxel_sizes_mm
        if (
            model_vox is not None
            and src_vox is not None
            and tuple(round(v, 4) for v in src_vox) != tuple(round(v, 4) for v in model_vox)
        ):
            required_transforms.append({
                "transform": "resample_spatial",
                "reason": f"source voxel {tuple(src_vox)} != model {tuple(model_vox)}",
                "from": list(src_vox),
                "to": list(model_vox),
            })

        if status == "compatible" and required_transforms:
            status = "compatible_with_transforms"
            warnings.append(
                "Source is compatible after applying required transforms: "
                + ", ".join(t["transform"] for t in required_transforms)
                + "."
            )

    if entry.input_contract is not None and entry.input_contract.evidence_status == EvidenceStatus.unknown:
        warnings.append("Input contract evidence is unknown; compatibility cannot be fully proven offline.")
        if status in ("compatible", "compatible_with_transforms"):
            status = "uncertain"

    if entry.input_contract is None and entry.external_engine_contract is None:
        warnings.append("Entry has no input or external engine contract; compatibility is uncertain.")
        status = "uncertain" if status == "compatible" else status

    return CompatibilityProof(
        status=status,
        blockers=blockers,
        warnings=warnings,
        evidence=evidence,
        required_transforms=required_transforms,
    )


def _geometry_plan(entry: ZooEntry, source_profile: SourceProfileSummary) -> GeometryPlan:
    input_axis = getattr(entry.input_contract, "axis_convention", None)
    output_axis = getattr(entry.output_contract, "axis_convention", None)
    notes: list[str] = []
    blockers: list[str] = []
    if source_profile.spatial_shape is not None:
        notes.append(
            f"Source spatial_shape={source_profile.spatial_shape}, "
            f"voxel_sizes_mm={source_profile.voxel_sizes_mm}, "
            f"orientation={source_profile.orientation} (confirmed from NIfTI header)."
        )
    if source_profile.n_channels is not None:
        notes.append(
            f"Source n_channels={source_profile.n_channels}, "
            f"sampling_rate_hz={source_profile.sampling_rate_hz}, "
            f"duration_s={source_profile.duration_s} (confirmed from EEG header)."
        )
    if entry.output_contract is not None and output_axis is None:
        notes.append("Output axis convention is not confirmed; downstream artifact must preserve source geometry lineage.")
    if entry.external_engine_contract is not None and entry.external_engine_contract.geometry_preservation_known is False:
        blockers.append("External engine declares geometry preservation as false.")
    if entry.external_engine_contract is not None and entry.external_engine_contract.geometry_preservation_known is None:
        notes.append("External engine geometry preservation is unknown; output validation must compare source/output geometry.")
    return GeometryPlan(
        source_coordinate_frame=source_profile.orientation,
        model_axis_convention=getattr(input_axis, "value", input_axis),
        output_axis_convention=getattr(output_axis, "value", output_axis),
        blockers=blockers,
        notes=notes,
    )


def _runtime_blocker(entry: ZooEntry, runtime: RuntimeStatus) -> str | None:
    if runtime == RuntimeStatus.blocked:
        return "Runtime status is blocked."
    if runtime == RuntimeStatus.checkpoint_unresolved:
        return "Checkpoint or executable workflow is unresolved; runtime execution is not truthful yet."
    if runtime == RuntimeStatus.architecture_available:
        return "Only architecture metadata is available; no verified executable weights are declared."
    if runtime == RuntimeStatus.unknown:
        return "Runtime status is unknown."
    if not is_runtime_executable(entry):
        return f"Runtime status {runtime.value!r} is not executable."
    return None


def _runtime_repairs(entry: ZooEntry, runtime: RuntimeStatus) -> list[RepairOption]:
    if runtime == RuntimeStatus.checkpoint_unresolved:
        return [RepairOption(
            code="resolve_checkpoint_contract",
            severity="blocking",
            title="Resolve checkpoint and execution contract",
            detail="Register a verified checkpoint, preprocessing contract, geometry restoration path, and end-to-end fixture before execution.",
            affects=[entry.id],
        )]
    if runtime == RuntimeStatus.architecture_available:
        return [RepairOption(
            code="register_verified_weights",
            severity="blocking",
            title="Register verified weights",
            detail="Architecture-only entries need confirmed weights and runtime integration before execution.",
            affects=[entry.id],
        )]
    return []


def _capability_state(
    *,
    entry: ZooEntry,
    runtime: RuntimeStatus,
    blockers: list[str],
    security_report: SecurityReport,
) -> CapabilityState:
    if blockers:
        if runtime in {RuntimeStatus.checkpoint_unresolved, RuntimeStatus.architecture_available, RuntimeStatus.unknown}:
            return CapabilityState.unavailable
        if security_report.resolved_executable is None and entry.execution_mode == ExecutionMode.external_cli:
            return CapabilityState.requires_local_executable
        return CapabilityState.blocked
    if runtime == RuntimeStatus.runnable_if_executable_available and entry.execution_mode == ExecutionMode.external_cli:
        return CapabilityState.executable if security_report.resolved_executable else CapabilityState.requires_local_executable
    if is_runtime_executable(entry):
        return CapabilityState.executable
    return CapabilityState.plan_only


_CAPABILITY_BASE_SCORE = {
    CapabilityState.executable: 70,
    CapabilityState.requires_local_executable: 55,
    CapabilityState.plan_only: 35,
    CapabilityState.unavailable: 10,
    CapabilityState.blocked: 0,
}

_COMPAT_ADJUSTMENT = {
    "compatible": 20,
    # A source that is compatible once specific, known transforms (resample,
    # reorient, channel-select) are applied is worth less than a native fit
    # but far more than an unprovable "uncertain" -- it is a real, actionable
    # match with a concrete preprocessing cost.
    "compatible_with_transforms": 10,
    "uncertain": 0,
    "incompatible": -40,
}


def _fit_score(candidate: ModelCandidate) -> tuple[float, list[str]]:
    reasons: list[str] = []
    base = _CAPABILITY_BASE_SCORE[candidate.capability_state]
    reasons.append(f"base tier for capability_state={candidate.capability_state.value}: {base}")
    score = float(base)

    compat_adj = _COMPAT_ADJUSTMENT.get(candidate.compatibility.status, 0)
    if compat_adj:
        reasons.append(f"compatibility status {candidate.compatibility.status!r} adjustment: {compat_adj:+d}")
    score += compat_adj

    if candidate.blockers:
        penalty = -8 * len(candidate.blockers)
        reasons.append(f"blocker penalty for {len(candidate.blockers)} blocker(s): {penalty}")
        score += penalty

    if (
        candidate.geometry_plan.source_coordinate_frame is not None
        and candidate.geometry_plan.model_axis_convention is not None
    ):
        reasons.append("geometry bonus: source_coordinate_frame and model_axis_convention both known: +5")
        score += 5

    score = max(0.0, min(100.0, score))
    return score, reasons


def _candidate_sort_key(candidate: ModelCandidate) -> tuple[int, int, str]:
    state_rank = {
        CapabilityState.executable: 0,
        CapabilityState.requires_local_executable: 1,
        CapabilityState.plan_only: 2,
        CapabilityState.unavailable: 3,
        CapabilityState.blocked: 4,
    }[candidate.capability_state]
    blocker_rank = len(candidate.blockers)
    return (state_rank, blocker_rank, candidate.id)


__all__ = ["build_candidates"]
