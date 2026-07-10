"""Result schema for the Qortex NeuroAI compiler."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from qortex.neuroai.compiler.acquisition import AcquisitionPlan
from qortex.neuroai.compiler.evidence import EvidenceGraph
from qortex.neuroai.compiler.repair import RepairOption
from qortex.neuroai.compiler.resources import ResourcePlan
from qortex.neuroai.compiler.serialization import pretty_json, sha256_json, to_plain
from qortex.neuroai.contracts import ArtifactContract, BaseModel, EvidenceStatus, Field, PreprocessPlan


class CapabilityState(str, Enum):
    executable = "executable"
    requires_local_executable = "requires_local_executable"
    unavailable = "unavailable"
    blocked = "blocked"
    plan_only = "plan_only"


class SourceProfileSummary(BaseModel):
    source: str
    source_type: str
    exists: bool
    size_bytes: int | None = None
    sha256: str | None = None
    modality: str | None = None
    available_suffixes: list[str] = Field(default_factory=list)
    evidence_status: EvidenceStatus = EvidenceStatus.unknown
    notes: list[str] = Field(default_factory=list)
    spatial_shape: tuple[int, ...] | None = None
    voxel_sizes_mm: tuple[float, ...] | None = None
    orientation: str | None = None
    n_channels: int | None = None
    sampling_rate_hz: float | None = None
    duration_s: float | None = None


class CompatibilityProof(BaseModel):
    status: Literal["compatible", "incompatible", "uncertain"]
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)


class LicenseReport(BaseModel):
    status: str
    evidence_status: EvidenceStatus
    name: str | None = None
    url: str | None = None
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SecurityReport(BaseModel):
    remote_code_required: bool = False
    remote_code_allowed: bool = False
    sandbox_required: bool = False
    executable_names: list[str] = Field(default_factory=list)
    resolved_executable: str | None = None
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class GeometryPlan(BaseModel):
    source_coordinate_frame: str | None = None
    model_axis_convention: str | None = None
    output_axis_convention: str | None = None
    lineage_required: bool = True
    blockers: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ModelCandidate(BaseModel):
    id: str
    display_name: str
    provider: str
    execution_mode: str
    entry_type: str
    tasks: list[str] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=list)
    runtime_status: str
    capability_state: CapabilityState
    runnable: bool
    compatibility: CompatibilityProof
    preprocess_plan: PreprocessPlan
    geometry_plan: GeometryPlan
    resource_plan: ResourcePlan
    license_report: LicenseReport
    security_report: SecurityReport
    artifact_contract: ArtifactContract
    repair_options: list[RepairOption] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class CompilationResult(BaseModel):
    request: dict[str, Any]
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_profile: SourceProfileSummary
    evidence_graph: EvidenceGraph
    acquisition_plan: AcquisitionPlan
    candidates: list[ModelCandidate]
    runnable: bool
    plan_hash: str

    @classmethod
    def build(
        cls,
        *,
        request: Any,
        source_profile: SourceProfileSummary,
        evidence_graph: EvidenceGraph,
        acquisition_plan: AcquisitionPlan,
        candidates: list[ModelCandidate],
    ) -> "CompilationResult":
        payload = {
            "request": to_plain(request),
            "source_profile": to_plain(source_profile),
            "evidence_graph": to_plain(evidence_graph),
            "acquisition_plan": to_plain(acquisition_plan),
            "candidates": to_plain(candidates),
            "runnable": any(candidate.runnable for candidate in candidates) and not acquisition_plan.blockers,
        }
        return cls(
            **payload,
            plan_hash=sha256_json(payload),
        )

    def to_json(self) -> str:
        return pretty_json(self)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_json(), encoding="utf-8")
        return output


__all__ = [
    "CapabilityState",
    "CompatibilityProof",
    "CompilationResult",
    "GeometryPlan",
    "LicenseReport",
    "ModelCandidate",
    "SecurityReport",
    "SourceProfileSummary",
]
