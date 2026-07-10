"""Source acquisition planning for NeuroAI compilation."""

from __future__ import annotations

from qortex.neuroai.contracts import BaseModel, EvidenceStatus, Field


class AcquisitionPlan(BaseModel):
    source: str
    source_type: str
    required_download: bool
    estimated_download_gb: float | None = None
    evidence_status: EvidenceStatus = EvidenceStatus.unknown
    blockers: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def build_acquisition_plan(
    *,
    source: str,
    source_type: str,
    local_size_bytes: int | None,
    max_download_gb: float | None,
) -> AcquisitionPlan:
    if source_type.startswith("local_"):
        return AcquisitionPlan(
            source=source,
            source_type=source_type,
            required_download=False,
            estimated_download_gb=0.0,
            evidence_status=EvidenceStatus.confirmed,
        )

    plan = AcquisitionPlan(
        source=source,
        source_type=source_type,
        required_download=True,
        estimated_download_gb=(local_size_bytes / 1e9) if local_size_bytes is not None else None,
        evidence_status=EvidenceStatus.unknown,
        notes=["Remote source size is not known without manifest inspection; no download is performed by compile."],
    )
    if (
        max_download_gb is not None
        and plan.estimated_download_gb is not None
        and plan.estimated_download_gb > max_download_gb
    ):
        plan.blockers.append(
            f"Estimated download {plan.estimated_download_gb:.3f} GB exceeds limit {max_download_gb:.3f} GB."
        )
    return plan


__all__ = ["AcquisitionPlan", "build_acquisition_plan"]
