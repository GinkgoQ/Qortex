"""License gate for NeuroAI model zoo entries."""

from __future__ import annotations

from enum import Enum

from qortex.core.exceptions import ModelAdapterError
from qortex.neuroai.contracts import EvidenceStatus
from qortex.neuroai.models.zoo.schema import LicenseInfo, ZooEntry


class LicenseStatus(str, Enum):
    safe_for_open_use = "safe_for_open_use"
    research_only = "research_only"
    non_commercial_only = "non_commercial_only"
    registration_required = "registration_required"
    unknown = "unknown"
    blocked = "blocked"


def evaluate_license(license_info: LicenseInfo) -> LicenseStatus:
    """Classify existing license metadata without inventing missing facts.

    ``research_only`` is reserved for a future explicit schema field. Qortex
    currently records commercial-use and registration restrictions, but not a
    distinct confirmed research-only flag.
    """

    if license_info.commercial_use is False:
        return LicenseStatus.non_commercial_only
    if license_info.requires_registration:
        return LicenseStatus.registration_required
    if license_info.evidence_status == EvidenceStatus.blocked:
        return LicenseStatus.blocked
    if license_info.evidence_status == EvidenceStatus.unknown:
        return LicenseStatus.unknown
    return LicenseStatus.safe_for_open_use


def check_license_gate(
    entry: ZooEntry,
    *,
    accept_unknown_license_risk: bool = False,
) -> None:
    """Block unsafe or unverified license states before model execution."""

    status = evaluate_license(entry.license)
    if status == LicenseStatus.blocked:
        raise ModelAdapterError(
            f"{entry.id}'s license is confirmed blocked for this use; execution is not allowed."
        )
    if status == LicenseStatus.unknown and not accept_unknown_license_risk:
        raise ModelAdapterError(
            f"{entry.id}'s license has not been verified "
            "(evidence_status=unknown). Pass --accept-unknown-license-risk "
            "to proceed explicitly."
        )


__all__ = ["LicenseStatus", "evaluate_license", "check_license_gate"]
