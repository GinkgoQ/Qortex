from __future__ import annotations

import pytest

from qortex.core.exceptions import ModelAdapterError
from qortex.neuroai.contracts import EvidenceStatus
from qortex.neuroai.models.license import LicenseStatus, check_license_gate, evaluate_license
from qortex.neuroai.models.zoo.schema import ExecutionMode, LicenseInfo, ZooEntry, ZooEntryType


def _entry(license_info: LicenseInfo) -> ZooEntry:
    return ZooEntry(
        id="test.model",
        display_name="Test Model",
        entry_type=ZooEntryType.model,
        provider="braindecode",
        execution_mode=ExecutionMode.in_process,
        source_url="https://example.org/model",
        modality=["eeg"],
        task=["classification"],
        license=license_info,
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="architecture_available",
        priority="P0",
    )


def test_unknown_evidence_status_maps_to_unknown():
    assert evaluate_license(LicenseInfo(evidence_status=EvidenceStatus.unknown)) == LicenseStatus.unknown


def test_blocked_evidence_status_maps_to_blocked():
    assert evaluate_license(LicenseInfo(evidence_status=EvidenceStatus.blocked)) == LicenseStatus.blocked


def test_confirmed_with_no_restrictions_maps_to_safe_for_open_use():
    license_info = LicenseInfo(evidence_status=EvidenceStatus.confirmed, name="MIT")
    assert evaluate_license(license_info) == LicenseStatus.safe_for_open_use


def test_commercial_use_false_maps_to_non_commercial_only():
    license_info = LicenseInfo(evidence_status=EvidenceStatus.confirmed, commercial_use=False)
    assert evaluate_license(license_info) == LicenseStatus.non_commercial_only


def test_requires_registration_maps_to_registration_required():
    license_info = LicenseInfo(evidence_status=EvidenceStatus.confirmed, requires_registration=True)
    assert evaluate_license(license_info) == LicenseStatus.registration_required


def test_gate_blocks_unknown_license_by_default():
    entry = _entry(LicenseInfo(evidence_status=EvidenceStatus.unknown))

    with pytest.raises(ModelAdapterError, match="license"):
        check_license_gate(entry)


def test_gate_allows_unknown_license_with_explicit_opt_in():
    entry = _entry(LicenseInfo(evidence_status=EvidenceStatus.unknown))

    check_license_gate(entry, accept_unknown_license_risk=True)


def test_gate_blocks_blocked_license_even_with_opt_in():
    entry = _entry(LicenseInfo(evidence_status=EvidenceStatus.blocked))

    with pytest.raises(ModelAdapterError):
        check_license_gate(entry, accept_unknown_license_risk=True)


def test_gate_allows_safe_license_with_no_flags():
    entry = _entry(LicenseInfo(evidence_status=EvidenceStatus.confirmed, name="MIT"))

    check_license_gate(entry)
