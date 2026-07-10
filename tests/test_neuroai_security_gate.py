from __future__ import annotations

import pytest

from qortex.core.exceptions import ModelAdapterError
from qortex.neuroai.contracts import EvidenceStatus
from qortex.neuroai.models.security import check_executable_allowlist, check_remote_code_gate
from qortex.neuroai.models.zoo.schema import (
    ExecutionMode,
    LicenseInfo,
    SecurityPolicy,
    ZooEntry,
    ZooEntryType,
)


def _entry(security: SecurityPolicy) -> ZooEntry:
    return ZooEntry(
        id="test.model",
        display_name="Test Model",
        entry_type=ZooEntryType.model,
        provider="plugin",
        execution_mode=ExecutionMode.in_process,
        source_url="https://example.org/model",
        modality=["eeg"],
        task=["classification"],
        license=LicenseInfo(evidence_status=EvidenceStatus.unknown),
        security=security,
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="architecture_available",
        priority="P0",
    )


def test_remote_code_gate_blocks_by_default_when_required():
    entry = _entry(SecurityPolicy(trust_remote_code_required=True))

    with pytest.raises(ModelAdapterError, match="remote"):
        check_remote_code_gate(entry)


def test_remote_code_gate_allows_with_caller_opt_in():
    entry = _entry(SecurityPolicy(trust_remote_code_required=True))

    check_remote_code_gate(entry, allow_remote_code=True)


def test_remote_code_gate_allows_when_entry_declares_it_allowed():
    entry = _entry(SecurityPolicy(trust_remote_code_required=True, allow_remote_code=True))

    check_remote_code_gate(entry)


def test_remote_code_gate_noop_when_not_required():
    entry = _entry(SecurityPolicy())

    check_remote_code_gate(entry)


def test_executable_allowlist_blocks_mismatched_path():
    entry = _entry(SecurityPolicy(executable_names=["TotalSegmentator"]))

    with pytest.raises(ModelAdapterError, match="executable"):
        check_executable_allowlist(entry, "/usr/local/bin/some_other_tool")


def test_executable_allowlist_allows_matching_basename():
    entry = _entry(SecurityPolicy(executable_names=["TotalSegmentator"]))

    check_executable_allowlist(entry, "/usr/local/bin/TotalSegmentator")


def test_executable_allowlist_noop_when_not_declared():
    entry = _entry(SecurityPolicy())

    check_executable_allowlist(entry, "/usr/local/bin/anything")
