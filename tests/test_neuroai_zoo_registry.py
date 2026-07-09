from __future__ import annotations

import pytest

from qortex.neuroai.contracts import EvidenceStatus
from qortex.neuroai.models.zoo.registry import (
    lookup,
    list_entries,
    register,
)
from qortex.neuroai.models.zoo.schema import (
    ExecutionMode,
    LicenseInfo,
    ZooEntry,
    ZooEntryType,
)


def _entry(entry_id: str, entry_type=ZooEntryType.model, provider="braindecode") -> ZooEntry:
    return ZooEntry(
        id=entry_id,
        display_name=entry_id,
        entry_type=entry_type,
        provider=provider,
        execution_mode=ExecutionMode.in_process,
        source_url="https://example.org/model",
        modality=["eeg"],
        task=["classification"],
        license=LicenseInfo(evidence_status=EvidenceStatus.unknown),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="architecture_available",
        priority="P0",
    )


def test_register_and_lookup():
    register(_entry("braindecode.ATCNet"))
    found = lookup("braindecode.ATCNet")
    assert found is not None
    assert found.id == "braindecode.ATCNet"


def test_lookup_missing_returns_none():
    assert lookup("nonexistent.model") is None


def test_register_duplicate_id_raises():
    register(_entry("braindecode.ATCNet"))
    with pytest.raises(ValueError):
        register(_entry("braindecode.ATCNet"))


def test_list_entries_filters_by_provider_and_sorts():
    # "braindecode.EEGNet" is a zoo seed entry (pre-registered by the
    # autouse conftest fixture) with provider="braindecode", so it is
    # expected in this filter's results alongside the entries below.
    register(_entry("braindecode.Deep4Net", provider="braindecode"))
    register(_entry("monai.vista3d", provider="monai_bundle"))
    register(_entry("braindecode.ATCNet", provider="braindecode"))

    bd = list_entries(provider="braindecode")
    assert [e.id for e in bd] == [
        "braindecode.ATCNet",
        "braindecode.Deep4Net",
        "braindecode.EEGNet",
    ]


def test_list_entries_filters_by_entry_type():
    # "external.totalsegmentator" is a zoo seed entry_type=external_engine
    # entry (pre-registered by the autouse conftest fixture), so it is
    # expected in this filter's results alongside the entry below.
    register(_entry("external.freesurfer", entry_type=ZooEntryType.external_engine, provider="external_cli"))
    register(_entry("braindecode.ATCNet"))

    engines = list_entries(entry_type=ZooEntryType.external_engine)
    assert [e.id for e in engines] == ["external.freesurfer", "external.totalsegmentator"]
