from __future__ import annotations

import pytest

from qortex.neuroai.contracts import EvidenceStatus
from qortex.neuroai.models.zoo.registry import clear_registry, register
from qortex.neuroai.models.zoo.schema import (
    ExecutionMode,
    ExternalEngineContract,
    InteractionContract,
    LicenseInfo,
    PromptType,
    ZooEntry,
    ZooEntryType,
)
from qortex.neuroai.models.zoo.validate import validate_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


def _base_kwargs(entry_id: str) -> dict:
    return dict(
        id=entry_id,
        display_name=entry_id,
        provider="braindecode",
        execution_mode=ExecutionMode.in_process,
        source_url="https://braindecode.org/stable/generated/braindecode.models.EEGNet.html",
        modality=["eeg"],
        task=["classification"],
        license=LicenseInfo(evidence_status=EvidenceStatus.unknown),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="architecture_available",
        priority="P0",
    )


def test_valid_registry_has_no_issues():
    register(ZooEntry(entry_type=ZooEntryType.model, **_base_kwargs("braindecode.EEGNet")))
    assert validate_registry() == []


def test_malformed_source_url_is_an_error():
    kwargs = _base_kwargs("braindecode.Bad")
    kwargs["source_url"] = "not-a-url"
    register(ZooEntry(entry_type=ZooEntryType.model, **kwargs))
    issues = validate_registry()
    assert any(i.entry_id == "braindecode.Bad" and i.severity == "error" for i in issues)


def test_promptable_entry_without_interaction_contract_is_an_error():
    register(ZooEntry(entry_type=ZooEntryType.promptable_model, **_base_kwargs("foundation.medsam")))
    issues = validate_registry()
    assert any(
        i.entry_id == "foundation.medsam" and "interaction_contract" in i.message
        for i in issues
    )


def test_promptable_entry_with_interaction_contract_passes():
    kwargs = _base_kwargs("foundation.medsam")
    kwargs["interaction_contract"] = InteractionContract(
        supported_prompt_types=[PromptType.point, PromptType.box]
    )
    register(ZooEntry(entry_type=ZooEntryType.promptable_model, **kwargs))
    assert validate_registry() == []


def test_external_engine_without_contract_is_an_error():
    kwargs = _base_kwargs("external.badengine")
    kwargs["provider"] = "external_cli"
    kwargs["execution_mode"] = ExecutionMode.external_cli
    register(ZooEntry(entry_type=ZooEntryType.external_engine, **kwargs))
    issues = validate_registry()
    assert any(
        i.entry_id == "external.badengine" and "external_engine_contract" in i.message
        for i in issues
    )


def test_unknown_provider_string_is_an_error():
    kwargs = _base_kwargs("bogus.model")
    kwargs["provider"] = "not_a_real_provider"
    register(ZooEntry(entry_type=ZooEntryType.model, **kwargs))
    issues = validate_registry()
    assert any(i.entry_id == "bogus.model" and "provider" in i.message for i in issues)
