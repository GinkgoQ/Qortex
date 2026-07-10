from __future__ import annotations

from qortex.neuroai.contracts import AxisConvention, EvidenceStatus, InputContract
from qortex.neuroai.models.zoo.registry import register
from qortex.neuroai.models.zoo.schema import (
    ExecutionMode,
    InteractionContract,
    LicenseInfo,
    PromptType,
    ZooEntry,
    ZooEntryType,
)
from qortex.neuroai.models.zoo.validate import validate_registry


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
    # "braindecode.EEGNet" is already a zoo seed entry id (pre-registered by
    # the autouse conftest fixture); use a distinct id to avoid colliding.
    register(ZooEntry(entry_type=ZooEntryType.model, **_base_kwargs("braindecode.TestOK")))
    issues = validate_registry()
    # Filter to only issues for the entry we just registered
    test_entry_issues = [i for i in issues if i.entry_id == "braindecode.TestOK"]
    assert test_entry_issues == []


def test_malformed_source_url_is_an_error():
    kwargs = _base_kwargs("braindecode.Bad")
    kwargs["source_url"] = "not-a-url"
    register(ZooEntry(entry_type=ZooEntryType.model, **kwargs))
    issues = validate_registry()
    assert any(i.entry_id == "braindecode.Bad" and i.severity == "error" for i in issues)


def test_promptable_entry_without_interaction_contract_is_an_error():
    # "foundation.medsam" is now a real registered zoo entry (Phase 5); use
    # a distinct placeholder id to avoid colliding with it.
    register(ZooEntry(entry_type=ZooEntryType.promptable_model, **_base_kwargs("foundation.TestPromptable")))
    issues = validate_registry()
    assert any(
        i.entry_id == "foundation.TestPromptable" and "interaction_contract" in i.message
        for i in issues
    )


def test_promptable_entry_with_interaction_contract_passes():
    kwargs = _base_kwargs("foundation.TestPromptableOK")
    kwargs["interaction_contract"] = InteractionContract(
        supported_prompt_types=[PromptType.point, PromptType.box]
    )
    register(ZooEntry(entry_type=ZooEntryType.promptable_model, **kwargs))
    issues = validate_registry()
    # Filter to only issues for the entry we just registered
    test_entry_issues = [i for i in issues if i.entry_id == "foundation.TestPromptableOK"]
    assert test_entry_issues == []


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


def test_eeg_entry_with_unknown_shape_and_confirmed_status_gets_a_warning():
    kwargs = _base_kwargs("braindecode.Unconfirmed")
    kwargs["input_contract"] = InputContract(
        modality="eeg",
        axis_convention=AxisConvention.batch_channels_time,
        evidence_status=EvidenceStatus.unknown,
    )
    register(ZooEntry(entry_type=ZooEntryType.model, **kwargs))

    issues = validate_registry()

    matches = [i for i in issues if i.entry_id == "braindecode.Unconfirmed" and i.severity == "warning"]
    assert len(matches) == 1
    assert "evidence_status=confirmed" in matches[0].message


def test_eeg_entry_with_confirmed_channels_and_rate_gets_no_warning():
    kwargs = _base_kwargs("braindecode.Confirmed")
    kwargs["input_contract"] = InputContract(
        modality="eeg",
        axis_convention=AxisConvention.batch_channels_time,
        n_channels=64,
        sampling_rate_hz=250.0,
        evidence_status=EvidenceStatus.confirmed,
    )
    register(ZooEntry(entry_type=ZooEntryType.model, **kwargs))

    issues = validate_registry()

    assert [i for i in issues if i.entry_id == "braindecode.Confirmed"] == []


def test_non_eeg_entry_is_not_subject_to_the_eeg_check():
    kwargs = _base_kwargs("monai.SomeImaging")
    kwargs["input_contract"] = InputContract(
        modality="ct",
        axis_convention=AxisConvention.channels_first,
        evidence_status=EvidenceStatus.unknown,
    )
    register(ZooEntry(entry_type=ZooEntryType.model, **kwargs))

    issues = validate_registry()

    assert [i for i in issues if i.entry_id == "monai.SomeImaging"] == []
