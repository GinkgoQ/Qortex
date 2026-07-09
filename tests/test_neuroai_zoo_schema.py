from __future__ import annotations

from qortex.neuroai.contracts import EvidenceStatus, InputContract, AxisConvention
from qortex.neuroai.models.zoo.schema import (
    ExecutionMode,
    ExternalEngineContract,
    InteractionContract,
    LicenseInfo,
    PromptCoordinateFrame,
    PromptType,
    SecurityPolicy,
    ZooEntry,
    ZooEntryType,
)


def test_evidence_status_has_contradicted_member():
    assert EvidenceStatus.contradicted == "contradicted"


def test_zoo_entry_minimal_model_construction():
    entry = ZooEntry(
        id="braindecode.EEGNet",
        display_name="EEGNet",
        entry_type=ZooEntryType.model,
        provider="braindecode",
        execution_mode=ExecutionMode.in_process,
        source_url="https://braindecode.org/stable/generated/braindecode.models.EEGNet.html",
        modality=["eeg"],
        task=["classification", "eeg_decoding", "bci"],
        input_contract=InputContract(
            modality="eeg",
            axis_convention=AxisConvention.batch_channels_time,
        ),
        license=LicenseInfo(evidence_status=EvidenceStatus.unknown),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="architecture_available",
        priority="P0",
    )
    assert entry.id == "braindecode.EEGNet"
    assert entry.entry_type is ZooEntryType.model
    assert entry.interaction_contract is None


def test_zoo_entry_promptable_construction():
    entry = ZooEntry(
        id="foundation.medsam",
        display_name="MedSAM",
        entry_type=ZooEntryType.promptable_model,
        provider="medsam",
        execution_mode=ExecutionMode.in_process,
        source_url="https://github.com/bowang-lab/MedSAM",
        modality=["ct", "mri"],
        task=["segmentation"],
        interaction_contract=InteractionContract(
            supported_prompt_types=[PromptType.point, PromptType.box],
            prompt_coordinate_frame=PromptCoordinateFrame.image_2d,
        ),
        license=LicenseInfo(evidence_status=EvidenceStatus.unknown),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
    )
    assert PromptType.text not in entry.interaction_contract.supported_prompt_types


def test_zoo_entry_external_engine_construction():
    entry = ZooEntry(
        id="external.totalsegmentator",
        display_name="TotalSegmentator",
        entry_type=ZooEntryType.external_engine,
        provider="external_cli",
        execution_mode=ExecutionMode.external_cli,
        source_url="https://github.com/wasserth/TotalSegmentator",
        modality=["ct", "mri"],
        task=["anatomical_segmentation"],
        external_engine_contract=ExternalEngineContract(
            engine="totalsegmentator",
            executable="TotalSegmentator",
            input_file_types=["nifti"],
            output_file_types=["nifti", "json"],
            supported_modalities=["ct", "mri"],
            supported_tasks=["total", "total_mr"],
            command_builder="_build_totalsegmentator_command",
            list_capabilities_command=["totalseg_info", "--json"],
            output_manifest_supported=True,
            evidence_status=EvidenceStatus.confirmed,
        ),
        license=LicenseInfo(evidence_status=EvidenceStatus.unknown),
        security=SecurityPolicy(executable_names=["TotalSegmentator"]),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_if_executable_available",
        priority="P0",
    )
    assert entry.external_engine_contract.executable == "TotalSegmentator"
    assert entry.input_contract is None
