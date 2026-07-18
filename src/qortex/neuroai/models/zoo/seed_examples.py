"""The three worked examples from the design spec (section 13), registered
verbatim as real ZooEntry instances so every layer of Phase 1 has real data
to validate against. Domain-specific entries (MONAI imaging bundles,
Braindecode expansion, external engines) land in Phases 2-4.
"""

from __future__ import annotations

from qortex.neuroai.contracts import (
    AxisConvention,
    EvidenceStatus,
    InputContract,
    OutputContract,
)
from qortex.neuroai.models.zoo.registry import register
from qortex.neuroai.models.zoo.schema import (
    ExecutionMode,
    ExternalEngineContract,
    LicenseInfo,
    SecurityPolicy,
    ZooEntry,
    ZooEntryType,
)


def _register_brats_mri_segmentation() -> None:
    register(ZooEntry(
        id="monai.brats_mri_segmentation",
        display_name="BraTS MRI Segmentation",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url="https://huggingface.co/MONAI/brats_mri_segmentation",
        model_url="https://huggingface.co/MONAI/brats_mri_segmentation/blob/main/models/model.pt",
        paper_url="https://arxiv.org/abs/1810.11654",
        docs_url="https://project-monai.github.io/model-zoo.html",
        maintainer="Project MONAI",
        modality=["mri"],
        task=["segmentation", "brain_tumor_segmentation"],
        input_contract=InputContract(
            modality="mri",
            axis_convention=AxisConvention.channels_first,
            required_channels=["T1", "T1c", "T2", "FLAIR"],
            n_channels=4,
            evidence_status=EvidenceStatus.confirmed,
        ),
        output_contract=OutputContract(
            output_type="segmentation_mask",
            classes=["tumor_core", "whole_tumor", "enhancing_tumor"],
            produces_probabilities=False,
        ),
        license=LicenseInfo(
            name="Apache-2.0",
            url="https://huggingface.co/MONAI/brats_mri_segmentation/blob/main/LICENSE",
            commercial_use=True,
            redistribution_allowed=True,
            requires_citation=False,
            evidence_status=EvidenceStatus.confirmed,
            notes=["Model bundle license; input datasets retain their own licenses."],
        ),
        security=SecurityPolicy(
            network_required_for_download=True,
            network_required_at_runtime=False,
        ),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
    ))


def _register_braindecode_eegnet() -> None:
    register(ZooEntry(
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
            required_metadata=["n_chans", "n_times"],
            evidence_status=EvidenceStatus.inferred,
        ),
        output_contract=OutputContract(
            output_type="class_logits",
            produces_probabilities=False,
        ),
        license=LicenseInfo(evidence_status=EvidenceStatus.unknown),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="architecture_available",
        priority="P0",
    ))


def _register_external_totalsegmentator() -> None:
    register(ZooEntry(
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
    ))


def register_all() -> None:
    _register_brats_mri_segmentation()
    _register_braindecode_eegnet()
    _register_external_totalsegmentator()


__all__ = ["register_all"]
