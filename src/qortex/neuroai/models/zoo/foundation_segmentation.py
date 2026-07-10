"""MedSAM and SAM-Med3D promptable foundation segmentation entries
(design spec section 12.4). Point/box prompts only -- neither model's
real, documented interface supports text prompts or automatic
(promptless) mode, per section 8.1.

The static zoo entry keeps ``qortex_status="checkpoint_unresolved"``
because checkpoints are local runtime artifacts, not bundled registry data.
Installed artifact readiness is reported by the standardized resolver and
the ``qortex neuroai zoo artifact-status`` command.
"""

from __future__ import annotations

from qortex.neuroai.contracts import AxisConvention, EvidenceStatus, InputContract, OutputContract
from qortex.neuroai.models.zoo.registry import register
from qortex.neuroai.models.zoo.schema import (
    ExecutionMode,
    InteractionContract,
    LicenseInfo,
    PromptType,
    ZooEntry,
    ZooEntryType,
)


def _unknown_ct_input() -> InputContract:
    return InputContract(
        modality="ct",
        axis_convention=AxisConvention.channels_first,
        evidence_status=EvidenceStatus.unknown,
    )


def _point_box_contract() -> InteractionContract:
    return InteractionContract(
        supported_prompt_types=[PromptType.point, PromptType.box],
        supports_automatic_mode=False,
        evidence_status=EvidenceStatus.confirmed,
    )


def _unlicensed() -> LicenseInfo:
    return LicenseInfo(evidence_status=EvidenceStatus.unknown, notes=["requires manual check"])


def register_all() -> None:
    register(ZooEntry(
        id="foundation.medsam",
        display_name="MedSAM",
        entry_type=ZooEntryType.promptable_model,
        provider="medsam",
        execution_mode=ExecutionMode.in_process,
        source_url="https://github.com/bowang-lab/MedSAM",
        modality=["ct", "mri"],
        task=["segmentation"],
        input_contract=_unknown_ct_input(),
        output_contract=OutputContract(output_type="segmentation", produces_probabilities=False),
        interaction_contract=_point_box_contract(),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="checkpoint_unresolved",
        priority="P0",
    ))
    register(ZooEntry(
        id="foundation.sam_med3d",
        display_name="SAM-Med3D",
        entry_type=ZooEntryType.promptable_model,
        provider="sam_med3d",
        execution_mode=ExecutionMode.in_process,
        source_url="https://github.com/uni-medical/SAM-Med3D",
        paper_url="https://arxiv.org/abs/2310.15161",
        modality=["ct", "mri"],
        task=["segmentation"],
        input_contract=_unknown_ct_input(),
        output_contract=OutputContract(output_type="segmentation", produces_probabilities=False),
        interaction_contract=_point_box_contract(),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="checkpoint_unresolved",
        priority="P0",
        notes=[
            "Trained on 22K 3D images with 143K masks per arXiv:2310.15161 "
            "-- training-scale fact only, not a tensor contract field.",
        ],
    ))


__all__ = ["register_all"]
