"""P0 MONAI imaging bundle entries (design spec section 12.1), excluding
brats_mri_segmentation which was seeded in Phase 1
(zoo/seed_examples.py). Every quantitative field is only set when
confirmed by the design spec's own text or reused from an existing
confirmed entry in qortex.neuroai.models._contracts — everything else is
left unknown rather than guessed.
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

_MAINTAINER = "Project MONAI"
_CATALOG_URL = "https://project-monai.github.io/model-zoo.html"


def _unknown_input(modality: str) -> InputContract:
    return InputContract(
        modality=modality,
        axis_convention=AxisConvention.channels_first,
        evidence_status=EvidenceStatus.unknown,
    )


def _unknown_output(output_type: str = "segmentation") -> OutputContract:
    return OutputContract(output_type=output_type, produces_probabilities=False)


def _unlicensed() -> LicenseInfo:
    return LicenseInfo(evidence_status=EvidenceStatus.unknown, notes=["requires manual check"])


def _hub_url(bundle_name: str) -> str:
    return f"https://huggingface.co/MONAI/{bundle_name}"


def register_all() -> None:
    register(ZooEntry(
        id="monai.wholeBrainSeg_Large_UNEST_segmentation",
        display_name="Whole Brain Segmentation (Large UNEST)",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url("wholeBrainSeg_Large_UNEST_segmentation"),
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=["mri"],
        task=["segmentation", "whole_brain_segmentation"],
        input_contract=_unknown_input("mri"),
        output_contract=_unknown_output(),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
        notes=["T1w whole-brain structural segmentation with 133 structures per design spec section 12.1; exact class-index count not confirmed offline."],
    ))

    register(ZooEntry(
        id="monai.vista3d",
        display_name="VISTA3D",
        entry_type=ZooEntryType.promptable_model,
        provider="vista3d",
        execution_mode=ExecutionMode.bundle,
        source_url="https://huggingface.co/MONAI/VISTA3D-HF",
        paper_url="https://arxiv.org/abs/2406.05285",
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=["ct"],
        task=["segmentation", "foundation_segmentation"],
        input_contract=_unknown_input("ct"),
        output_contract=_unknown_output(),
        interaction_contract=InteractionContract(
            supported_prompt_types=[PromptType.point, PromptType.box],
            supports_automatic_mode=True,
            evidence_status=EvidenceStatus.confirmed,
        ),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="checkpoint_unresolved",
        priority="P0",
        notes=[
            "Foundation-style 3D CT segmentation and annotation.",
            "VISTA3D paper (arXiv:2406.05285) reports 127 automatic classes; "
            "not encoded as n_classes here since exact figure is unconfirmed offline.",
            "Phase 5 (promptable segmentation): upgraded to entry_type="
            "promptable_model with provider=vista3d (dedicated VISTA3DAdapter) "
            "and a confirmed InteractionContract (point/box prompts, automatic "
            "mode supported) — see models/monai.py VISTA3DAdapter. Runtime "
            "artifact readiness is reported by the standardized artifact "
            "resolver and the `qortex neuroai zoo artifact-status` command.",
        ],
    ))

    register(ZooEntry(
        id="monai.swin_unetr_btcv_segmentation",
        display_name="Swin UNETR BTCV Segmentation",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url("swin_unetr_btcv_segmentation"),
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=["ct"],
        task=["segmentation"],
        input_contract=_unknown_input("ct"),
        output_contract=_unknown_output(),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
        notes=["Transformer CT segmentation baseline per design spec section 12.1."],
    ))

    # Reuses confirmed contract data from the existing legacy
    # qortex.neuroai.models._contracts entry "wholeBody_ct_segmentation"
    # (n_channels, intensity_range, n_classes) describing the same real model.
    register(ZooEntry(
        id="monai.wholeBody_ct_segmentation",
        display_name="Whole Body CT Segmentation",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url("wholeBody_ct_segmentation"),
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=["ct"],
        task=["segmentation", "whole_body_segmentation"],
        input_contract=InputContract(
            modality="ct",
            axis_convention=AxisConvention.channels_first,
            n_channels=1,
            intensity_range=(-1024.0, 3071.0),
            dtype="float32",
            evidence_status=EvidenceStatus.confirmed,
        ),
        output_contract=OutputContract(
            output_type="segmentation",
            n_classes=105,
            produces_probabilities=False,
        ),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
        notes=["Contract reused from the existing curated qortex.neuroai.models._contracts entry for the same model."],
    ))

    register(ZooEntry(
        id="monai.spleen_ct_segmentation",
        display_name="Spleen CT Segmentation",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url("spleen_ct_segmentation"),
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=["ct"],
        task=["segmentation"],
        input_contract=_unknown_input("ct"),
        output_contract=_unknown_output(),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
        notes=["Small MONAI bundle useful for tests/demos per design spec section 12.1."],
    ))

    register(ZooEntry(
        id="monai.multi_organ_segmentation",
        display_name="Multi-Organ Segmentation",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url("multi_organ_segmentation"),
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=["ct"],
        task=["segmentation"],
        input_contract=_unknown_input("ct"),
        output_contract=_unknown_output(),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
    ))

    register(ZooEntry(
        id="monai.pancreas_ct_dints_segmentation",
        display_name="Pancreas CT DiNTS Segmentation",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url("pancreas_ct_dints_segmentation"),
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=["ct"],
        task=["segmentation"],
        input_contract=_unknown_input("ct"),
        output_contract=_unknown_output(),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
    ))

    register(ZooEntry(
        id="monai.prostate_mri_anatomy",
        display_name="Prostate MRI Anatomy",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url("prostate_mri_anatomy"),
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=["mri"],
        task=["segmentation"],
        input_contract=_unknown_input("mri"),
        output_contract=_unknown_output(),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
    ))

    register(ZooEntry(
        id="monai.renalStructures_CECT_segmentation",
        display_name="Renal Structures CECT Segmentation",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url("renalStructures_CECT_segmentation"),
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=["ct"],
        task=["segmentation"],
        input_contract=_unknown_input("ct"),
        output_contract=_unknown_output(),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
    ))

    register(ZooEntry(
        id="monai.renalStructures_UNEST_segmentation",
        display_name="Renal Structures UNEST Segmentation",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url("renalStructures_UNEST_segmentation"),
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=["ct"],
        task=["segmentation"],
        input_contract=_unknown_input("ct"),
        output_contract=_unknown_output(),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
    ))

    # "3label" in the bundle name confirms exactly 3 output classes.
    register(ZooEntry(
        id="monai.ventricular_short_axis_3label",
        display_name="Ventricular Short Axis (3-label)",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url("ventricular_short_axis_3label"),
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=["mri"],
        task=["segmentation", "cardiac_segmentation"],
        input_contract=_unknown_input("mri"),
        output_contract=OutputContract(
            output_type="segmentation",
            n_classes=3,
            produces_probabilities=False,
        ),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
    ))

    register(ZooEntry(
        id="monai.valve_landmarks",
        display_name="Valve Landmarks",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url("valve_landmarks"),
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=["mri"],
        task=["landmark_detection"],
        input_contract=_unknown_input("mri"),
        output_contract=OutputContract(output_type="landmark_detection", produces_probabilities=False),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
    ))

    register(ZooEntry(
        id="monai.retinalOCT_RPD_segmentation",
        display_name="Retinal OCT RPD Segmentation",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url("retinalOCT_RPD_segmentation"),
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=["oct"],
        task=["segmentation"],
        input_contract=_unknown_input("oct"),
        output_contract=_unknown_output(),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
    ))


__all__ = ["register_all"]
