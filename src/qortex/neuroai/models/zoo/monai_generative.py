"""P0 MONAI generative bundle entries (design spec section 12.5).

Generative models are never tagged as segmentation/classification —
output_type is always "image_generation" and produces_probabilities is
always False, per the spec's explicit invariant that a generative model
must not be mistaken for a diagnostic one.
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
    LicenseInfo,
    ZooEntry,
    ZooEntryType,
)

_MAINTAINER = "Project MONAI"
_CATALOG_URL = "https://project-monai.github.io/model-zoo.html"
_CLINICAL_USE_NOTE = "clinical_use=prohibited, research_use=allowed (design spec section 12.5)."


def _hub_url(bundle_name: str) -> str:
    return f"https://huggingface.co/MONAI/{bundle_name}"


def _generative_entry(bundle_name: str, display_name: str, modality: str, extra_notes: list[str] | None = None) -> ZooEntry:
    return ZooEntry(
        id=f"monai.{bundle_name}",
        display_name=display_name,
        entry_type=ZooEntryType.generative_model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url(bundle_name),
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=[modality],
        task=["image_generation", "synthesis"],
        input_contract=InputContract(
            modality=modality,
            axis_convention=AxisConvention.channels_first,
            evidence_status=EvidenceStatus.unknown,
        ),
        output_contract=OutputContract(
            output_type="image_generation",
            produces_probabilities=False,
        ),
        license=LicenseInfo(evidence_status=EvidenceStatus.unknown, notes=["requires manual check"]),
        evidence_status=EvidenceStatus.confirmed,
        # Not "runnable_after_contract_validation": there is no real
        # generative execution adapter yet (sampler/conditioning/seed
        # handling, synthetic-output writer). MONAIBundleAdapter's generic
        # segmentation-style forward pass must not be run against these --
        # see MONAIBundleAdapter.output_schema()/predict() in models/monai.py,
        # which now refuses generative entries rather than mislabeling their
        # output as a segmentation mask.
        qortex_status="checkpoint_unresolved",
        priority="P1",
        notes=[_CLINICAL_USE_NOTE] + (extra_notes or []),
    )


def register_all() -> None:
    register(_generative_entry(
        "brain_image_synthesis_latent_diffusion_model",
        "Brain Image Synthesis (Latent Diffusion)",
        "mri",
    ))
    register(_generative_entry(
        "brats_mri_generative_diffusion",
        "BraTS MRI Generative Diffusion",
        "mri",
    ))
    register(_generative_entry(
        "brats_mri_axial_slices_generative_diffusion",
        "BraTS MRI Axial Slices Generative Diffusion",
        "mri",
    ))
    register(_generative_entry(
        "maisi_ct_generative",
        "MAISI CT Generative",
        "ct",
        extra_notes=[
            "MAISI: diffusion-based synthetic 3D CT with anatomical control, "
            "up to 512x512x768 voxels conditioned on organ segmentations "
            "(design spec section 12.5).",
        ],
    ))
    register(_generative_entry(
        "cxr_image_synthesis_latent_diffusion_model",
        "Chest X-Ray Image Synthesis (Latent Diffusion)",
        "xray",
    ))
    register(_generative_entry(
        "mednist_ddpm",
        "MedNIST DDPM",
        "mixed",
    ))
    register(_generative_entry(
        "mednist_gan",
        "MedNIST GAN",
        "mixed",
    ))


def synthetic_data_notice(entry: ZooEntry) -> dict[str, object]:
    """Return the structured clinical-use notice for a generative entry."""

    if entry.entry_type != ZooEntryType.generative_model:
        raise ValueError(
            f"synthetic_data_notice() called on non-generative entry {entry.id!r} "
            f"(entry_type={entry.entry_type.value})"
        )
    return {
        "clinical_use": "prohibited",
        "research_use": "allowed",
        "watermark_synthetic": True,
        "require_generation_metadata": True,
    }


__all__ = ["register_all", "synthetic_data_notice"]
