"""Offline MONAI bundle metadata extractor.

Turns an already-loaded MONAI bundle metadata.json (and optionally
inference.json) dict into Qortex contract fields. Pure function — no
network access, no bundle download. Missing fields are left unknown, never
guessed, per docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md
section 11.1.

MONAI bundles conventionally describe I/O under
metadata["network_data_format"]["inputs"/"outputs"], each keyed by tensor
name with "type"/"format"/"num_channels"/"spatial_shape"/"dtype" — this is
MONAI's own public, stable bundle metadata convention, not something Qortex
invents.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qortex.neuroai.contracts import (
    AxisConvention,
    EvidenceStatus,
    InputContract,
    OutputContract,
)

# MONAI transform class names that Qortex already knows how to translate
# into its own preprocessing plan. Anything outside this set in a bundle's
# inference.json preprocessing chain is a custom callable Qortex cannot
# safely auto-apply.
_KNOWN_MONAI_TRANSFORMS = {
    "LoadImaged", "EnsureChannelFirstd", "Orientationd", "Spacingd",
    "ScaleIntensityRanged", "NormalizeIntensityd", "CropForegroundd",
    "Resized", "ToTensord", "EnsureTyped",
}


@dataclass
class ExtractedMONAIContract:
    model_id: str
    input_contract: InputContract | None = None
    output_contract: OutputContract | None = None
    unresolved_transforms: list[str] = field(default_factory=list)


def _extract_input_contract(inputs: dict) -> InputContract | None:
    if not inputs:
        return None
    # Bundles may declare multiple named inputs; Qortex's InputContract
    # models a single primary tensor, so take the first declared input.
    _, spec = next(iter(inputs.items()))
    n_channels = spec.get("num_channels")
    spatial_shape = spec.get("spatial_shape")
    dtype = spec.get("dtype")

    confirmed = n_channels is not None and spatial_shape is not None
    kwargs = dict(
        modality="mri",  # MONAI bundle inputs are volumetric medical images;
                          # the specific modality (mri/ct) is not encoded in
                          # network_data_format and must come from the zoo
                          # entry's own modality field, not this extractor.
        axis_convention=AxisConvention.channels_first,  # MONAI bundles conventionally use channels-first tensors,
                                                          # but this value is not derived from network_data_format
                                                          # and is therefore an assumption, not confirmed by evidence_status.
        n_channels=n_channels,
        spatial_shape=tuple(spatial_shape) if spatial_shape else None,
        evidence_status=EvidenceStatus.confirmed if confirmed else EvidenceStatus.inferred,
    )
    if dtype is not None:
        kwargs["dtype"] = dtype
    return InputContract(**kwargs)


def _extract_output_contract(outputs: dict) -> OutputContract | None:
    if not outputs:
        return None
    _, spec = next(iter(outputs.items()))
    n_channels = spec.get("num_channels")
    if n_channels is None:
        return None
    return OutputContract(
        output_type="segmentation",
        n_classes=n_channels,
        produces_probabilities=False,
    )


def _find_unresolved_transforms(inference: dict | None) -> list[str]:
    if not inference:
        return []
    unresolved = []
    for step in inference.get("preprocessing", []):
        target = step.get("_target_", "")
        class_name = target.rsplit(".", 1)[-1]
        if class_name not in _KNOWN_MONAI_TRANSFORMS:
            unresolved.append(target)
    return unresolved


def extract_monai_contract(
    model_id: str,
    metadata: dict,
    inference: dict | None = None,
) -> ExtractedMONAIContract:
    ndf = metadata.get("network_data_format", {})
    return ExtractedMONAIContract(
        model_id=model_id,
        input_contract=_extract_input_contract(ndf.get("inputs", {})),
        output_contract=_extract_output_contract(ndf.get("outputs", {})),
        unresolved_transforms=_find_unresolved_transforms(inference),
    )


__all__ = ["ExtractedMONAIContract", "extract_monai_contract"]
