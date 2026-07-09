from __future__ import annotations

from qortex.neuroai.models.zoo.registry import list_entries
from qortex.neuroai.models.zoo.schema import ZooEntryType
from qortex.neuroai.models.zoo.validate import validate_registry

_EXPECTED_IDS = {
    "monai.brain_image_synthesis_latent_diffusion_model",
    "monai.brats_mri_generative_diffusion",
    "monai.brats_mri_axial_slices_generative_diffusion",
    "monai.maisi_ct_generative",
    "monai.cxr_image_synthesis_latent_diffusion_model",
    "monai.mednist_ddpm",
    "monai.mednist_gan",
}


def test_all_7_generative_entries_registered():
    entries = list_entries(entry_type=ZooEntryType.generative_model)
    assert {e.id for e in entries} == _EXPECTED_IDS


def test_generative_entries_are_never_tagged_as_segmentation_or_classification():
    entries = list_entries(entry_type=ZooEntryType.generative_model)
    for entry in entries:
        assert entry.output_contract.output_type == "image_generation"
        assert entry.output_contract.produces_probabilities is False


def test_generative_entries_pass_offline_validation():
    issues = validate_registry()
    relevant = [i for i in issues if i.entry_id in _EXPECTED_IDS]
    assert relevant == []
