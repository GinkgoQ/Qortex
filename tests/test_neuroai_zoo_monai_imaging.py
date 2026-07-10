from __future__ import annotations

from qortex.neuroai.models.zoo.registry import list_entries, lookup
from qortex.neuroai.models.zoo.validate import validate_registry

_EXPECTED_IDS = {
    "monai.wholeBrainSeg_Large_UNEST_segmentation",
    "monai.vista3d",
    "monai.swin_unetr_btcv_segmentation",
    "monai.wholeBody_ct_segmentation",
    "monai.spleen_ct_segmentation",
    "monai.multi_organ_segmentation",
    "monai.pancreas_ct_dints_segmentation",
    "monai.prostate_mri_anatomy",
    "monai.renalStructures_CECT_segmentation",
    "monai.renalStructures_UNEST_segmentation",
    "monai.ventricular_short_axis_3label",
    "monai.valve_landmarks",
    "monai.retinalOCT_RPD_segmentation",
}


def test_all_13_monai_imaging_entries_registered():
    # monai.vista3d's provider is "vista3d" (Phase 5 promptable upgrade), not
    # "monai", so registration is checked by id via lookup() rather than by
    # filtering list_entries(provider="monai").
    assert all(lookup(entry_id) is not None for entry_id in _EXPECTED_IDS)
    registered_ids = {e.id for e in list_entries(provider="monai")}
    # brats_mri_segmentation (Phase 1 seed) + 12 other imaging entries (vista3d
    # now provider="vista3d") = 13 + 7 generative (Task 3) = 20 monai-provider
    # entries, plus 1 vista3d-provider entry = 21 total imaging+generative+seed.
    assert len(registered_ids) == 20
    assert lookup("monai.vista3d").provider == "vista3d"


def test_monai_imaging_entries_pass_offline_validation():
    issues = validate_registry()
    relevant = [i for i in issues if i.entry_id in _EXPECTED_IDS]
    assert relevant == []


def test_wholebody_ct_reuses_confirmed_legacy_contract():
    entry = lookup("monai.wholeBody_ct_segmentation")
    assert entry is not None
    assert entry.input_contract.n_channels == 1
    assert entry.input_contract.intensity_range == (-1024.0, 3071.0)
    assert entry.output_contract.n_classes == 105


def test_ventricular_short_axis_has_confirmed_3_classes():
    entry = lookup("monai.ventricular_short_axis_3label")
    assert entry.output_contract.n_classes == 3


def test_entries_without_confirmed_shape_leave_fields_unknown():
    entry = lookup("monai.swin_unetr_btcv_segmentation")
    assert entry.input_contract.n_channels is None
    assert entry.input_contract.evidence_status.value == "unknown"
