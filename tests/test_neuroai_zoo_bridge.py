from __future__ import annotations

import pytest

from qortex.neuroai.models import _contracts
from qortex.neuroai.models._contracts import lookup as legacy_lookup
from qortex.neuroai.models.zoo.bridge import sync_into_legacy_registry


@pytest.fixture(autouse=True)
def _isolated_legacy_registry():
    # ponytail: _contracts has no public reset, and its module-level list is
    # the only mutable state the bridge touches, so snapshot/restore it here
    # directly rather than adding a reset API nothing else needs.
    snapshot = list(_contracts._REGISTRY)
    yield
    _contracts._REGISTRY[:] = snapshot


def test_sync_registers_contracted_zoo_entries_into_legacy_registry():
    synced_count = sync_into_legacy_registry()

    assert synced_count > 0
    # braindecode.EEGNet has both input_contract and output_contract (Phase 1 seed)
    assert legacy_lookup("braindecode.EEGNet") is not None


def test_sync_skips_entries_without_both_contracts():
    sync_into_legacy_registry()

    # external.totalsegmentator has no input_contract (external CLI engine)
    assert legacy_lookup("external.totalsegmentator") is None


def test_sync_is_idempotent():
    first = sync_into_legacy_registry()
    second = sync_into_legacy_registry()

    assert second == 0
    assert first > 0


def test_synced_entry_preserves_original_contracts():
    sync_into_legacy_registry()

    entry = legacy_lookup("braindecode.EEGNet")

    assert entry.input_contract.modality == "eeg"
    assert entry.provider == "braindecode"


def test_sync_skips_model_already_present_under_its_legacy_bundle_name():
    # wholeBody_ct_segmentation is already a curated legacy entry (registered
    # under the un-namespaced bundle name before the zoo package existed).
    # The zoo entry for the same real model is "monai.wholeBody_ct_segmentation"
    # — the bridge must not register a second, duplicate legacy entry for it.
    assert legacy_lookup("wholeBody_ct_segmentation") is not None

    sync_into_legacy_registry()

    assert legacy_lookup("monai.wholeBody_ct_segmentation") is None
