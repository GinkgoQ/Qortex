from __future__ import annotations

from qortex.neuroai.models.zoo.registry import list_entries, lookup
from qortex.neuroai.models.zoo.validate import validate_registry

_EXPECTED_IDS = {
    "braindecode.Deep4Net",
    "braindecode.ShallowFBCSPNet",
    "braindecode.EEGConformer",
    "braindecode.BENDR",
    "braindecode.BIOT",
    "braindecode.Labram",
    "braindecode.REVE",
    "braindecode.USleep",
    "braindecode.AttnSleep",
    "braindecode.DeepSleepNet",
    "braindecode.SignalJEPA",
}


def test_all_11_braindecode_entries_registered():
    registered_ids = {e.id for e in list_entries(provider="braindecode")}
    # braindecode.EEGNet (Phase 1 seed) + these 11 = 12 braindecode entries
    assert _EXPECTED_IDS.issubset(registered_ids)
    assert len(registered_ids) == 12


def test_braindecode_entries_pass_offline_validation():
    issues = validate_registry()
    relevant = [i for i in issues if i.entry_id in _EXPECTED_IDS and i.severity == "error"]
    assert relevant == []


def test_sleep_staging_models_get_sleep_task():
    for model_id in ("braindecode.USleep", "braindecode.AttnSleep", "braindecode.DeepSleepNet"):
        entry = lookup(model_id)
        assert "sleep_staging" in entry.task


def test_bci_models_do_not_get_sleep_task():
    entry = lookup("braindecode.Deep4Net")
    assert "sleep_staging" not in entry.task


def test_no_entry_has_fabricated_channel_count():
    for model_id in _EXPECTED_IDS:
        entry = lookup(model_id)
        assert entry.input_contract.n_channels is None
        assert entry.input_contract.evidence_status.value == "unknown"


def test_labram_and_reve_cite_pretraining_scale_in_notes_not_fields():
    labram = lookup("braindecode.Labram")
    reve = lookup("braindecode.REVE")
    assert any("2,500 hours" in n for n in labram.notes)
    assert any("60,000" in n for n in reve.notes)
    # Pretraining scale is a fact about training data, not the model's
    # tensor contract — must never appear as a fabricated n_channels/etc.
    assert labram.input_contract.n_channels is None
    assert reve.input_contract.n_channels is None
