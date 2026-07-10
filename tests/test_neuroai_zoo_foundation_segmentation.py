from __future__ import annotations

from qortex.neuroai.models.zoo.registry import lookup
from qortex.neuroai.models.zoo.validate import validate_registry

_EXPECTED_IDS = {"foundation.medsam", "foundation.sam_med3d"}


def test_both_entries_registered_as_promptable():
    for entry_id in _EXPECTED_IDS:
        entry = lookup(entry_id)
        assert entry is not None
        assert entry.entry_type.value == "promptable_model"


def test_neither_entry_declares_text_or_automatic_mode():
    for entry_id in _EXPECTED_IDS:
        entry = lookup(entry_id)
        ic = entry.interaction_contract
        assert "text" not in {t.value for t in ic.supported_prompt_types}
        assert ic.supports_automatic_mode is False


def test_foundation_entries_pass_offline_validation():
    issues = validate_registry()
    relevant = [i for i in issues if i.entry_id in _EXPECTED_IDS]
    assert relevant == []
