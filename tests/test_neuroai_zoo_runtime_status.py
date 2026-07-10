from __future__ import annotations

from qortex.neuroai.models.zoo.registry import lookup
from qortex.neuroai.models.zoo.status import RuntimeStatus, runtime_status


def test_checkpoint_unresolved_promptable_entries_are_not_runtime_executable():
    for entry_id in ("foundation.medsam", "foundation.sam_med3d", "monai.vista3d"):
        entry = lookup(entry_id)
        assert entry is not None
        assert runtime_status(entry) == RuntimeStatus.checkpoint_unresolved


def test_external_engine_entry_is_only_executable_if_binary_available():
    entry = lookup("external.totalsegmentator")
    assert entry is not None
    assert runtime_status(entry) == RuntimeStatus.runnable_if_executable_available
