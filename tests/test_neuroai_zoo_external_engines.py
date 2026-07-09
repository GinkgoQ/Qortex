from __future__ import annotations

from qortex.neuroai.models.zoo.registry import list_entries, lookup
from qortex.neuroai.models.zoo.schema import ZooEntryType
from qortex.neuroai.models.zoo.validate import validate_registry

_EXPECTED_IDS = {
    "external.synthseg",
    "external.synthstrip",
    "external.hdbet",
    "external.fastsurfer",
    "external.tractseg",
}


def test_all_5_external_engine_entries_registered():
    registered_ids = {e.id for e in list_entries(entry_type=ZooEntryType.external_engine)}
    # external.totalsegmentator (Phase 1 seed) + these 5 = 6 external engines
    assert _EXPECTED_IDS.issubset(registered_ids)
    assert len(registered_ids) == 6


def test_external_engine_entries_pass_offline_validation():
    issues = validate_registry()
    relevant = [i for i in issues if i.entry_id in _EXPECTED_IDS]
    assert relevant == []


def test_no_external_engine_entry_has_a_tensor_input_contract():
    for entry_id in _EXPECTED_IDS:
        entry = lookup(entry_id)
        assert entry.input_contract is None
        assert entry.external_engine_contract is not None


def test_fastsurfer_declares_directory_output():
    entry = lookup("external.fastsurfer")
    assert "directory" in entry.external_engine_contract.output_file_types


def test_command_builder_names_match_external_py_function_names():
    from qortex.neuroai import external as external_module

    for entry_id in _EXPECTED_IDS:
        entry = lookup(entry_id)
        builder_name = entry.external_engine_contract.command_builder
        assert hasattr(external_module, builder_name), (
            f"{entry_id}'s command_builder={builder_name!r} does not exist in external.py"
        )
