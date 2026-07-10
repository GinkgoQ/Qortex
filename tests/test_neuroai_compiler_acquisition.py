from __future__ import annotations

import json

from qortex.neuroai.compiler import CompilationRequest, compile_neuroai
from qortex.neuroai.compiler.acquisition import build_local_companion_plan


def _build_bids_tree(tmp_path):
    (tmp_path / "dataset_description.json").write_text(
        json.dumps({"Name": "test", "BIDSVersion": "1.8.0"})
    )
    (tmp_path / "participants.tsv").write_text("participant_id\tage\nsub-01\t30\n")

    eeg_dir = tmp_path / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    stem = "sub-01_task-rest_eeg"
    (eeg_dir / f"{stem}.set").write_text("eeglab-header")
    (eeg_dir / f"{stem}.fdt").write_bytes(b"eeglab-binary-payload")
    (eeg_dir / f"{stem}.json").write_text(json.dumps({"SamplingFrequency": 250}))
    (eeg_dir / "sub-01_task-rest_channels.tsv").write_text("name\ttype\nCz\tEEG\n")
    (eeg_dir / "sub-01_task-rest_events.tsv").write_text("onset\tduration\n0\t1\n")
    return tmp_path


def test_build_local_companion_plan_finds_one_set_fdt_recording(tmp_path):
    tree = _build_bids_tree(tmp_path)

    plan = build_local_companion_plan(str(tree))

    assert plan.n_recordings == 1
    assert plan.source_type == "local_bids_directory"
    assert plan.required_download is False
    assert plan.estimated_download_gb == 0.0

    set_path = "sub-01/eeg/sub-01_task-rest_eeg.set"
    fdt_path = "sub-01/eeg/sub-01_task-rest_eeg.fdt"
    channels_path = "sub-01/eeg/sub-01_task-rest_channels.tsv"
    events_path = "sub-01/eeg/sub-01_task-rest_events.tsv"
    sidecar_path = "sub-01/eeg/sub-01_task-rest_eeg.json"

    assert set_path in plan.required_files
    assert fdt_path in plan.required_files
    assert fdt_path in plan.companion_files
    assert channels_path in plan.required_files
    assert events_path in plan.required_files
    assert sidecar_path in plan.required_files


def test_build_local_companion_plan_no_recordings_is_graceful(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    plan = build_local_companion_plan(str(empty_dir))

    assert plan.n_recordings == 0
    assert plan.required_files == []
    assert plan.notes


def test_compile_neuroai_wires_local_companion_plan(tmp_path):
    tree = _build_bids_tree(tmp_path)

    result = compile_neuroai(CompilationRequest(
        source=str(tree),
        task="classification",
        accept_unknown_license_risk=True,
    ))

    assert result.acquisition_plan.n_recordings == 1
    assert result.acquisition_plan.source_type == "local_bids_directory"
    fdt_path = "sub-01/eeg/sub-01_task-rest_eeg.fdt"
    assert fdt_path in result.acquisition_plan.required_files
