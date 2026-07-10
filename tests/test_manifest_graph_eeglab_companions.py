from __future__ import annotations

from qortex.core.entities import BIDSEntities, FileRecord, Manifest
from qortex.manifest.graph import ManifestGraph


def _eeg_file(filename: str, extension: str) -> FileRecord:
    path = f"sub-036/eeg/{filename}"
    return FileRecord(
        id=path,
        path=path,
        filename=filename,
        extension=extension,
        size=1024,
        urls=[f"https://example.org/{path}"],
        datatype="eeg",
        suffix="eeg",
        modality="eeg",
        entities=BIDSEntities(subject="036", task="rest"),
    )


def _manifest_with_eeglab_pair() -> Manifest:
    set_file = _eeg_file("sub-036_task-rest_eeg.set", ".set")
    fdt_file = _eeg_file("sub-036_task-rest_eeg.fdt", ".fdt")
    return Manifest(dataset_id="ds999999", snapshot="1.0.0", files=[set_file, fdt_file])


def test_fdt_file_is_never_its_own_primary_recording():
    graph = ManifestGraph(_manifest_with_eeglab_pair())

    recordings = graph.recordings()

    assert len(recordings) == 1
    assert recordings[0].primary.extension == ".set"


def test_set_primary_gets_fdt_as_a_required_companion():
    graph = ManifestGraph(_manifest_with_eeglab_pair())

    recording = graph.recordings()[0]

    assert recording.companions.eeg_data is not None
    assert recording.companions.eeg_data.extension == ".fdt"
    companion_paths = {f.path for f in recording.companions.files}
    assert recording.companions.eeg_data.path in companion_paths


def test_companion_closure_includes_the_fdt_file_for_a_set_primary():
    manifest = _manifest_with_eeglab_pair()
    graph = ManifestGraph(manifest)
    set_file = next(f for f in manifest.files if f.extension == ".set")

    expanded = graph.companion_closure([set_file])

    assert {f.extension for f in expanded} == {".set", ".fdt"}
