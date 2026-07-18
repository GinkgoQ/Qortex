from __future__ import annotations

from qortex.core.entities import BIDSEntities, FileRecord, Manifest
from qortex.eda.coverage import observed_coverage_report


def _file(path: str, subject: str, run: str) -> FileRecord:
    return FileRecord(
        id=path,
        path=path,
        filename=path.rsplit("/", 1)[-1],
        extension=".nii.gz",
        datatype="func",
        suffix="bold",
        modality="fmri",
        entities=BIDSEntities(subject=subject, task="rest", run=run),
    )


def test_observed_coverage_does_not_invent_missing_expectations():
    manifest = Manifest(
        dataset_id="ds-test",
        snapshot="1.0.0",
        files=[
            _file("sub-01/func/sub-01_task-rest_run-1_bold.nii.gz", "01", "1"),
            _file("sub-01/func/sub-01_task-rest_run-2_bold.nii.gz", "01", "2"),
            _file("sub-02/func/sub-02_task-rest_run-1_bold.nii.gz", "02", "1"),
            FileRecord(
                id="sidecar", path="sub-01/func/sub-01_task-rest_run-1_bold.json",
                filename="sub-01_task-rest_run-1_bold.json", extension=".json",
                modality="fmri", suffix="bold",
                entities=BIDSEntities(subject="01", task="rest", run="1"),
            ),
        ],
    )

    report = observed_coverage_report(manifest)

    assert report["absence_semantics"] == "not_observed"
    assert [column["run"] for column in report["columns"]] == ["1", "2"]
    assert report["columns"][0]["label"] == "task-rest · run-1 · fmri · bold"
    assert [cell["status"] for cell in report["subjects"][1]["cells"]] == [
        "available", "not_observed",
    ]
    assert report["available_cells"] == 3
    assert report["visible_cells"] == 4
