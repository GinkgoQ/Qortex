from __future__ import annotations

from pathlib import Path

from qortex.core.entities import BIDSEntities, FileRecord, LabelPolicy, Manifest, ManifestSummary
from qortex.decision import can_train


def _manifest() -> Manifest:
    files: list[FileRecord] = []
    for subject in ("01", "02"):
        stem = f"sub-{subject}_task-faces_run-01"
        entities = BIDSEntities(subject=subject, task="faces", run="01")
        files.extend([
            FileRecord(
                id=f"{stem}-bold",
                path=f"sub-{subject}/func/{stem}_bold.nii.gz",
                filename=f"{stem}_bold.nii.gz",
                extension=".nii.gz",
                datatype="func",
                suffix="bold",
                modality="fmri",
                size=1024,
                urls=[f"https://example.org/{stem}_bold.nii.gz"],
                entities=entities,
            ),
            FileRecord(
                id=f"{stem}-events",
                path=f"sub-{subject}/func/{stem}_events.tsv",
                filename=f"{stem}_events.tsv",
                extension=".tsv",
                datatype="func",
                suffix="events",
                modality="behavior",
                size=64,
                urls=[f"https://example.org/{stem}_events.tsv"],
                entities=entities,
            ),
        ])
    return Manifest(
        dataset_id="ds-test",
        snapshot="1.0.0",
        files=files,
        summary=ManifestSummary(
            subjects=["01", "02"], tasks=["faces"], modalities=["fmri"],
            datatypes=["func"], suffixes=["bold", "events"], file_count=len(files),
            has_events=True,
        ),
    )


def _write_events(root: Path) -> None:
    for subject in ("01", "02"):
        path = root / f"sub-{subject}/func/sub-{subject}_task-faces_run-01_events.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("onset\tduration\tstim_type\n0\t1\tface\n", encoding="utf-8")


def test_explicit_local_label_policy_confirms_labels(tmp_path: Path) -> None:
    _write_events(tmp_path)

    report = can_train(
        _manifest(),
        modality="fmri",
        local_path=tmp_path,
        label_policy=LabelPolicy(column="stim_type", missing="error"),
        split_strategy="subject",
    )

    assert report.status == "possible"
    assert report.label_status == "confirmed"
    assert report.n_label_ready == 2
    assert report.split_group_count == 2
    assert report.split_status == "valid"
    assert report.label_policy is not None
    assert report.label_policy.column == "stim_type"


def test_missing_policy_column_fails_closed(tmp_path: Path) -> None:
    _write_events(tmp_path)

    report = can_train(
        _manifest(),
        local_path=tmp_path,
        label_policy=LabelPolicy(column="diagnosis"),
    )

    assert report.status == "not_possible"
    assert report.label_status == "missing"
    assert report.n_label_ready == 0
    assert sum(f.code == "labels.policy_column_missing" for f in report.findings) == 2


def test_explicit_recording_split_reports_subject_leakage_risk(tmp_path: Path) -> None:
    _write_events(tmp_path)

    report = can_train(
        _manifest(),
        local_path=tmp_path,
        label_policy=LabelPolicy(column="stim_type"),
        split_strategy="recording",
    )

    assert report.split_group_count == 2
    assert report.split_status == "valid"
    assert any("multiple partitions" in risk for risk in report.leakage_risks)


def test_modality_scope_applies_to_readiness_and_split_counts(tmp_path: Path) -> None:
    _write_events(tmp_path)

    report = can_train(
        _manifest(),
        modality="eeg",
        local_path=tmp_path,
        label_policy=LabelPolicy(column="stim_type"),
    )

    assert report.status == "not_possible"
    assert report.n_recordings == 0
    assert report.n_subjects == 0
    assert report.n_label_ready == 0
    assert report.split_group_count == 0
