"""Deterministic tests for the qortex.checks data integrity system."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qortex.checks import (
    CheckReport,
    CheckSeverity,
    EvidenceRecord,
    EvidenceState,
    PreflightReport,
    SuggestedFix,
    run_preflight,
)
from qortex.checks._report import CheckFinding
from qortex.checks.domains import (
    EventsChecker,
    GeometryChecker,
    LeakageChecker,
    StructureChecker,
    UnitsChecker,
)
from qortex.checks.lazy import lazy_check_dataset, LazyCheckResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_bids(tmp_path: Path) -> Path:
    """A minimal valid BIDS dataset with one EEG subject."""
    root = tmp_path / "ds_test"
    root.mkdir()
    (root / "dataset_description.json").write_text(json.dumps({
        "Name": "Test", "BIDSVersion": "1.8.0"
    }))
    (root / "participants.tsv").write_text(
        "participant_id\tdiagnosis\tsite\n"
        "sub-01\tcontrol\tA\n"
        "sub-02\tpatient\tB\n"
        "sub-03\tcontrol\tA\n"
        "sub-04\tpatient\tB\n"
        "sub-05\tcontrol\tA\n"
        "sub-06\tpatient\tB\n"
    )
    for sub in ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]:
        eeg_dir = root / sub / "eeg"
        eeg_dir.mkdir(parents=True)
        stem = f"{sub}_task-rest_eeg"
        (eeg_dir / f"{stem}.edf").write_bytes(b"0" * 256)  # dummy
        (eeg_dir / f"{stem}.json").write_text(json.dumps({
            "SamplingFrequency": 256.0, "EEGReference": "Cz", "PowerLineFrequency": 50,
        }))
        (eeg_dir / f"{stem}_channels.tsv").write_text(
            "name\ttype\tunits\nFp1\tEEG\tuV\nFp2\tEEG\tuV\n"
        )
        (eeg_dir / f"{stem}_events.tsv").write_text(
            "onset\tduration\ttrial_type\n0.5\t1.0\teyes_open\n2.5\t1.0\teyes_closed\n"
        )
    return root


# ── CheckReport core ──────────────────────────────────────────────────────────

def test_check_report_finalize_pass():
    r = CheckReport(name="x", scope="s")
    r.finalize()
    assert r.status == CheckSeverity.PASS


def test_check_report_finalize_block_precedence():
    r = CheckReport(name="x", scope="s")
    r.add(CheckFinding(code="W", severity=CheckSeverity.WARN, message="w"))
    r.add(CheckFinding(code="B", severity=CheckSeverity.BLOCK, message="b"))
    r.finalize()
    assert r.status == CheckSeverity.BLOCK
    assert len(r.blockers) == 1
    assert len(r.warnings) == 1


def test_check_report_tracks_affected_files_and_subjects():
    r = CheckReport(name="x", scope="s")
    r.add(CheckFinding(
        code="W", severity=CheckSeverity.WARN, message="w",
        path="/data/sub-01_eeg.edf", bids_entities={"subject": "01"},
    ))
    r.finalize()
    assert "/data/sub-01_eeg.edf" in r.affected_files
    assert "01" in r.affected_subjects


def test_check_report_serializable():
    r = CheckReport(name="x", scope="s")
    r.add(CheckFinding(
        code="W", severity=CheckSeverity.WARN, message="w",
        suggested_fix=SuggestedFix(description="do x"),
    ))
    r.finalize()
    d = r.to_dict()
    assert d["status"] == "WARN"
    # round-trips through JSON
    json.dumps(d)


def test_evidence_record_serializable():
    e = EvidenceRecord(
        field="SamplingFrequency", state=EvidenceState.contradicted,
        claimed_value=512, observed_value=500,
    )
    d = e.to_dict()
    assert d["state"] == "contradicted"
    json.dumps(d)


def test_seven_evidence_states_exist():
    expected = {"confirmed", "inferred", "claimed", "missing",
                "contradicted", "unknown", "blocked"}
    actual = {s.value for s in EvidenceState}
    assert expected == actual


# ── Structure checker ─────────────────────────────────────────────────────────

def test_structure_checker_pass(minimal_bids: Path):
    report = StructureChecker(modality="eeg").run(minimal_bids)
    assert report.status != CheckSeverity.BLOCK


def test_structure_checker_missing_path(tmp_path: Path):
    report = StructureChecker().run(tmp_path / "nonexistent")
    assert report.status == CheckSeverity.BLOCK
    assert any("PATH_NOT_FOUND" in b.code for b in report.blockers)


def test_structure_checker_no_subjects(tmp_path: Path):
    root = tmp_path / "empty"
    root.mkdir()
    (root / "dataset_description.json").write_text("{}")
    report = StructureChecker().run(root)
    assert report.status == CheckSeverity.BLOCK


# ── Events checker ────────────────────────────────────────────────────────────

def test_events_checker_valid(minimal_bids: Path):
    report = EventsChecker(modality="eeg").run(minimal_bids)
    assert report.status != CheckSeverity.BLOCK


def test_events_checker_missing_onset(tmp_path: Path):
    root = tmp_path / "ds"
    eeg = root / "sub-01" / "eeg"
    eeg.mkdir(parents=True)
    (eeg / "sub-01_task-x_events.tsv").write_text("duration\ttrial_type\n1.0\ta\n")
    report = EventsChecker().run(root)
    assert report.status == CheckSeverity.BLOCK
    assert any("MISSING_ONSET" in b.code for b in report.blockers)


def test_events_checker_negative_onset(tmp_path: Path):
    root = tmp_path / "ds"
    eeg = root / "sub-01" / "eeg"
    eeg.mkdir(parents=True)
    (eeg / "sub-01_task-x_events.tsv").write_text("onset\tduration\n-1.0\t1.0\n")
    report = EventsChecker().run(root)
    assert any("NEGATIVE_ONSET" in w.code for w in report.warnings)


def test_events_checker_no_files(tmp_path: Path):
    root = tmp_path / "ds"
    root.mkdir()
    report = EventsChecker().run(root)
    # No events is INFO, not BLOCK
    assert report.status in (CheckSeverity.INFO, CheckSeverity.PASS)


# ── Geometry checker ──────────────────────────────────────────────────────────

def test_geometry_checker_no_nifti(minimal_bids: Path):
    report = GeometryChecker().run(minimal_bids)
    # No NIfTI → INFO
    assert report.status in (CheckSeverity.INFO, CheckSeverity.PASS)


def test_geometry_dwi_bval_bvec_mismatch(tmp_path: Path):
    root = tmp_path / "ds"
    dwi = root / "sub-01" / "dwi"
    dwi.mkdir(parents=True)
    (dwi / "sub-01_dwi.bval").write_text("0 1000 1000 1000")
    (dwi / "sub-01_dwi.bvec").write_text("0 1 0\n0 0 1\n1 0 0")  # only 3 vectors
    report = GeometryChecker(
        check_affine=False, check_voxel_size_consistency=False, check_dwi_gradients=True,
    ).run(root)
    assert report.status == CheckSeverity.BLOCK
    assert any("COUNT_MISMATCH" in b.code for b in report.blockers)


def test_geometry_dwi_valid(tmp_path: Path):
    root = tmp_path / "ds"
    dwi = root / "sub-01" / "dwi"
    dwi.mkdir(parents=True)
    (dwi / "sub-01_dwi.bval").write_text("0 1000 1000 1000")
    (dwi / "sub-01_dwi.bvec").write_text(
        "0 1 0 0\n0 0 1 0\n0 0 0 1"
    )
    report = GeometryChecker(
        check_affine=False, check_voxel_size_consistency=False, check_dwi_gradients=True,
    ).run(root)
    assert report.status != CheckSeverity.BLOCK


# ── Units checker ─────────────────────────────────────────────────────────────

def test_units_checker_valid(minimal_bids: Path):
    report = UnitsChecker(modality="eeg", check_signal_scale=False).run(minimal_bids)
    assert report.status != CheckSeverity.BLOCK


def test_units_checker_unknown_unit(tmp_path: Path):
    root = tmp_path / "ds"
    eeg = root / "sub-01" / "eeg"
    eeg.mkdir(parents=True)
    (eeg / "sub-01_channels.tsv").write_text("name\ttype\tunits\nFp1\tEEG\tbananas\n")
    report = UnitsChecker(check_signal_scale=False).run(root)
    assert any("UNKNOWN_UNIT" in w.code for w in report.warnings)


# ── Leakage checker ───────────────────────────────────────────────────────────

def test_leakage_target_missing(minimal_bids: Path):
    report = LeakageChecker(target="nonexistent_column").run(minimal_bids)
    assert report.status == CheckSeverity.BLOCK
    assert any("TARGET_COLUMN_MISSING" in b.code for b in report.blockers)


def test_leakage_confound_association(minimal_bids: Path):
    # site is perfectly associated with diagnosis in fixture
    report = LeakageChecker(
        target="diagnosis", confound_columns=["site"],
    ).run(minimal_bids)
    # Perfect association → should be flagged
    assert any("CONFOUND_ASSOCIATION" in f.code for f in report.all_findings)


def test_leakage_valid_target(minimal_bids: Path):
    report = LeakageChecker(target="diagnosis", confound_columns=[]).run(minimal_bids)
    # diagnosis has both classes and full coverage
    assert not any("LOW_LABEL_COVERAGE" in b.code for b in report.blockers)


# ── Preflight ─────────────────────────────────────────────────────────────────

def test_preflight_train_goal(minimal_bids: Path):
    report = run_preflight(
        minimal_bids, goal="train", modality="eeg", target="diagnosis",
    )
    assert isinstance(report, PreflightReport)
    assert report.goal == "train"
    assert len(report.checks) > 0
    json.dumps(report.to_dict())


def test_preflight_visualize_goal(minimal_bids: Path):
    report = run_preflight(minimal_bids, goal="visualize", modality="eeg")
    assert report.goal == "visualize"
    assert report.status in CheckSeverity


def test_preflight_invalid_goal(minimal_bids: Path):
    with pytest.raises(ValueError):
        run_preflight(minimal_bids, goal="not_a_goal")


def test_preflight_status_is_worst_check(minimal_bids: Path):
    report = run_preflight(
        minimal_bids, goal="train", target="missing_col",
    )
    # missing target → leakage check BLOCKs → preflight BLOCKs
    assert report.status == CheckSeverity.BLOCK


# ── Lazy checks ───────────────────────────────────────────────────────────────

def test_lazy_check_off(minimal_bids: Path, monkeypatch):
    monkeypatch.setenv("QORTEX_LAZY_CHECKS", "off")
    result = lazy_check_dataset(minimal_bids)
    assert result.mode == "off"
    assert not result.hints


def test_lazy_check_detects_empty_file(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("QORTEX_LAZY_CHECKS", "warn")
    root = tmp_path / "ds"
    sub = root / "sub-01" / "eeg"
    sub.mkdir(parents=True)
    (sub / "sub-01_eeg.edf").write_bytes(b"")  # empty
    result = lazy_check_dataset(root)
    assert any("EMPTY_FILE" in h.code for h in result.hints)


def test_lazy_check_strict_raises(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("QORTEX_LAZY_CHECKS", "strict")
    root = tmp_path / "ds"
    sub = root / "sub-01" / "eeg"
    sub.mkdir(parents=True)
    (sub / "sub-01_eeg.edf").write_bytes(b"")
    result = lazy_check_dataset(root)
    with pytest.raises(RuntimeError):
        result.raise_if_strict()
