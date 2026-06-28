"""Structure check domain.

Validates dataset layout, BIDS entity consistency, companion-file closure,
raw/derivative separation, and local copy completeness.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from qortex.checks._base import BaseChecker
from qortex.checks._report import (
    CheckFinding,
    CheckReport,
    CheckSeverity,
    EvidenceRecord,
    EvidenceState,
    SuggestedFix,
)

# BIDS entity regex — matches subject, session, task, run, acquisition, etc.
_BIDS_ENTITY_RE = re.compile(
    r"(sub-(?P<subject>[A-Za-z0-9]+))"
    r"(_ses-(?P<session>[A-Za-z0-9]+))?"
    r"(_task-(?P<task>[A-Za-z0-9]+))?"
    r"(_run-(?P<run>[0-9]+))?"
    r"(_acq-(?P<acq>[A-Za-z0-9]+))?"
)

_SIGNAL_EXTS = {".edf", ".bdf", ".fif", ".set", ".vhdr", ".eeg"}
_VOLUME_EXTS = {".nii", ".nii.gz"}
_SIDECAR_EXTS = {".json"}
_EVENT_SUFFIX = "_events.tsv"
_CHANNEL_SUFFIX = "_channels.tsv"

# Modality → expected companion suffixes
_REQUIRED_COMPANIONS: dict[str, list[str]] = {
    "eeg": ["_eeg.json", "_channels.tsv"],
    "meg": ["_meg.json", "_channels.tsv"],
    "ieeg": ["_ieeg.json", "_channels.tsv"],
    "bold": ["_bold.json"],
    "dwi": ["_dwi.json", ".bval", ".bvec"],
    "T1w": ["_T1w.json"],
    "fnirs": ["_fnirs.json", "_channels.tsv"],
}


class StructureChecker(BaseChecker):
    """Validate BIDS dataset layout and file relationships."""

    name = "structure"
    required_for = frozenset({"visualize", "convert", "train", "neuroai-run"})

    def __init__(
        self,
        *,
        modality: str | None = None,
        check_participants_tsv: bool = True,
        check_companion_closure: bool = True,
        check_raw_derivative_separation: bool = True,
    ) -> None:
        self._modality = modality
        self._check_participants = check_participants_tsv
        self._check_closure = check_companion_closure
        self._check_separation = check_raw_derivative_separation

    def run(self, dataset_path: Path, **kwargs) -> CheckReport:
        report = CheckReport(
            name=self.name,
            scope=str(dataset_path),
            inputs={"dataset_path": str(dataset_path), "modality": self._modality},
        )

        if not dataset_path.exists():
            report.add(CheckFinding(
                code="STRUCTURE.PATH_NOT_FOUND",
                severity=CheckSeverity.BLOCK,
                message=f"Dataset path does not exist: {dataset_path}",
                path=str(dataset_path),
                suggested_fix=SuggestedFix(
                    description="Verify the dataset path or download the dataset first.",
                    command=f"qortex download <dataset_id> --output {dataset_path.parent}",
                    safe=True,
                ),
            ))
            return report.finalize()

        dataset_description = dataset_path / "dataset_description.json"
        if not dataset_description.exists():
            report.add(CheckFinding(
                code="STRUCTURE.NO_DATASET_DESCRIPTION",
                severity=CheckSeverity.WARN,
                message="dataset_description.json is missing; BIDS compliance cannot be confirmed.",
                path=str(dataset_path),
                evidence=[EvidenceRecord(
                    field="dataset_description.json",
                    state=EvidenceState.missing,
                    observed_source=str(dataset_path),
                )],
            ))
        else:
            try:
                desc = json.loads(dataset_description.read_text())
                bids_version = desc.get("BIDSVersion", "unknown")
                report.record_evidence(EvidenceRecord(
                    field="BIDSVersion",
                    state=EvidenceState.confirmed,
                    observed_value=bids_version,
                    observed_source="dataset_description.json",
                ))
            except (json.JSONDecodeError, OSError) as exc:
                report.add(CheckFinding(
                    code="STRUCTURE.DATASET_DESCRIPTION_INVALID",
                    severity=CheckSeverity.WARN,
                    message=f"dataset_description.json is malformed: {exc}",
                    path=str(dataset_description),
                ))

        if self._check_participants:
            self._check_participants_tsv(dataset_path, report)

        subjects = self._discover_subjects(dataset_path)
        if not subjects:
            report.add(CheckFinding(
                code="STRUCTURE.NO_SUBJECTS",
                severity=CheckSeverity.BLOCK,
                message="No sub-* directories found; dataset appears empty.",
                path=str(dataset_path),
            ))
            return report.finalize()

        report.record_evidence(EvidenceRecord(
            field="subjects",
            state=EvidenceState.confirmed,
            observed_value=sorted(subjects),
            observed_source=str(dataset_path),
        ))

        if self._check_separation:
            self._check_raw_derivative_separation(dataset_path, report)

        if self._check_closure:
            self._check_companion_file_closure(dataset_path, subjects, report)

        self._check_entity_consistency(dataset_path, subjects, report)

        return report.finalize()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_participants_tsv(self, dataset_path: Path, report: CheckReport) -> None:
        p = dataset_path / "participants.tsv"
        if not p.exists():
            report.add(CheckFinding(
                code="STRUCTURE.NO_PARTICIPANTS_TSV",
                severity=CheckSeverity.WARN,
                message="participants.tsv is missing; subject-level metadata unavailable.",
                path=str(dataset_path),
                evidence=[EvidenceRecord(
                    field="participants.tsv",
                    state=EvidenceState.missing,
                    observed_source=str(dataset_path),
                )],
                suggested_fix=SuggestedFix(
                    description="Add participants.tsv with at minimum a 'participant_id' column.",
                    safe=True,
                ),
            ))
        else:
            report.record_evidence(EvidenceRecord(
                field="participants.tsv",
                state=EvidenceState.confirmed,
                observed_value=True,
                observed_source=str(p),
            ))

    def _discover_subjects(self, dataset_path: Path) -> list[str]:
        return [d.name for d in sorted(dataset_path.iterdir())
                if d.is_dir() and d.name.startswith("sub-")]

    def _check_raw_derivative_separation(self, dataset_path: Path, report: CheckReport) -> None:
        derivatives_dir = dataset_path / "derivatives"
        if derivatives_dir.exists():
            report.add(CheckFinding(
                code="STRUCTURE.DERIVATIVES_PRESENT",
                severity=CheckSeverity.INFO,
                message="A 'derivatives' directory exists. Ensure raw and derivative files are not mixed.",
                path=str(derivatives_dir),
                evidence=[EvidenceRecord(
                    field="derivatives",
                    state=EvidenceState.confirmed,
                    observed_value=True,
                    observed_source=str(derivatives_dir),
                )],
            ))

        for subject_dir in dataset_path.glob("sub-*/"):
            for f in subject_dir.rglob("*"):
                if f.is_file() and "derivatives" in f.parts:
                    continue
                if f.is_file() and any(tok in f.name for tok in ["_desc-", "_space-", "_res-", "_den-"]):
                    report.add(CheckFinding(
                        code="STRUCTURE.DERIVATIVE_IN_RAW",
                        severity=CheckSeverity.WARN,
                        message=(
                            f"Possible derivative file found in raw directory: {f.name}. "
                            "Derived entities (_desc-, _space-, _res-) belong under derivatives/."
                        ),
                        path=str(f),
                        bids_entities=self._parse_bids_entities(f.name),
                        suggested_fix=SuggestedFix(
                            description="Move to derivatives/<pipeline>/ directory.",
                            safe=True,
                        ),
                    ))

    def _check_companion_file_closure(
        self, dataset_path: Path, subjects: list[str], report: CheckReport
    ) -> None:
        modality = self._modality
        if modality is None:
            return  # can only check closure when modality is known

        required_suffixes = _REQUIRED_COMPANIONS.get(modality, [])
        if not required_suffixes:
            return

        for sub in subjects:
            sub_dir = dataset_path / sub
            for primary in sub_dir.rglob("*"):
                if not primary.is_file():
                    continue
                stem = primary.stem
                if primary.suffix in _SIGNAL_EXTS or primary.suffix in _VOLUME_EXTS:
                    for suf in required_suffixes:
                        companion = primary.parent / (stem + suf)
                        if not companion.exists():
                            entities = self._parse_bids_entities(primary.name)
                            report.add(CheckFinding(
                                code="STRUCTURE.MISSING_COMPANION",
                                severity=CheckSeverity.WARN,
                                message=f"Missing companion {suf} for {primary.name}",
                                path=str(primary),
                                bids_entities=entities,
                                evidence=[EvidenceRecord(
                                    field=suf,
                                    state=EvidenceState.missing,
                                    observed_source=str(primary.parent),
                                )],
                                suggested_fix=SuggestedFix(
                                    description=f"Create {companion.name} with required fields.",
                                    safe=True,
                                ),
                            ))

    def _check_entity_consistency(
        self, dataset_path: Path, subjects: list[str], report: CheckReport
    ) -> None:
        """Check that BIDS entities are consistent across subject directories."""
        tasks_seen: dict[str, set[str]] = {}
        runs_seen: dict[str, set[str]] = {}

        for sub in subjects:
            sub_dir = dataset_path / sub
            for f in sub_dir.rglob("*"):
                if not f.is_file():
                    continue
                entities = self._parse_bids_entities(f.name)
                task = entities.get("task")
                run = entities.get("run")
                if task:
                    tasks_seen.setdefault(sub, set()).add(task)
                if run:
                    runs_seen.setdefault(sub, set()).add(run)

        all_task_sets = [frozenset(v) for v in tasks_seen.values()]
        if all_task_sets and len(set(all_task_sets)) > 1:
            report.add(CheckFinding(
                code="STRUCTURE.INCONSISTENT_TASKS",
                severity=CheckSeverity.INFO,
                message=(
                    "Subjects have different task sets. This may be expected (e.g., multi-session "
                    "protocols) or a missing-file error."
                ),
                evidence=[EvidenceRecord(
                    field="task_sets",
                    state=EvidenceState.inferred,
                    observed_value={sub: sorted(tasks) for sub, tasks in tasks_seen.items()},
                    observed_source=str(dataset_path),
                )],
            ))

    @staticmethod
    def _parse_bids_entities(filename: str) -> dict[str, str]:
        m = _BIDS_ENTITY_RE.match(filename)
        if not m:
            return {}
        return {k: v for k, v in m.groupdict().items() if v is not None}
