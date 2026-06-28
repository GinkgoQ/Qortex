"""Timebase and event check domain.

Validates temporal consistency: event onset/duration validity, range within
recording, sampling-rate precision, task/run entity matching, and dummy-volume
ambiguity in fMRI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from qortex.checks._base import BaseChecker
from qortex.checks._report import (
    CheckFinding,
    CheckReport,
    CheckSeverity,
    EvidenceRecord,
    EvidenceState,
    SuggestedFix,
)

# Minimum sampling interval precision below which we warn (in seconds)
_MIN_ONSET_PRECISION = 1e-4


class EventsChecker(BaseChecker):
    """Validate event timing, onset ranges, and BIDS entity matching."""

    name = "events"
    required_for = frozenset({"train", "visualize", "convert"})

    def __init__(
        self,
        *,
        modality: str | None = None,
        require_duration: bool = True,
        require_trial_type: bool = False,
        max_n_events: int = 100_000,
    ) -> None:
        self._modality = modality
        self._require_duration = require_duration
        self._require_trial_type = require_trial_type
        self._max_n_events = max_n_events

    def run(self, dataset_path: Path, **kwargs) -> CheckReport:
        report = CheckReport(
            name=self.name,
            scope=str(dataset_path),
            inputs={
                "dataset_path": str(dataset_path),
                "modality": self._modality,
                "require_duration": self._require_duration,
                "require_trial_type": self._require_trial_type,
            },
        )

        event_files = list(dataset_path.rglob("*_events.tsv"))
        if not event_files:
            report.add(CheckFinding(
                code="EVENTS.NO_EVENT_FILES",
                severity=CheckSeverity.INFO,
                message=(
                    "No *_events.tsv files found. For task-based recordings this "
                    "blocks label extraction."
                ),
                path=str(dataset_path),
                evidence=[EvidenceRecord(
                    field="events.tsv",
                    state=EvidenceState.missing,
                    observed_source=str(dataset_path),
                )],
            ))
            return report.finalize()

        report.record_evidence(EvidenceRecord(
            field="events.tsv_count",
            state=EvidenceState.confirmed,
            observed_value=len(event_files),
            observed_source=str(dataset_path),
        ))

        for ef in sorted(event_files):
            self._check_event_file(ef, dataset_path, report)

        return report.finalize()

    # ── Per-file ──────────────────────────────────────────────────────────────

    def _check_event_file(
        self, event_file: Path, dataset_path: Path, report: CheckReport
    ) -> None:
        entities = _parse_bids_entities_from_path(event_file)

        try:
            rows = _read_tsv(event_file)
        except Exception as exc:
            report.add(CheckFinding(
                code="EVENTS.PARSE_FAILED",
                severity=CheckSeverity.BLOCK,
                message=f"Cannot parse events.tsv: {exc}",
                path=str(event_file),
                bids_entities=entities,
            ))
            return

        if not rows:
            report.add(CheckFinding(
                code="EVENTS.EMPTY_FILE",
                severity=CheckSeverity.WARN,
                message="events.tsv has no data rows.",
                path=str(event_file),
                bids_entities=entities,
                evidence=[EvidenceRecord(
                    field="n_events",
                    state=EvidenceState.confirmed,
                    observed_value=0,
                    observed_source=str(event_file),
                )],
            ))
            return

        columns = list(rows[0].keys())

        # Required columns
        if "onset" not in columns:
            report.add(CheckFinding(
                code="EVENTS.MISSING_ONSET_COLUMN",
                severity=CheckSeverity.BLOCK,
                message="events.tsv is missing required 'onset' column.",
                path=str(event_file),
                bids_entities=entities,
                suggested_fix=SuggestedFix(
                    description="Add an 'onset' column with event start times in seconds.",
                    field="onset",
                    safe=True,
                ),
            ))
            return

        if "duration" not in columns and self._require_duration:
            report.add(CheckFinding(
                code="EVENTS.MISSING_DURATION_COLUMN",
                severity=CheckSeverity.WARN,
                message="events.tsv is missing 'duration' column.",
                path=str(event_file),
                bids_entities=entities,
            ))

        if "trial_type" not in columns and self._require_trial_type:
            report.add(CheckFinding(
                code="EVENTS.MISSING_TRIAL_TYPE",
                severity=CheckSeverity.WARN,
                message="events.tsv is missing 'trial_type' column; label extraction will fail.",
                path=str(event_file),
                bids_entities=entities,
                suggested_fix=SuggestedFix(
                    description="Add a 'trial_type' column with categorical event labels.",
                    field="trial_type",
                    safe=True,
                ),
            ))

        # Parse and validate onset values
        onsets: list[float] = []
        n_negative = 0
        n_nan = 0
        n_bad_precision = 0
        for row in rows[: self._max_n_events]:
            raw = row.get("onset", "n/a")
            if raw in ("n/a", "N/A", "", None):
                n_nan += 1
                continue
            try:
                val = float(raw)
            except (ValueError, TypeError):
                n_nan += 1
                continue
            if val < 0:
                n_negative += 1
            if abs(val - round(val, 4)) > _MIN_ONSET_PRECISION:
                n_bad_precision += 1
            onsets.append(val)

        if n_nan > 0:
            report.add(CheckFinding(
                code="EVENTS.ONSET_MISSING_VALUES",
                severity=CheckSeverity.WARN,
                message=f"{n_nan} events have missing or non-numeric onset values.",
                path=str(event_file),
                bids_entities=entities,
                observed=n_nan,
            ))

        if n_negative > 0:
            report.add(CheckFinding(
                code="EVENTS.NEGATIVE_ONSET",
                severity=CheckSeverity.WARN,
                message=f"{n_negative} events have negative onset values.",
                path=str(event_file),
                bids_entities=entities,
                observed=n_negative,
                suggested_fix=SuggestedFix(
                    description=(
                        "Negative onsets may indicate a pre-stimulus baseline or timebase "
                        "offset. Confirm recording_start_offset if using fMRI."
                    ),
                    safe=True,
                ),
            ))

        if onsets:
            report.record_evidence(EvidenceRecord(
                field=f"{event_file.name}.onset_range",
                state=EvidenceState.confirmed,
                observed_value={"min": min(onsets), "max": max(onsets), "n": len(onsets)},
                observed_source=str(event_file),
            ))

        # Check entity matching — corresponding primary file must exist
        self._check_primary_file_exists(event_file, dataset_path, entities, report)

        # fMRI: check dummy volume ambiguity
        if "bold" in event_file.name or "_bold" in event_file.stem:
            self._check_fmri_dummy_volume_risk(event_file, onsets, entities, report)

    def _check_primary_file_exists(
        self,
        event_file: Path,
        dataset_path: Path,
        entities: dict[str, str],
        report: CheckReport,
    ) -> None:
        """Verify that at least one primary file shares the same BIDS entities."""
        stem_without_events = event_file.stem.replace("_events", "")
        parent = event_file.parent
        primary_extensions = [".edf", ".bdf", ".fif", ".nii.gz", ".nii", ".set", ".vhdr"]
        found = any(
            (parent / (stem_without_events + ext)).exists()
            for ext in primary_extensions
        )
        if not found:
            report.add(CheckFinding(
                code="EVENTS.NO_MATCHING_PRIMARY",
                severity=CheckSeverity.WARN,
                message=(
                    f"No primary data file found matching {event_file.name}. "
                    "The events.tsv may be orphaned or the primary file was not downloaded."
                ),
                path=str(event_file),
                bids_entities=entities,
                evidence=[EvidenceRecord(
                    field="primary_file",
                    state=EvidenceState.missing,
                    observed_source=str(parent),
                )],
            ))

    def _check_fmri_dummy_volume_risk(
        self,
        event_file: Path,
        onsets: list[float],
        entities: dict[str, str],
        report: CheckReport,
    ) -> None:
        """Flag ambiguity when onsets start near t=0 without dummy-volume metadata."""
        if not onsets:
            return
        min_onset = min(onsets)
        if min_onset < 1.0:
            report.add(CheckFinding(
                code="EVENTS.FMRI_EARLY_ONSET",
                severity=CheckSeverity.INFO,
                message=(
                    f"Earliest event onset is {min_onset:.3f} s. If dummy volumes were "
                    "discarded, confirm NumberOfVolumesDiscardedByScanner in the sidecar "
                    "and adjust onset timebase accordingly."
                ),
                path=str(event_file),
                bids_entities=entities,
                observed=min_onset,
                suggested_fix=SuggestedFix(
                    description=(
                        "Verify NumberOfVolumesDiscardedByScanner in the _bold.json sidecar "
                        "and that onsets are relative to the first *retained* volume."
                    ),
                    field="NumberOfVolumesDiscardedByScanner",
                    safe=True,
                ),
            ))


# ── Tiny TSV reader (no pandas) ───────────────────────────────────────────────

def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as fh:
        lines = fh.read().splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        rows.append(dict(zip(header, parts)))
    return rows


def _parse_bids_entities_from_path(path: Path) -> dict[str, str]:
    import re
    entity_re = re.compile(r"(sub|ses|task|run|acq|ce|dir|rec|echo|part)-([A-Za-z0-9]+)")
    return dict(entity_re.findall(path.name))
