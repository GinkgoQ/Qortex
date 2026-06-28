"""Label and leakage check domain.

Validates ML safety: label existence, label completeness, split group integrity,
confound association reporting, and train/test boundary violations.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qortex.checks._base import BaseChecker
from qortex.checks._report import (
    CheckFinding,
    CheckReport,
    CheckSeverity,
    EvidenceRecord,
    EvidenceState,
    SuggestedFix,
)

# Minimum fraction of subjects that must have a label for the target to be usable
_MIN_LABEL_COVERAGE = 0.5

# Maximum fraction of null labels in events.tsv before we BLOCK
_MAX_NULL_TRIAL_TYPE_FRACTION = 0.5


class LeakageChecker(BaseChecker):
    """Validate label availability and train/test leakage risk."""

    name = "leakage"
    required_for = frozenset({"train"})

    def __init__(
        self,
        *,
        target: str | None = None,
        split_unit: str = "subject",
        confound_columns: list[str] | None = None,
    ) -> None:
        self._target = target
        self._split_unit = split_unit
        self._confound_columns = confound_columns or ["site", "scanner", "acquisition", "sex", "age"]

    def run(self, dataset_path: Path, **kwargs) -> CheckReport:
        report = CheckReport(
            name=self.name,
            scope=str(dataset_path),
            inputs={
                "dataset_path": str(dataset_path),
                "target": self._target,
                "split_unit": self._split_unit,
                "confound_columns": self._confound_columns,
            },
        )

        # 1. Check participants.tsv label coverage
        self._check_participants_labels(dataset_path, report)

        # 2. Check events.tsv trial_type coverage
        if self._target:
            self._check_events_label_coverage(dataset_path, report)

        # 3. Confound association checks
        self._check_confound_associations(dataset_path, report)

        # 4. Split-unit leakage risk
        self._check_split_unit_risk(dataset_path, report)

        return report.finalize()

    # ── Participants.tsv ──────────────────────────────────────────────────────

    def _check_participants_labels(self, dataset_path: Path, report: CheckReport) -> None:
        p = dataset_path / "participants.tsv"
        if not p.exists():
            report.add(CheckFinding(
                code="LEAKAGE.NO_PARTICIPANTS_TSV",
                severity=CheckSeverity.WARN,
                message=(
                    "participants.tsv is missing. Subject-level labels (diagnosis, group, etc.) "
                    "cannot be verified."
                ),
                path=str(dataset_path),
                evidence=[EvidenceRecord(
                    field="participants.tsv",
                    state=EvidenceState.missing,
                    observed_source=str(dataset_path),
                )],
            ))
            return

        try:
            rows = _read_tsv(p)
        except Exception as exc:
            report.add(CheckFinding(
                code="LEAKAGE.PARTICIPANTS_TSV_PARSE_FAILED",
                severity=CheckSeverity.WARN,
                message=f"Cannot parse participants.tsv: {exc}",
                path=str(p),
            ))
            return

        if not rows:
            return

        columns = list(rows[0].keys())
        target = self._target

        if target and target not in columns:
            report.add(CheckFinding(
                code="LEAKAGE.TARGET_COLUMN_MISSING",
                severity=CheckSeverity.BLOCK,
                message=(
                    f"Target column '{target}' not found in participants.tsv. "
                    f"Available columns: {columns}."
                ),
                path=str(p),
                expected=target,
                observed=columns,
                evidence=[EvidenceRecord(
                    field=f"participants.{target}",
                    state=EvidenceState.missing,
                    claimed_value=target,
                    observed_source=str(p),
                )],
                suggested_fix=SuggestedFix(
                    description=f"Add a '{target}' column to participants.tsv.",
                    field=target,
                    safe=True,
                ),
            ))
            return

        if target and target in columns:
            n_total = len(rows)
            n_labeled = sum(
                1 for r in rows
                if r.get(target, "n/a") not in ("n/a", "N/A", "", None)
            )
            coverage = n_labeled / n_total if n_total > 0 else 0.0

            report.record_evidence(EvidenceRecord(
                field=f"participants.{target}.coverage",
                state=EvidenceState.confirmed,
                observed_value={"n_total": n_total, "n_labeled": n_labeled, "coverage": coverage},
                observed_source=str(p),
            ))

            if coverage < _MIN_LABEL_COVERAGE:
                report.add(CheckFinding(
                    code="LEAKAGE.LOW_LABEL_COVERAGE",
                    severity=CheckSeverity.BLOCK,
                    message=(
                        f"Only {coverage*100:.0f}% of subjects have a '{target}' label "
                        f"({n_labeled}/{n_total}). Training will use an unrepresentative subset."
                    ),
                    path=str(p),
                    expected=f">= {_MIN_LABEL_COVERAGE*100:.0f}%",
                    observed=f"{coverage*100:.1f}%",
                    evidence=[EvidenceRecord(
                        field=f"participants.{target}.coverage",
                        state=EvidenceState.confirmed,
                        observed_value=coverage,
                        observed_source=str(p),
                    )],
                ))
            else:
                label_values = [r[target] for r in rows if r.get(target, "n/a") not in ("n/a", "N/A", "")]
                label_counts = Counter(label_values)
                report.record_evidence(EvidenceRecord(
                    field=f"participants.{target}.class_counts",
                    state=EvidenceState.confirmed,
                    observed_value=dict(label_counts),
                    observed_source=str(p),
                ))

                if len(label_counts) == 1:
                    report.add(CheckFinding(
                        code="LEAKAGE.SINGLE_CLASS",
                        severity=CheckSeverity.BLOCK,
                        message=(
                            f"Target '{target}' has only one unique class: "
                            f"'{list(label_counts.keys())[0]}'. Classification is not possible."
                        ),
                        path=str(p),
                        observed=dict(label_counts),
                    ))

    # ── Events.tsv trial_type ─────────────────────────────────────────────────

    def _check_events_label_coverage(self, dataset_path: Path, report: CheckReport) -> None:
        event_files = list(dataset_path.rglob("*_events.tsv"))
        if not event_files:
            return

        target = self._target
        total_events = 0
        labeled_events = 0

        for ef in event_files:
            try:
                rows = _read_tsv(ef)
            except Exception:
                continue
            if not rows or target not in rows[0]:
                continue
            for row in rows:
                val = row.get(target, "n/a")
                total_events += 1
                if val not in ("n/a", "N/A", "", None):
                    labeled_events += 1

        if total_events == 0:
            return

        null_frac = 1.0 - labeled_events / total_events
        report.record_evidence(EvidenceRecord(
            field=f"events.{target}.null_fraction",
            state=EvidenceState.confirmed,
            observed_value=null_frac,
            observed_source=str(dataset_path),
        ))

        if null_frac > _MAX_NULL_TRIAL_TYPE_FRACTION:
            report.add(CheckFinding(
                code="LEAKAGE.HIGH_NULL_TRIAL_TYPE",
                severity=CheckSeverity.WARN,
                message=(
                    f"{null_frac*100:.1f}% of events have no '{target}' label. "
                    "Epoch-level training will exclude most events."
                ),
                path=str(dataset_path),
                expected=f"<= {_MAX_NULL_TRIAL_TYPE_FRACTION*100:.0f}% null",
                observed=f"{null_frac*100:.1f}% null",
            ))

    # ── Confound association ──────────────────────────────────────────────────

    def _check_confound_associations(self, dataset_path: Path, report: CheckReport) -> None:
        p = dataset_path / "participants.tsv"
        if not p.exists():
            return

        try:
            rows = _read_tsv(p)
        except Exception:
            return

        if not rows or not self._target:
            return

        columns = list(rows[0].keys())
        target = self._target
        if target not in columns:
            return

        target_values = [r.get(target, "n/a") for r in rows]
        target_unique = set(v for v in target_values if v not in ("n/a", "N/A", ""))

        for confound_col in self._confound_columns:
            if confound_col not in columns:
                continue

            conf_values = [r.get(confound_col, "n/a") for r in rows]
            conf_unique = set(v for v in conf_values if v not in ("n/a", "N/A", ""))
            if len(conf_unique) < 2:
                continue

            # Compute Cramér's V for categorical × categorical association
            association = _cramers_v(target_values, conf_values)
            if association is None:
                continue

            severity = CheckSeverity.INFO
            if association > 0.5:
                severity = CheckSeverity.WARN
            if association > 0.7:
                severity = CheckSeverity.BLOCK

            if association > 0.3:
                report.add(CheckFinding(
                    code="LEAKAGE.CONFOUND_ASSOCIATION",
                    severity=severity,
                    message=(
                        f"'{confound_col}' is associated with target '{target}' "
                        f"(Cramér's V = {association:.2f}). Model evaluation may be confounded."
                    ),
                    path=str(p),
                    observed={"cramers_v": association, "confound": confound_col, "target": target},
                    evidence=[EvidenceRecord(
                        field=f"confound.{confound_col}",
                        state=EvidenceState.inferred,
                        observed_value=association,
                        observed_source=str(p),
                        note="Cramér's V computed on participants.tsv categorical columns.",
                    )],
                    suggested_fix=SuggestedFix(
                        description=(
                            f"Control for '{confound_col}' as a covariate or use site-stratified splits."
                        ),
                        safe=True,
                    ),
                ))

    # ── Split-unit risk ───────────────────────────────────────────────────────

    def _check_split_unit_risk(self, dataset_path: Path, report: CheckReport) -> None:
        split_unit = self._split_unit

        if split_unit == "subject":
            # Verify that multiple sessions / runs don't cross split boundaries implicitly
            subjects = sorted(d.name for d in dataset_path.iterdir()
                              if d.is_dir() and d.name.startswith("sub-"))
            multi_session = []
            for sub in subjects:
                sessions = list((dataset_path / sub).glob("ses-*/"))
                if len(sessions) > 1:
                    multi_session.append(sub)

            if multi_session:
                report.add(CheckFinding(
                    code="LEAKAGE.MULTI_SESSION_SPLIT_RISK",
                    severity=CheckSeverity.WARN,
                    message=(
                        f"{len(multi_session)} subjects have multiple sessions. "
                        "Ensure the split groups all sessions of a subject together. "
                        "Window-level or session-level splits will cause data leakage."
                    ),
                    path=str(dataset_path),
                    observed=multi_session[:10],
                    evidence=[EvidenceRecord(
                        field="multi_session_subjects",
                        state=EvidenceState.confirmed,
                        observed_value=len(multi_session),
                        observed_source=str(dataset_path),
                    )],
                    suggested_fix=SuggestedFix(
                        description=(
                            "Use GroupKFold or LeaveOneGroupOut with subject IDs as groups. "
                            "Run: qortex check leakage --split-unit subject"
                        ),
                        command=f"qortex check leakage {dataset_path} --split-unit subject",
                        safe=True,
                    ),
                ))


# ── Statistics helpers ────────────────────────────────────────────────────────

def _cramers_v(x: list[str], y: list[str]) -> float | None:
    """Cramér's V for two categorical sequences (ignores n/a values)."""
    pairs = [
        (a, b) for a, b in zip(x, y)
        if a not in ("n/a", "N/A", "") and b not in ("n/a", "N/A", "")
    ]
    if not pairs:
        return None
    n = len(pairs)
    if n < 5:
        return None

    unique_x = sorted({p[0] for p in pairs})
    unique_y = sorted({p[1] for p in pairs})
    r, c = len(unique_x), len(unique_y)
    if r < 2 or c < 2:
        return None

    xi = {v: i for i, v in enumerate(unique_x)}
    yi = {v: i for i, v in enumerate(unique_y)}
    table = [[0] * c for _ in range(r)]
    for a, b in pairs:
        table[xi[a]][yi[b]] += 1

    chi2 = _chi2_stat(table, n, r, c)
    if chi2 < 0:
        return None

    phi2 = chi2 / n
    k = min(r, c)
    v2 = max(0.0, phi2 - (k - 1) / (n - 1))
    denom = (r - (r - 1) / (n - 1)) * (c - (c - 1) / (n - 1)) - 1
    if denom <= 0:
        return None
    return math.sqrt(v2 / ((min(r, c) - 1) if min(r, c) > 1 else 1))


def _chi2_stat(table: list[list[int]], n: int, r: int, c: int) -> float:
    row_totals = [sum(table[i]) for i in range(r)]
    col_totals = [sum(table[i][j] for i in range(r)) for j in range(c)]
    chi2 = 0.0
    for i in range(r):
        for j in range(c):
            expected = row_totals[i] * col_totals[j] / n
            if expected > 0:
                chi2 += (table[i][j] - expected) ** 2 / expected
    return chi2


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
