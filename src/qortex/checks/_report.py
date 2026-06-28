"""Core report structures for Qortex's check system.

Evidence-based, goal-aware, machine-readable.  Every check returns a CheckReport;
every preflight returns a PreflightReport that aggregates multiple CheckReports.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CheckSeverity(str, Enum):
    PASS = "PASS"
    INFO = "INFO"
    WARN = "WARN"
    BLOCK = "BLOCK"
    UNKNOWN = "UNKNOWN"


class EvidenceState(str, Enum):
    """7-state evidence model covering all knowable epistemic positions."""
    confirmed = "confirmed"       # directly read from a reliable source
    inferred = "inferred"         # derived from deterministic computation
    claimed = "claimed"           # declared by metadata, not independently verified
    missing = "missing"           # required evidence is absent
    contradicted = "contradicted" # two evidence sources disagree
    unknown = "unknown"           # not knowable without more data or user input
    blocked = "blocked"           # prerequisite evidence is invalid; cannot continue


@dataclass(frozen=True)
class EvidenceRecord:
    """One piece of evidence supporting or refuting a check conclusion."""
    field: str
    state: EvidenceState
    claimed_value: Any = None
    observed_value: Any = None
    claimed_source: str | None = None
    observed_source: str | None = None
    note: str | None = None

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "state": self.state.value,
            "claimed_value": self.claimed_value,
            "observed_value": self.observed_value,
            "claimed_source": self.claimed_source,
            "observed_source": self.observed_source,
            "note": self.note,
        }


@dataclass(frozen=True)
class SuggestedFix:
    """Actionable repair hint returned alongside a finding."""
    description: str
    command: str | None = None        # CLI command to run
    field: str | None = None          # JSON/TSV field to update
    safe: bool = True                 # True = write-safe (new file, patch); False = requires user approval
    reversible: bool = True

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "command": self.command,
            "field": self.field,
            "safe": self.safe,
            "reversible": self.reversible,
        }


@dataclass(frozen=True)
class CheckFinding:
    """One finding from a check — may be a blocker, warning, info, or unknown."""
    code: str
    severity: CheckSeverity
    message: str
    path: str | None = None
    bids_entities: dict[str, str] = field(default_factory=dict)
    expected: Any = None
    observed: Any = None
    evidence: list[EvidenceRecord] = field(default_factory=list)
    suggested_fix: SuggestedFix | None = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "path": self.path,
            "bids_entities": self.bids_entities,
            "expected": self.expected,
            "observed": self.observed,
            "evidence": [e.to_dict() for e in self.evidence],
            "suggested_fix": self.suggested_fix.to_dict() if self.suggested_fix else None,
        }


@dataclass
class CheckReport:
    """Structured output from a single targeted check.

    Immutable after construction except for the ``status`` field which is
    derived from the accumulated findings when ``finalize()`` is called.
    """
    name: str
    scope: str
    checked_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    inputs: dict[str, Any] = field(default_factory=dict)

    # Accumulated findings by severity bucket
    blockers: list[CheckFinding] = field(default_factory=list)
    warnings: list[CheckFinding] = field(default_factory=list)
    infos: list[CheckFinding] = field(default_factory=list)
    unknowns: list[CheckFinding] = field(default_factory=list)

    # Aggregate evidence map (field → EvidenceRecord)
    evidence: dict[str, EvidenceRecord] = field(default_factory=dict)

    affected_files: list[str] = field(default_factory=list)
    affected_subjects: list[str] = field(default_factory=list)
    suggested_fixes: list[SuggestedFix] = field(default_factory=list)

    # Derived — call finalize() to compute
    status: CheckSeverity = CheckSeverity.UNKNOWN

    def add(self, finding: CheckFinding) -> None:
        bucket = {
            CheckSeverity.BLOCK: self.blockers,
            CheckSeverity.WARN: self.warnings,
            CheckSeverity.INFO: self.infos,
            CheckSeverity.UNKNOWN: self.unknowns,
            CheckSeverity.PASS: [],  # PASS findings are not stored individually
        }[finding.severity]
        bucket.append(finding)
        if finding.path and finding.path not in self.affected_files:
            self.affected_files.append(finding.path)
        sub = finding.bids_entities.get("subject")
        if sub and sub not in self.affected_subjects:
            self.affected_subjects.append(sub)
        if finding.suggested_fix:
            self.suggested_fixes.append(finding.suggested_fix)

    def record_evidence(self, record: EvidenceRecord) -> None:
        self.evidence[record.field] = record

    def finalize(self) -> "CheckReport":
        if self.blockers:
            self.status = CheckSeverity.BLOCK
        elif self.warnings:
            self.status = CheckSeverity.WARN
        elif self.unknowns and not self.infos:
            self.status = CheckSeverity.UNKNOWN
        elif self.infos or self.unknowns:
            self.status = CheckSeverity.INFO
        else:
            self.status = CheckSeverity.PASS
        return self

    @property
    def all_findings(self) -> list[CheckFinding]:
        return self.blockers + self.warnings + self.infos + self.unknowns

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "scope": self.scope,
            "status": self.status.value,
            "checked_at": self.checked_at.isoformat(),
            "inputs": self.inputs,
            "blockers": [f.to_dict() for f in self.blockers],
            "warnings": [f.to_dict() for f in self.warnings],
            "infos": [f.to_dict() for f in self.infos],
            "unknowns": [f.to_dict() for f in self.unknowns],
            "evidence": {k: v.to_dict() for k, v in self.evidence.items()},
            "affected_files": self.affected_files,
            "affected_subjects": self.affected_subjects,
            "suggested_fixes": [s.to_dict() for s in self.suggested_fixes],
        }


@dataclass
class PreflightReport:
    """Aggregate report combining multiple CheckReports for a workflow goal.

    Status is the worst severity across all constituent reports.
    """
    goal: str
    dataset_path: str
    modality: str | None
    target: str | None
    split_unit: str | None
    checked_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    checks: list[CheckReport] = field(default_factory=list)
    status: CheckSeverity = CheckSeverity.UNKNOWN

    def add_check(self, report: CheckReport) -> None:
        self.checks.append(report)

    def finalize(self) -> "PreflightReport":
        severity_rank = {
            CheckSeverity.BLOCK: 4,
            CheckSeverity.WARN: 3,
            CheckSeverity.INFO: 2,
            CheckSeverity.UNKNOWN: 1,
            CheckSeverity.PASS: 0,
        }
        if not self.checks:
            self.status = CheckSeverity.UNKNOWN
            return self
        worst = max(self.checks, key=lambda r: severity_rank[r.status])
        self.status = worst.status
        return self

    @property
    def blockers(self) -> list[CheckFinding]:
        return [f for r in self.checks for f in r.blockers]

    @property
    def warnings(self) -> list[CheckFinding]:
        return [f for r in self.checks for f in r.warnings]

    @property
    def affected_files(self) -> list[str]:
        seen: set[str] = set()
        out = []
        for r in self.checks:
            for p in r.affected_files:
                if p not in seen:
                    seen.add(p)
                    out.append(p)
        return out

    @property
    def affected_subjects(self) -> list[str]:
        seen: set[str] = set()
        out = []
        for r in self.checks:
            for s in r.affected_subjects:
                if s not in seen:
                    seen.add(s)
                    out.append(s)
        return out

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "dataset_path": self.dataset_path,
            "modality": self.modality,
            "target": self.target,
            "split_unit": self.split_unit,
            "status": self.status.value,
            "checked_at": self.checked_at.isoformat(),
            "n_checks": len(self.checks),
            "checks": [c.to_dict() for c in self.checks],
        }
