"""QC-based subject filtering — chainable DSL over imaging quality metrics.

Architecture
------------
QCFilter accumulates rule groups.  Each rule group is bound to one data source
(a Polars DataFrame of quality metrics).  When ``apply()`` is called:

  1.  Each rule group's DataFrame is joined with its active rules.
  2.  A per-(subject, session, run) verdict is computed.
  3.  Results are merged across rule groups: a subject must pass ALL groups.
  4.  A QCMask is returned with structured per-subject pass/fail records.

Operators supported in ``require()``:
  ``">"`` | ``">="`` | ``"<"`` | ``"<="`` | ``"=="`` | ``"!="``

The filtering is pure-Polars; no pandas or numpy dependency at call time.
All DataFrame construction is deferred until ``apply()`` is invoked.
"""

from __future__ import annotations

import logging
import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

log = logging.getLogger(__name__)

_OP_MAP: dict[str, Callable[[Any, Any], bool]] = {
    ">":  operator.gt,
    ">=": operator.ge,
    "<":  operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}

OperatorStr = Literal[">", ">=", "<", "<=", "==", "!="]

# MRIQC IQM column groupings for convenience methods
_MRIQC_T1W_DEFAULTS: list[tuple[str, OperatorStr, float]] = [
    ("snr_total",  ">",  10.0),
    ("cnr",        ">",  1.0),
    ("qi_1",       "<",  0.1),
    ("cjv",        "<",  0.7),
    ("efc",        "<",  0.6),
    ("fber",       ">",  500.0),
    ("wm2max",     "<",  0.8),
]

_MRIQC_BOLD_DEFAULTS: list[tuple[str, OperatorStr, float]] = [
    ("fd_mean",  "<", 0.5),
    ("aor",      "<", 0.2),
    ("aqi",      "<", 0.05),
    ("dvars_nstd", "<", 100.0),
    ("gcor",     "<", 0.8),
    ("tsnr",     ">", 20.0),
]


@dataclass
class QCRule:
    """One threshold rule on a named metric column."""

    column: str
    op_str: OperatorStr
    threshold: float
    description: str = ""

    @property
    def op_fn(self) -> Callable[[Any, Any], bool]:
        return _OP_MAP[self.op_str]

    def evaluate(self, value: Any) -> bool:
        if value is None:
            return False
        try:
            return self.op_fn(float(value), self.threshold)
        except (TypeError, ValueError):
            return False

    def describe(self) -> str:
        label = self.description or self.column
        return f"{label} {self.op_str} {self.threshold}"


@dataclass
class QCViolation:
    """One failing rule for one subject/session/run."""

    subject: str
    session: str | None
    run: str | None
    rule: str
    observed_value: Any
    threshold: float
    op_str: str


@dataclass
class _RuleGroup:
    """A DataFrame source bound to a set of rules."""

    source_name: str                    # human label e.g. "mriqc_T1w"
    dataframe: Any                      # polars.DataFrame — held loosely
    subject_col: str
    session_col: str | None
    run_col: str | None
    task_col: str | None
    rules: list[QCRule] = field(default_factory=list)


class QCMask:
    """Result of ``QCFilter.apply()``.

    Provides structured per-subject pass/fail records plus convenience
    attributes for downstream consumption.
    """

    def __init__(
        self,
        passing: list[str],
        excluded: dict[str, list[str]],
        violations: list[QCViolation],
        per_subject_rows: list[dict[str, Any]],
    ) -> None:
        self.passing_subjects = sorted(passing)
        self.excluded_subjects = excluded          # sub → list of violation descriptions
        self.violations = violations
        self._per_subject_rows = per_subject_rows

    @property
    def n_passing(self) -> int:
        return len(self.passing_subjects)

    @property
    def n_excluded(self) -> int:
        return len(self.excluded_subjects)

    @property
    def pass_rate(self) -> float:
        total = self.n_passing + self.n_excluded
        return self.n_passing / total if total > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"QC Mask — {self.n_passing} passing / {self.n_excluded} excluded "
            f"({self.pass_rate * 100:.1f}% pass rate)",
        ]
        if self.excluded_subjects:
            lines.append("Excluded:")
            for sub in sorted(self.excluded_subjects)[:20]:
                reasons = self.excluded_subjects[sub]
                lines.append(f"  {sub}: {'; '.join(reasons)}")
            remaining = len(self.excluded_subjects) - 20
            if remaining > 0:
                lines.append(f"  ... {remaining} more excluded subjects")
        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        """Return a Polars DataFrame with one row per subject and QC columns."""
        import polars as pl
        if not self._per_subject_rows:
            return pl.DataFrame()
        return pl.DataFrame(self._per_subject_rows)

    def filter_manifest(self, manifest) -> Any:
        """Return a new Manifest with failing subjects removed.

        Parameters
        ----------
        manifest:
            A ``qortex.core.entities.Manifest`` object.

        Returns
        -------
        Manifest
            Copy of the manifest with all files belonging to excluded subjects
            filtered out.
        """
        from qortex.core.entities import Manifest

        passing_set = {s.removeprefix("sub-") for s in self.passing_subjects}
        filtered_files = [
            f for f in manifest.files
            if f.entities.subject is None or f.entities.subject in passing_set
        ]
        new = Manifest(
            dataset_id=manifest.dataset_id,
            snapshot=manifest.snapshot,
            doi=manifest.doi,
            files=filtered_files,
            summary=manifest.summary,
            built_at=manifest.built_at,
        )
        new.rebuild_index()
        return new

    def exclude_subjects_from(self, subject_list: list[str]) -> list[str]:
        """Return the input list with excluded subjects removed."""
        excluded_set = set(self.excluded_subjects)
        return [
            s for s in subject_list
            if s not in excluded_set
            and (f"sub-{s}" not in excluded_set)
        ]


class QCFilter:
    """Chainable, source-agnostic QC filter for subject exclusion.

    Parameters
    ----------
    derivative_index:
        A ``DerivativeIndex`` from ``qortex.derivatives``.  Optional when
        using ``from_dataframe()`` directly.
    """

    def __init__(self, derivative_index: Any | None = None) -> None:
        self._index = derivative_index
        self._rule_groups: list[_RuleGroup] = []
        self._manual_exclude: set[str] = set()
        self._manual_include: set[str] = set()
        self._current_group: _RuleGroup | None = None

    # ── Source methods ────────────────────────────────────────────────────

    def mriqc_T1w(
        self,
        *,
        apply_defaults: bool = True,
    ) -> "QCFilter":
        """Activate MRIQC T1w IQM rules from the derivative index.

        Parameters
        ----------
        apply_defaults:
            When True (default), conservative default thresholds for
            common T1w IQMs are pre-loaded.  Individual ``require()``
            calls override or extend these.
        """
        df = self._load_mriqc_table("T1w")
        if df is None:
            log.warning("No MRIQC T1w table found. mriqc_T1w() has no effect.")
            return self
        group = _RuleGroup(
            source_name="mriqc_T1w",
            dataframe=df,
            subject_col=self._detect_subject_col(df),
            session_col=self._detect_col(df, ["session_id", "ses"]),
            run_col=self._detect_col(df, ["run_id", "run"]),
            task_col=self._detect_col(df, ["task_id", "task"]),
        )
        if apply_defaults:
            for col, op_str, thresh in _MRIQC_T1W_DEFAULTS:
                if col in self._column_names(df):
                    group.rules.append(QCRule(column=col, op_str=op_str, threshold=thresh))
        self._rule_groups.append(group)
        self._current_group = group
        return self

    def mriqc_bold(
        self,
        *,
        apply_defaults: bool = True,
    ) -> "QCFilter":
        """Activate MRIQC BOLD IQM rules from the derivative index."""
        df = self._load_mriqc_table("bold")
        if df is None:
            log.warning("No MRIQC BOLD table found. mriqc_bold() has no effect.")
            return self
        group = _RuleGroup(
            source_name="mriqc_bold",
            dataframe=df,
            subject_col=self._detect_subject_col(df),
            session_col=self._detect_col(df, ["session_id", "ses"]),
            run_col=self._detect_col(df, ["run_id", "run"]),
            task_col=self._detect_col(df, ["task_id", "task"]),
        )
        if apply_defaults:
            for col, op_str, thresh in _MRIQC_BOLD_DEFAULTS:
                if col in self._column_names(df):
                    group.rules.append(QCRule(column=col, op_str=op_str, threshold=thresh))
        self._rule_groups.append(group)
        self._current_group = group
        return self

    def fmriprep_bold(
        self,
        *,
        fd_threshold: float = 0.5,
    ) -> "QCFilter":
        """Activate fMRIPrep confound-based filtering.

        Summarises each subject's confound TSVs and applies a mean framewise
        displacement threshold.

        Parameters
        ----------
        fd_threshold:
            Exclude subjects whose mean FD across any run exceeds this value.
        """
        if self._index is None:
            raise RuntimeError(
                "fmriprep_bold() requires a DerivativeIndex. "
                "Pass one to QCFilter(derivative_index=...)."
            )
        subjects = self._index.subjects("fmriprep")
        if not subjects:
            log.warning("No fMRIPrep subjects found in derivative index.")
            return self

        import polars as pl

        rows: list[dict[str, Any]] = []
        for sub in subjects:
            summary = self._index.confound_summary(sub)
            for run_info in summary.get("runs", []):
                fd_mean = run_info.get("fd_mean")
                rows.append({
                    "participant_id": sub,
                    "task": run_info.get("task"),
                    "run": run_info.get("run"),
                    "fd_mean": fd_mean,
                    "n_volumes": run_info.get("n_volumes"),
                    "high_motion_fraction": run_info.get("high_motion_fraction"),
                })

        if not rows:
            log.warning("No confound data found for fmriprep_bold().")
            return self

        df = pl.DataFrame(rows, schema_overrides={
            "fd_mean": pl.Float64,
            "high_motion_fraction": pl.Float64,
        })
        group = _RuleGroup(
            source_name="fmriprep_bold",
            dataframe=df,
            subject_col="participant_id",
            session_col=None,
            run_col="run" if "run" in df.columns else None,
            task_col="task" if "task" in df.columns else None,
        )
        group.rules.append(QCRule(
            column="fd_mean",
            op_str="<",
            threshold=fd_threshold,
            description=f"fMRIPrep mean FD < {fd_threshold} mm",
        ))
        self._rule_groups.append(group)
        self._current_group = group
        return self

    def from_dataframe(
        self,
        df: Any,
        *,
        source_name: str = "custom",
        subject_col: str = "participant_id",
        session_col: str | None = None,
        run_col: str | None = None,
        task_col: str | None = None,
    ) -> "QCFilter":
        """Add any Polars DataFrame as a QC rule source.

        Parameters
        ----------
        df:
            Polars DataFrame with at minimum a subject column and QC metric
            columns.  Any column name can then be referenced in ``require()``.
        source_name:
            Label for this source group (used in violation messages).
        subject_col:
            Name of the column containing subject IDs.
        """
        group = _RuleGroup(
            source_name=source_name,
            dataframe=df,
            subject_col=subject_col,
            session_col=session_col,
            run_col=run_col,
            task_col=task_col,
        )
        self._rule_groups.append(group)
        self._current_group = group
        return self

    # ── Rule specification ────────────────────────────────────────────────

    def require(
        self,
        column: str,
        op_str: OperatorStr,
        threshold: float,
        *,
        description: str = "",
    ) -> "QCFilter":
        """Add a threshold rule to the most recently activated source.

        Multiple ``require()`` calls are ANDed together within the same source.

        Parameters
        ----------
        column:
            Metric column name in the source DataFrame.
        op_str:
            Comparison operator string: ``">"`` | ``">="`` | ``"<"`` |
            ``"<="`` | ``"=="`` | ``"!="``
        threshold:
            Numeric threshold value.
        description:
            Human-readable description used in violation messages.

        Raises
        ------
        RuntimeError
            When called before any source method (no active rule group).
        """
        if op_str not in _OP_MAP:
            raise ValueError(
                f"Unknown operator {op_str!r}. Must be one of {list(_OP_MAP)}"
            )
        if self._current_group is None:
            raise RuntimeError(
                "Call a source method first (e.g. mriqc_T1w(), from_dataframe()) "
                "before adding rules with require()."
            )
        self._current_group.rules.append(
            QCRule(column=column, op_str=op_str, threshold=threshold, description=description)
        )
        return self

    def exclude_subjects(self, subjects: list[str]) -> "QCFilter":
        """Unconditionally exclude specific subjects (manual denylist)."""
        for sub in subjects:
            self._manual_exclude.add(sub if sub.startswith("sub-") else f"sub-{sub}")
        return self

    def include_subjects(self, subjects: list[str]) -> "QCFilter":
        """Unconditionally pass specific subjects regardless of metric thresholds."""
        for sub in subjects:
            self._manual_include.add(sub if sub.startswith("sub-") else f"sub-{sub}")
        return self

    # ── Execution ─────────────────────────────────────────────────────────

    def apply(self) -> QCMask:
        """Evaluate all rule groups and return a QCMask.

        Each subject must satisfy ALL rule groups to be in ``passing_subjects``.
        """
        if not self._rule_groups and not self._manual_exclude:
            log.warning("No QC rules defined. Returning empty QCMask (no subjects).")
            return QCMask(passing=[], excluded={}, violations=[], per_subject_rows=[])

        # Collect all subjects across all sources
        all_subjects: set[str] = set()
        for group in self._rule_groups:
            subs = self._subjects_from_group(group)
            all_subjects.update(subs)

        # Per-subject verdict accumulation
        subject_violations: dict[str, list[str]] = {}
        all_violations: list[QCViolation] = []
        per_subject_rows: list[dict[str, Any]] = []

        for sub in sorted(all_subjects):
            subject_key = sub if sub.startswith("sub-") else f"sub-{sub}"
            row_data: dict[str, Any] = {"subject": subject_key, "passed": True}
            sub_violations: list[str] = []

            if subject_key in self._manual_include:
                row_data["passed"] = True
                per_subject_rows.append(row_data)
                continue

            if subject_key in self._manual_exclude:
                sub_violations.append("manually excluded")
                row_data["passed"] = False
                subject_violations[subject_key] = sub_violations
                row_data["violations"] = sub_violations
                per_subject_rows.append(row_data)
                continue

            for group in self._rule_groups:
                group_rows = self._rows_for_subject(group, sub)
                if not group_rows:
                    # Subject absent from this source → fail-closed
                    msg = f"missing from {group.source_name}"
                    sub_violations.append(msg)
                    row_data[f"{group.source_name}_status"] = "absent"
                    continue

                # Subject must pass ALL rules across ALL rows (e.g. all runs)
                for rule in group.rules:
                    for row in group_rows:
                        val = row.get(rule.column)
                        row_data[f"{group.source_name}_{rule.column}"] = val
                        if not rule.evaluate(val):
                            violation = QCViolation(
                                subject=subject_key,
                                session=row.get(group.session_col) if group.session_col else None,
                                run=row.get(group.run_col) if group.run_col else None,
                                rule=rule.describe(),
                                observed_value=val,
                                threshold=rule.threshold,
                                op_str=rule.op_str,
                            )
                            all_violations.append(violation)
                            sub_violations.append(
                                f"{group.source_name}: {rule.column}={val} "
                                f"violates {rule.op_str} {rule.threshold}"
                            )

            if sub_violations:
                row_data["passed"] = False
                subject_violations[subject_key] = sub_violations
            row_data["violations"] = sub_violations
            per_subject_rows.append(row_data)

        passing = [
            sub if sub.startswith("sub-") else f"sub-{sub}"
            for sub in sorted(all_subjects)
            if (sub if sub.startswith("sub-") else f"sub-{sub}") not in subject_violations
        ]
        # Manually included subjects always pass
        passing = sorted(set(passing) | (self._manual_include - set(subject_violations)))

        return QCMask(
            passing=passing,
            excluded=subject_violations,
            violations=all_violations,
            per_subject_rows=per_subject_rows,
        )

    # ── Private helpers ───────────────────────────────────────────────────

    def _load_mriqc_table(self, image_type: str) -> Any | None:
        if self._index is None:
            raise RuntimeError(
                "MRIQC methods require a DerivativeIndex. "
                "Pass one to QCFilter(derivative_index=...)."
            )
        df = self._index.qc_table("mriqc")
        if df is None or len(df) == 0:
            return None
        # Filter by image type prefix in _source_table column
        if "_source_table" in df.columns:
            import polars as pl
            pattern = image_type.lower()
            df = df.filter(pl.col("_source_table").str.to_lowercase().str.contains(pattern))
        return df if len(df) > 0 else None

    def _detect_subject_col(self, df: Any) -> str:
        for col in ("bids_name", "participant_id", "subject", "sub"):
            if col in self._column_names(df):
                return col
        cols = self._column_names(df)
        return cols[0] if cols else "participant_id"

    def _detect_col(self, df: Any, candidates: list[str]) -> str | None:
        for col in candidates:
            if col in self._column_names(df):
                return col
        return None

    def _column_names(self, df: Any) -> list[str]:
        try:
            return df.columns
        except AttributeError:
            return []

    def _subjects_from_group(self, group: _RuleGroup) -> list[str]:
        """Extract unique subject IDs from a rule group's DataFrame."""
        try:
            col = group.dataframe[group.subject_col]
            return sorted({str(v) for v in col.to_list() if v is not None})
        except Exception as exc:
            log.warning("Cannot read subjects from group %s: %s", group.source_name, exc)
            return []

    def _rows_for_subject(
        self,
        group: _RuleGroup,
        subject: str,
    ) -> list[dict[str, Any]]:
        """Return all rows in the DataFrame matching this subject."""
        import polars as pl
        try:
            sub_raw = subject.removeprefix("sub-")
            # MRIQC uses full "sub-XX" in bids_name, others use raw value
            col = group.subject_col
            try:
                filtered = group.dataframe.filter(
                    pl.col(col).cast(pl.Utf8).str.ends_with(sub_raw)
                )
            except Exception:
                filtered = group.dataframe.filter(
                    pl.col(col).cast(pl.Utf8) == subject
                )
            return filtered.to_dicts()
        except Exception as exc:
            log.debug("Row lookup failed for %s in %s: %s", subject, group.source_name, exc)
            return []
