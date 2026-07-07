"""Label Landscape Analyzer — remote event analysis without download.

This module streams all events TSV files from an OpenNeuro dataset's CDN
concurrently, analyzes trial_type distributions across all subjects and
sessions, and produces a structured report — without downloading any signal
data files.

For a dataset like ds000117 with 256 events files, each ~5KB, the total
network cost is ~1.3 MB and completes in seconds. This enables:

  * Understanding class distribution before committing to a download
  * Detecting class imbalance that would bias a classifier
  * Identifying which subjects/sessions have missing events
  * Computing inter-stimulus interval statistics for ERP/synchrony analysis
  * Validating label column consistency across all files

Key design decisions
--------------------
* Concurrent fetching via RemoteFileGateway.batch_fetch_tsv() — 24 parallel
  connections by default.
* Results are cached on the gateway for repeated analysis passes.
* All analysis is pure Polars — no pandas dependency.
* The imbalance ratio uses the max/min ratio: 1.0 = perfect balance,
  >3.0 = severe imbalance, >10.0 = critical imbalance.
* ISI (inter-stimulus interval) analysis uses onset differences to measure
  timing regularity — important for continuous decoding pipelines.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any

from qortex.client.remote import RemoteFileGateway, _pick_url
from qortex.core.entities import FileRecord, Manifest

log = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class TrialTypeStats:
    """Statistics for one trial type / event class."""
    trial_type: str
    total_count: int = 0
    subject_counts: dict[str, int] = field(default_factory=dict)
    mean_duration_s: float | None = None
    std_duration_s: float | None = None
    min_duration_s: float | None = None
    max_duration_s: float | None = None

    @property
    def n_subjects_with_this_class(self) -> int:
        return len(self.subject_counts)

    @property
    def count(self) -> int:
        """Backward-compatible alias for ``total_count``."""
        return self.total_count

    @property
    def n_subjects(self) -> int:
        """Backward-compatible alias for ``n_subjects_with_this_class``."""
        return self.n_subjects_with_this_class

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_type": self.trial_type,
            "total_count": self.total_count,
            "count": self.count,
            "n_subjects": self.n_subjects_with_this_class,
            "mean_duration_s": round(self.mean_duration_s, 3) if self.mean_duration_s is not None else None,
            "std_duration_s": round(self.std_duration_s, 3) if self.std_duration_s is not None else None,
        }


@dataclass
class SubjectEventProfile:
    """Per-subject event profile across all sessions and tasks."""
    subject: str
    n_events_files: int = 0
    trial_type_counts: dict[str, int] = field(default_factory=dict)
    total_events: int = 0
    tasks_covered: list[str] = field(default_factory=list)
    has_missing_events: bool = False

    @property
    def dominant_class(self) -> str | None:
        if not self.trial_type_counts:
            return None
        return max(self.trial_type_counts, key=self.trial_type_counts.__getitem__)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "n_events_files": self.n_events_files,
            "trial_type_counts": self.trial_type_counts,
            "total_events": self.total_events,
            "tasks_covered": self.tasks_covered,
            "dominant_class": self.dominant_class,
        }


@dataclass
class ISIStats:
    """Inter-stimulus interval statistics for one task."""
    task: str
    mean_isi_s: float
    std_isi_s: float
    min_isi_s: float
    max_isi_s: float
    jitter_cv: float  # coefficient of variation = std/mean; 0 = regular, >0.3 = jittered

    @property
    def is_jittered(self) -> bool:
        return self.jitter_cv > 0.15

    @property
    def is_fixed_rate(self) -> bool:
        return self.jitter_cv < 0.05

    @property
    def mean_s(self) -> float:
        """Backward-compatible alias for ``mean_isi_s``."""
        return self.mean_isi_s

    @property
    def std_s(self) -> float:
        """Backward-compatible alias for ``std_isi_s``."""
        return self.std_isi_s


@dataclass
class LabelLandscape:
    """Complete label and event analysis for a dataset, fetched remotely.

    Attributes
    ----------
    dataset_id:
        The OpenNeuro dataset accession number.
    n_events_files:
        Total events TSV files in the manifest.
    n_files_fetched:
        Successfully fetched (network failures reduce this).
    n_files_failed:
        Files that could not be fetched or parsed.
    label_column:
        The detected events column used as class label (usually ``trial_type``).
    trial_type_stats:
        Per-class statistics sorted by frequency.
    subject_profiles:
        Per-subject event profiles.
    isi_stats:
        Per-task ISI statistics.
    coverage_pct:
        Fraction of signal entity keys (subject+session+task+run combos) that
        have at least one events file.
    imbalance_ratio:
        max_class_count / min_class_count. 1.0 = perfect balance.
    cross_subject_consistency:
        Fraction of subjects that have exactly the same set of trial types as
        the global set. 1.0 = fully consistent.
    recommendations:
        Actionable list of data quality observations.
    """
    dataset_id: str
    n_events_files: int = 0
    n_files_fetched: int = 0
    n_files_failed: int = 0
    label_column: str | None = None
    trial_type_stats: list[TrialTypeStats] = field(default_factory=list)
    subject_profiles: dict[str, SubjectEventProfile] = field(default_factory=dict)
    isi_stats: dict[str, ISIStats] = field(default_factory=dict)
    coverage_pct: float = 0.0
    imbalance_ratio: float | None = None
    cross_subject_consistency: float = 0.0
    recommendations: list[str] = field(default_factory=list)

    @property
    def n_classes(self) -> int:
        return len(self.trial_type_stats)

    @property
    def total_events(self) -> int:
        return sum(s.total_count for s in self.trial_type_stats)

    @property
    def class_names(self) -> list[str]:
        return [s.trial_type for s in self.trial_type_stats]

    @property
    def class_counts(self) -> dict[str, int]:
        return {s.trial_type: s.total_count for s in self.trial_type_stats}

    @property
    def imbalance_severity(self) -> str:
        if self.imbalance_ratio is None:
            return "unknown"
        if self.imbalance_ratio <= 1.2:
            return "balanced"
        if self.imbalance_ratio <= 3.0:
            return "mild"
        if self.imbalance_ratio <= 10.0:
            return "severe"
        return "critical"

    def summary(self) -> str:
        lines = [
            f"Label Landscape — {self.dataset_id}",
            f"Events files: {self.n_files_fetched}/{self.n_events_files} fetched"
            + (f" ({self.n_files_failed} failed)" if self.n_files_failed else ""),
            f"Label column: {self.label_column or '(not detected)'}",
            f"Classes: {self.n_classes}  Total events: {self.total_events:,}",
            f"Coverage: {self.coverage_pct * 100:.1f}% of signal keys have events",
            f"Imbalance: {self.imbalance_ratio:.2f}x ({self.imbalance_severity})"
            if self.imbalance_ratio is not None else "Imbalance: n/a",
            f"Cross-subject consistency: {self.cross_subject_consistency * 100:.1f}%",
            "",
            "Class distribution:",
        ]
        for stat in self.trial_type_stats[:15]:
            bar = "█" * min(30, int(stat.total_count / max(1, self.total_events) * 30))
            lines.append(
                f"  {stat.trial_type:30s}  {stat.total_count:6d}  {bar}"
            )
        if self.recommendations:
            lines += ["", "Recommendations:"]
            for rec in self.recommendations:
                lines.append(f"  • {rec}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "n_events_files": self.n_events_files,
            "n_files_fetched": self.n_files_fetched,
            "n_files_failed": self.n_files_failed,
            "label_column": self.label_column,
            "n_classes": self.n_classes,
            "total_events": self.total_events,
            "class_counts": self.class_counts,
            "coverage_pct": round(self.coverage_pct, 4),
            "imbalance_ratio": round(self.imbalance_ratio, 3) if self.imbalance_ratio else None,
            "imbalance_severity": self.imbalance_severity,
            "cross_subject_consistency": round(self.cross_subject_consistency, 4),
            "trial_type_stats": [s.to_dict() for s in self.trial_type_stats],
            "subject_profiles": {k: v.to_dict() for k, v in self.subject_profiles.items()},
            "isi_stats": {
                task: {
                    "mean_isi_s": s.mean_isi_s,
                    "std_isi_s": s.std_isi_s,
                    "jitter_cv": s.jitter_cv,
                    "is_jittered": s.is_jittered,
                }
                for task, s in self.isi_stats.items()
            },
            "recommendations": self.recommendations,
        }


# ── Analyzer ──────────────────────────────────────────────────────────────────

class LabelLandscapeAnalyzer:
    """Analyze event/label distributions across a dataset remotely.

    Fetches all events TSVs concurrently from the CDN (typically <2 MB total
    for datasets with hundreds of subjects), aggregates across all files, and
    produces a ``LabelLandscape`` report.

    Usage::

        analyzer = LabelLandscapeAnalyzer(gateway)
        landscape = analyzer.analyze(manifest)
        print(landscape.summary())
    """

    # Common events column names in BIDS, in preference order
    _LABEL_COLUMNS = ["trial_type", "stim_type", "condition", "event_type", "category"]
    # Columns to skip when looking for label columns
    _NON_LABEL_COLUMNS = {"onset", "duration", "sample", "response_time",
                           "reaction_time", "rt", "stim_file", "HED",
                           "trigger", "button_pushed", "circle_duration"}

    def __init__(self, gateway: RemoteFileGateway | None = None) -> None:
        self._gateway = gateway or RemoteFileGateway()

    def analyze(
        self,
        manifest: Manifest,
        *,
        label_column: str | None = None,
        concurrency: int = 24,
        max_events_files: int | None = None,
    ) -> LabelLandscape:
        """Fetch all events files and analyze label distributions.

        Parameters
        ----------
        manifest:
            Built ``Manifest`` (provides file list with CDN URLs).
        label_column:
            Override auto-detected label column name. If None, auto-detects
            from ``trial_type``, ``stim_type``, ``condition``, etc.
        concurrency:
            Max parallel HTTP connections for fetching events files.
        max_events_files:
            Cap the number of events files to fetch (useful for large datasets
            where a sample is sufficient).

        Returns
        -------
        LabelLandscape
        """
        events_files = [
            f for f in manifest.files
            if not f.is_dir
            and f.suffix == "events"
            and f.extension == ".tsv"
            and f.urls
        ]

        log.info(
            "LabelLandscape: found %d events files in manifest for %s",
            len(events_files), manifest.dataset_id,
        )

        if max_events_files is not None:
            events_files = events_files[:max_events_files]

        landscape = LabelLandscape(
            dataset_id=manifest.dataset_id,
            n_events_files=len(events_files),
        )

        if not events_files:
            landscape.recommendations.append(
                "No events TSV files found. This dataset has no event markers — "
                "trial_type labels are unavailable. Consider using participants.tsv "
                "group/age columns for subject-level labels instead."
            )
            return landscape

        # Build URL map for concurrent fetching
        url_map: dict[str, str] = {}
        fr_map: dict[str, FileRecord] = {}
        for fr in events_files:
            try:
                url = _pick_url(fr)
                url_map[fr.path] = url
                fr_map[fr.path] = fr
            except Exception:
                pass

        log.info("Fetching %d events files (concurrency=%d)...", len(url_map), concurrency)
        results = self._gateway.batch_fetch_tsv(url_map, concurrency=concurrency)

        # Accumulate stats
        frames_by_path: dict[str, Any] = {}
        n_failed = 0
        for path, result in results.items():
            if isinstance(result, Exception):
                log.debug("Events fetch failed for %s: %s", path, result)
                n_failed += 1
            else:
                frames_by_path[path] = result

        landscape.n_files_fetched = len(frames_by_path)
        landscape.n_files_failed = n_failed

        if not frames_by_path:
            landscape.recommendations.append(
                "All events files failed to fetch. Check network connectivity "
                "or whether the dataset requires authentication."
            )
            return landscape

        # Auto-detect label column from first successful frame
        detected_col = label_column
        if detected_col is None:
            first_frame = next(iter(frames_by_path.values()))
            detected_col = _detect_label_column(first_frame.columns)

        landscape.label_column = detected_col

        # Aggregate across all files
        _aggregate(
            landscape=landscape,
            frames_by_path=frames_by_path,
            fr_map=fr_map,
            label_column=detected_col,
        )

        # Coverage computation
        _compute_coverage(landscape, manifest, events_files)

        # Recommendations
        landscape.recommendations = _build_recommendations(landscape)

        return landscape


# ── Internal aggregation ──────────────────────────────────────────────────────

def _detect_label_column(columns: list[str]) -> str | None:
    """Return the most likely label column from the events TSV header."""
    col_lower = {c.lower(): c for c in columns}
    for candidate in LabelLandscapeAnalyzer._LABEL_COLUMNS:
        if candidate.lower() in col_lower:
            return col_lower[candidate.lower()]
    # Fallback: any non-standard column that isn't a numeric/timing field
    for col in columns:
        if col.lower() not in LabelLandscapeAnalyzer._NON_LABEL_COLUMNS:
            return col
    return None


def _aggregate(
    *,
    landscape: LabelLandscape,
    frames_by_path: dict[str, Any],
    fr_map: dict[str, FileRecord],
    label_column: str | None,
) -> None:
    """Aggregate trial_type counts across all fetched events DataFrames."""
    # trial_type → {subject → count}, durations
    class_totals: dict[str, int] = {}
    class_subject_counts: dict[str, dict[str, int]] = {}
    class_durations: dict[str, list[float]] = {}

    # Per-subject accumulators
    subject_stats: dict[str, SubjectEventProfile] = {}

    # Per-task ISI accumulators: task → list of ISI values (seconds)
    task_isis: dict[str, list[float]] = {}

    for path, df in frames_by_path.items():
        fr = fr_map.get(path)
        subject = (fr.subject if fr else None) or "unknown"
        task = (fr.task if fr else None) or "unknown"

        if subject not in subject_stats:
            subject_stats[subject] = SubjectEventProfile(subject=subject)
        sp = subject_stats[subject]
        sp.n_events_files += 1
        if task not in sp.tasks_covered:
            sp.tasks_covered.append(task)

        # ISI computation from onset column
        if "onset" in df.columns:
            try:
                onsets = df["onset"].cast(float).sort().to_list()
                if len(onsets) >= 2:
                    isis = [onsets[i + 1] - onsets[i] for i in range(len(onsets) - 1)]
                    if task not in task_isis:
                        task_isis[task] = []
                    task_isis[task].extend(isis)
            except Exception:
                pass

        if label_column is None or label_column not in df.columns:
            # Count all events without class breakdown
            sp.total_events += len(df)
            continue

        # Tally per trial_type
        has_duration = "duration" in df.columns
        for row in df.iter_rows(named=True):
            tt = str(row.get(label_column) or "n/a").strip()
            if not tt or tt.lower() in ("n/a", "nan", "none", ""):
                tt = "(unlabeled)"

            class_totals[tt] = class_totals.get(tt, 0) + 1
            class_subject_counts.setdefault(tt, {})
            class_subject_counts[tt][subject] = class_subject_counts[tt].get(subject, 0) + 1

            if has_duration:
                try:
                    dur = float(row.get("duration") or 0)
                    if dur > 0:
                        class_durations.setdefault(tt, []).append(dur)
                except (TypeError, ValueError):
                    pass

            sp.trial_type_counts[tt] = sp.trial_type_counts.get(tt, 0) + 1
            sp.total_events += 1

    # Build TrialTypeStats list (sorted by frequency desc)
    stats: list[TrialTypeStats] = []
    for tt, total in sorted(class_totals.items(), key=lambda x: -x[1]):
        durations = class_durations.get(tt, [])
        stats.append(TrialTypeStats(
            trial_type=tt,
            total_count=total,
            subject_counts=class_subject_counts.get(tt, {}),
            mean_duration_s=statistics.mean(durations) if durations else None,
            std_duration_s=statistics.stdev(durations) if len(durations) > 1 else 0.0,
            min_duration_s=min(durations) if durations else None,
            max_duration_s=max(durations) if durations else None,
        ))
    landscape.trial_type_stats = stats
    landscape.subject_profiles = subject_stats

    # Imbalance ratio
    counts = [s.total_count for s in stats if s.total_count > 0]
    if len(counts) >= 2:
        landscape.imbalance_ratio = max(counts) / min(counts)

    # Cross-subject consistency
    global_classes = frozenset(class_totals)
    if global_classes and subject_stats:
        consistent = sum(
            1 for sp in subject_stats.values()
            if frozenset(sp.trial_type_counts) == global_classes
        )
        landscape.cross_subject_consistency = consistent / len(subject_stats)

    # ISI stats
    for task, isis_list in task_isis.items():
        if len(isis_list) >= 2:
            mean_isi = statistics.mean(isis_list)
            std_isi = statistics.stdev(isis_list)
            landscape.isi_stats[task] = ISIStats(
                task=task,
                mean_isi_s=round(mean_isi, 3),
                std_isi_s=round(std_isi, 3),
                min_isi_s=round(min(isis_list), 3),
                max_isi_s=round(max(isis_list), 3),
                jitter_cv=round(std_isi / mean_isi, 3) if mean_isi > 0 else 0.0,
            )


def _compute_coverage(
    landscape: LabelLandscape,
    manifest: Manifest,
    events_files: list[FileRecord],
) -> None:
    """Compute what fraction of signal files have a matching events file.

    Uses ``SidecarResolver.find_events()`` which implements BIDS fallback
    matching (exact key → without run → without session+run) instead of
    strict exact-key equality. This correctly handles datasets where events
    files omit run or session entities.
    """
    from qortex.manifest.sidecar import SidecarResolver

    signal_modalities = {"eeg", "meg", "ieeg", "fmri", "fnirs", "bold"}
    signal_files = [
        f for f in manifest.files
        if not f.is_dir
        and f.modality in signal_modalities
        and f.extension not in (".json", ".tsv", ".csv")
    ]

    if not signal_files:
        landscape.coverage_pct = 1.0 if events_files else 0.0
        return

    resolver = SidecarResolver(manifest.files)
    covered = sum(1 for sf in signal_files if resolver.find_events(sf) is not None)
    landscape.coverage_pct = covered / len(signal_files)


def _build_recommendations(landscape: LabelLandscape) -> list[str]:
    recs: list[str] = []

    if landscape.n_files_failed > 0:
        fail_pct = landscape.n_files_failed / max(1, landscape.n_events_files) * 100
        recs.append(
            f"{landscape.n_files_failed} events files ({fail_pct:.0f}%) failed to fetch. "
            "Some subjects/sessions may have incomplete label data."
        )

    if landscape.label_column is None:
        recs.append(
            "No standard label column (trial_type, stim_type, condition) found. "
            "Review your events TSV column names — the ML pipeline cannot assign "
            "class labels without an identified label column."
        )
        return recs

    if landscape.coverage_pct < 0.5:
        recs.append(
            f"Only {landscape.coverage_pct * 100:.0f}% of signal files have matching events. "
            "Over half the signal data cannot be labeled from events files. "
            "Consider checking if events files use different entity naming conventions."
        )

    if landscape.imbalance_ratio is not None and landscape.imbalance_ratio > 3.0:
        counts = landscape.class_counts
        majority = max(counts, key=counts.__getitem__)
        minority = min(counts, key=counts.__getitem__)
        recs.append(
            f"Severe class imbalance: {majority!r} has {counts[majority]:,} trials "
            f"vs {minority!r} with {counts[minority]:,} ({landscape.imbalance_ratio:.1f}×). "
            "Use class-weighted loss or resampling to prevent majority-class bias."
        )
    elif landscape.imbalance_ratio is not None and landscape.imbalance_ratio > 1.5:
        recs.append(
            f"Mild class imbalance ({landscape.imbalance_ratio:.1f}×). "
            "Consider stratified splits (stratify_by_label=True) to maintain balance."
        )

    if landscape.cross_subject_consistency < 0.8:
        recs.append(
            f"Only {landscape.cross_subject_consistency * 100:.0f}% of subjects have "
            "the full set of trial types. Some subjects may be missing classes — "
            "verify whether this is intentional (e.g., different protocol versions) "
            "or a data integrity issue."
        )

    for task, isi in landscape.isi_stats.items():
        if isi.is_jittered:
            recs.append(
                f"Task '{task}' has jittered ISIs (CV={isi.jitter_cv:.2f}, "
                f"mean={isi.mean_isi_s:.2f}s ± {isi.std_isi_s:.2f}s). "
                "Jittered designs are preferred for fMRI; for EEG/MEG, ensure "
                "your epoch rejection handles variable baseline periods."
            )

    if landscape.n_classes == 1:
        recs.append(
            "Only one trial type found. This may be a continuous monitoring or "
            "resting-state paradigm — classification is not directly applicable. "
            "Consider regression targets or connectivity features instead."
        )

    if landscape.n_classes == 0:
        recs.append(
            "No labeled events found despite having events files. "
            f"Column '{landscape.label_column}' may contain only null/n/a values. "
            "Inspect the raw events files or try a different label_column."
        )

    return recs
