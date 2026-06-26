"""DatasetInspector — full metadata fetch and profile without downloading.

This module is the primary answer to "inspect and investigate the full DB":
it fetches every piece of metadata OpenNeuro exposes, builds a Manifest from
the file tree, and produces a DatasetProfile that quantifies ML readiness,
modality coverage, subject/session/task structure, companion completeness,
and download cost — all without writing a single data byte to disk.

Architecture
------------
* Uses ``OpenNeuroClient`` (sync GraphQL) to fetch dataset info, snapshot list,
  and recursive file tree.
* Uses ``ManifestBuilder`` to turn raw API file dicts into a typed ``Manifest``.
* All analysis is done in-memory over the manifest's ``FileRecord`` list.
* No I/O on local files; no downloads.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qortex.client.graphql import OpenNeuroClient
from qortex.core.config import QortexConfig, get_config
from qortex.core.entities import (
    DatasetRef,
    FileRecord,
    Manifest,
    SnapshotRef,
)
from qortex.core.exceptions import DatasetNotFoundError
from qortex.manifest.builder import ManifestBuilder

log = logging.getLogger(__name__)

# ── ML readiness scoring constants ────────────────────────────────────────────

_PERMISSIVE_LICENSES = {
    "cc0": 30,
    "cc0-1.0": 30,
    "pddl": 28,
    "cc-by": 22,
    "cc-by-4.0": 22,
    "cc-by-3.0": 20,
    "odc-by": 18,
    "mit": 18,
    "apache-2.0": 15,
}

_SUPPORTED_MODALITIES = {
    "eeg", "meg", "ieeg", "fnirs",
    "bold", "t1w", "t2w", "dwi", "cbv",
    "pet",
}


# ── Result model ──────────────────────────────────────────────────────────────

@dataclass
class ModalityBreakdown:
    modality: str
    n_files: int
    total_bytes: int
    n_subjects: set[str] = field(default_factory=set)
    tasks: set[str] = field(default_factory=set)
    extensions: Counter = field(default_factory=Counter)
    has_events: bool = False
    n_events_files: int = 0

    @property
    def total_gb(self) -> float:
        return self.total_bytes / 1e9

    @property
    def n_unique_subjects(self) -> int:
        return len(self.n_subjects)

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "n_files": self.n_files,
            "total_gb": round(self.total_gb, 3),
            "n_subjects": self.n_unique_subjects,
            "tasks": sorted(self.tasks),
            "top_extensions": dict(self.extensions.most_common(5)),
            "has_events": self.has_events,
            "n_events_files": self.n_events_files,
        }


@dataclass
class MLReadinessScore:
    """Multidimensional ML readiness breakdown."""

    events_score: float = 0.0       # 0–30: events/labels present and covering
    subjects_score: float = 0.0     # 0–20: subject count adequacy
    license_score: float = 0.0      # 0–15: license permissiveness
    modality_score: float = 0.0     # 0–15: supported ML modalities present
    structure_score: float = 0.0    # 0–10: BIDS structural completeness
    companion_score: float = 0.0    # 0–10: events+channels+sidecar coverage
    total: float = 0.0

    @property
    def grade(self) -> str:
        if self.total >= 80:
            return "A"
        if self.total >= 65:
            return "B"
        if self.total >= 50:
            return "C"
        if self.total >= 35:
            return "D"
        return "F"

    def breakdown_lines(self) -> list[str]:
        return [
            f"  Events/labels     : {self.events_score:.0f}/30",
            f"  Subject count     : {self.subjects_score:.0f}/20",
            f"  License           : {self.license_score:.0f}/15",
            f"  Modality support  : {self.modality_score:.0f}/15",
            f"  BIDS structure    : {self.structure_score:.0f}/10",
            f"  Companion coverage: {self.companion_score:.0f}/10",
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": round(self.total, 1),
            "grade": self.grade,
            "events_score": self.events_score,
            "subjects_score": self.subjects_score,
            "license_score": self.license_score,
            "modality_score": self.modality_score,
            "structure_score": self.structure_score,
            "companion_score": self.companion_score,
        }


@dataclass
class DatasetProfile:
    """Complete inspection profile for one dataset snapshot."""

    dataset_ref: DatasetRef
    snapshot: SnapshotRef
    all_snapshots: list[SnapshotRef]
    manifest: Manifest

    # Computed analytics — filled in by _analyse()
    n_subjects: int = 0
    n_sessions: int = 0
    n_tasks: int = 0
    subjects: list[str] = field(default_factory=list)
    sessions: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    modality_breakdown: dict[str, ModalityBreakdown] = field(default_factory=dict)
    subject_task_matrix: dict[str, list[str]] = field(default_factory=dict)
    subject_session_matrix: dict[str, list[str]] = field(default_factory=dict)
    events_coverage: float = 0.0          # fraction of signal files with an events companion
    channels_coverage: float = 0.0        # fraction of signal files with a channels TSV
    sidecar_coverage: float = 0.0         # fraction of signal files with a JSON sidecar
    total_size_gb: float = 0.0
    per_subject_avg_gb: float = 0.0
    n_signal_files: int = 0
    n_events_files: int = 0
    n_sidecar_files: int = 0
    n_derivative_files: int = 0
    has_participants_tsv: bool = False
    has_dataset_description: bool = False
    has_readme: bool = False
    ml_readiness: MLReadinessScore = field(default_factory=MLReadinessScore)
    recommendations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a compact human-readable summary."""
        ds = self.dataset_ref
        snap = self.snapshot
        ml = self.ml_readiness

        lines = [
            "=" * 60,
            f"Dataset : {ds.id}  ({snap.tag})",
            f"Name    : {ds.name or '—'}",
            f"DOI     : {ds.doi or '—'}",
            f"License : {ds.license or '—'}",
            f"Authors : {', '.join(ds.authors) or '—'}",
            "-" * 60,
            f"Subjects     : {self.n_subjects}",
            f"Sessions     : {self.n_sessions}",
            f"Tasks        : {self.n_tasks} — {', '.join(self.tasks) or '—'}",
            f"Modalities   : {', '.join(self.modality_breakdown) or '—'}",
            f"Total size   : {self.total_size_gb:.2f} GB",
            f"Avg/subject  : {self.per_subject_avg_gb:.3f} GB",
            "-" * 60,
            f"Signal files    : {self.n_signal_files}",
            f"Events files    : {self.n_events_files}",
            f"Sidecar files   : {self.n_sidecar_files}",
            f"Events coverage : {self.events_coverage * 100:.1f}%",
            f"Channel coverage: {self.channels_coverage * 100:.1f}%",
            f"Sidecar coverage: {self.sidecar_coverage * 100:.1f}%",
            "-" * 60,
            f"ML Readiness: {ml.total:.0f}/100 (Grade {ml.grade})",
            *ml.breakdown_lines(),
        ]
        if self.recommendations:
            lines += ["", "Recommendations:"]
            for rec in self.recommendations:
                lines.append(f"  • {rec}")
        if len(self.all_snapshots) > 1:
            lines += [
                "-" * 60,
                f"Snapshots: {len(self.all_snapshots)} available "
                f"({', '.join(s.tag for s in self.all_snapshots[-5:])})",
            ]
        lines.append("=" * 60)
        return "\n".join(lines)

    def report(self) -> str:
        """Return the full modality-level report."""
        header = self.summary()
        mod_lines = ["\nModality breakdown:"]
        for mod, mb in self.modality_breakdown.items():
            mod_lines.append(
                f"  {mod:12s}  {mb.n_files:5d} files  "
                f"{mb.total_gb:7.3f} GB  "
                f"{mb.n_unique_subjects:4d} subjects  "
                f"tasks={','.join(sorted(mb.tasks)) or '—'}"
            )
        subj_lines = []
        if self.subject_task_matrix:
            subj_lines = ["\nSubject × task matrix (first 10 subjects):"]
            for subj, tasks in list(self.subject_task_matrix.items())[:10]:
                subj_lines.append(f"  {subj}: {', '.join(sorted(tasks))}")
        return "\n".join([header] + mod_lines + subj_lines)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the full profile."""
        return {
            "dataset_id": self.dataset_ref.id,
            "name": self.dataset_ref.name,
            "doi": self.dataset_ref.doi,
            "license": self.dataset_ref.license,
            "authors": self.dataset_ref.authors,
            "snapshot": self.snapshot.tag,
            "all_snapshot_tags": [s.tag for s in self.all_snapshots],
            "n_subjects": self.n_subjects,
            "n_sessions": self.n_sessions,
            "n_tasks": self.n_tasks,
            "subjects": self.subjects,
            "sessions": self.sessions,
            "tasks": self.tasks,
            "modality_breakdown": {
                mod: mb.to_dict() for mod, mb in self.modality_breakdown.items()
            },
            "subject_task_matrix": {
                k: sorted(v) for k, v in self.subject_task_matrix.items()
            },
            "subject_session_matrix": {
                k: sorted(v) for k, v in self.subject_session_matrix.items()
            },
            "total_size_gb": round(self.total_size_gb, 4),
            "per_subject_avg_gb": round(self.per_subject_avg_gb, 4),
            "n_signal_files": self.n_signal_files,
            "n_events_files": self.n_events_files,
            "n_sidecar_files": self.n_sidecar_files,
            "n_derivative_files": self.n_derivative_files,
            "events_coverage": round(self.events_coverage, 4),
            "channels_coverage": round(self.channels_coverage, 4),
            "sidecar_coverage": round(self.sidecar_coverage, 4),
            "has_participants_tsv": self.has_participants_tsv,
            "has_dataset_description": self.has_dataset_description,
            "has_readme": self.has_readme,
            "ml_readiness": self.ml_readiness.to_dict(),
            "recommendations": self.recommendations,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, ensure_ascii=False)


# ── Inspector ─────────────────────────────────────────────────────────────────

class DatasetInspector:
    """Fetch and profile an OpenNeuro dataset without downloading any files.

    Parameters
    ----------
    token:
        Optional OpenNeuro API token (for private datasets).
    config:
        ``QortexConfig`` instance; ``None`` → ``get_config()``.

    Examples
    --------
    >>> inspector = DatasetInspector()
    >>> profile = inspector.inspect("ds000117")
    >>> print(profile.summary())
    >>> print(profile.report())
    >>> profile.as_dict()  # JSON-serialisable
    """

    def __init__(
        self,
        token: str | None = None,
        config: QortexConfig | None = None,
    ) -> None:
        self._cfg = config or get_config()
        self._client = OpenNeuroClient(token=token, config=self._cfg)
        self._builder = ManifestBuilder()

    def inspect(
        self,
        dataset_id: str,
        tag: str | None = None,
    ) -> DatasetProfile:
        """Fetch all metadata and return a DatasetProfile.

        Parameters
        ----------
        dataset_id:
            OpenNeuro dataset accession number (e.g. ``"ds000117"``).
        tag:
            Snapshot tag.  ``None`` → latest published snapshot.

        Returns
        -------
        DatasetProfile
            Fully populated, no download performed.
        """
        log.info("Inspecting dataset %s (tag=%s)", dataset_id, tag or "latest")

        # ── 1. Fetch dataset-level metadata ──────────────────────────────
        dataset_ref = self._client.get_dataset(dataset_id)
        log.debug("Got dataset ref: %s", dataset_ref.id)

        # ── 2. Fetch snapshot list ────────────────────────────────────────
        all_snapshots = self._client.get_snapshots(dataset_id)
        log.debug("Got %d snapshots", len(all_snapshots))

        # ── 3. Fetch full recursive file tree ────────────────────────────
        snapshot_ref, raw_files = self._client.get_files(dataset_id, tag=tag)
        log.info(
            "Fetched %d file records for %s snapshot=%s",
            len(raw_files), dataset_id, snapshot_ref.tag,
        )

        # ── 4. Build typed Manifest ───────────────────────────────────────
        manifest = self._builder.build(dataset_id, snapshot_ref, raw_files)

        # ── 5. Deep analysis ─────────────────────────────────────────────
        profile = DatasetProfile(
            dataset_ref=dataset_ref,
            snapshot=snapshot_ref,
            all_snapshots=all_snapshots,
            manifest=manifest,
        )
        _analyse(profile)
        return profile

    def compare(
        self,
        dataset_id_a: str,
        dataset_id_b: str,
        tag_a: str | None = None,
        tag_b: str | None = None,
    ) -> dict[str, Any]:
        """Side-by-side comparison of two datasets.

        Returns a dict with both profiles and a diff section highlighting
        differences in subject count, modalities, events coverage, size,
        and ML readiness.
        """
        pa = self.inspect(dataset_id_a, tag=tag_a)
        pb = self.inspect(dataset_id_b, tag=tag_b)

        def _diff_num(label: str, a: float, b: float) -> dict:
            return {
                "field": label,
                f"{dataset_id_a}": round(a, 3),
                f"{dataset_id_b}": round(b, 3),
                "delta": round(b - a, 3),
                "winner": dataset_id_b if b > a else (dataset_id_a if a > b else "tie"),
            }

        diff = [
            _diff_num("n_subjects", pa.n_subjects, pb.n_subjects),
            _diff_num("total_size_gb", pa.total_size_gb, pb.total_size_gb),
            _diff_num("events_coverage", pa.events_coverage, pb.events_coverage),
            _diff_num("sidecar_coverage", pa.sidecar_coverage, pb.sidecar_coverage),
            _diff_num("ml_readiness.total", pa.ml_readiness.total, pb.ml_readiness.total),
            _diff_num("n_tasks", pa.n_tasks, pb.n_tasks),
        ]
        modality_union = sorted(set(pa.modality_breakdown) | set(pb.modality_breakdown))

        return {
            "datasets": [dataset_id_a, dataset_id_b],
            "diff": diff,
            "modality_union": modality_union,
            f"{dataset_id_a}_only_modalities": sorted(
                set(pa.modality_breakdown) - set(pb.modality_breakdown)
            ),
            f"{dataset_id_b}_only_modalities": sorted(
                set(pb.modality_breakdown) - set(pa.modality_breakdown)
            ),
            f"{dataset_id_a}": pa.as_dict(),
            f"{dataset_id_b}": pb.as_dict(),
        }

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "DatasetInspector":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ── Analysis engine ───────────────────────────────────────────────────────────

def _analyse(profile: DatasetProfile) -> None:
    """Populate all computed fields of a DatasetProfile in-place."""
    manifest = profile.manifest
    files = [f for f in manifest.files if not f.is_dir]

    # ── Structural BIDS files ─────────────────────────────────────────────
    file_paths = {f.path for f in files}
    profile.has_participants_tsv = any(
        p.endswith("participants.tsv") for p in file_paths
    )
    profile.has_dataset_description = any(
        p.endswith("dataset_description.json") for p in file_paths
    )
    profile.has_readme = any(
        p.rsplit("/", 1)[-1].upper().startswith("README") for p in file_paths
    )

    # ── Entity sets ───────────────────────────────────────────────────────
    subjects: set[str] = set()
    sessions: set[str] = set()
    tasks: set[str] = set()

    for f in files:
        if f.subject:
            subjects.add(f.subject)
        if f.session:
            sessions.add(f.session)
        if f.task:
            tasks.add(f.task)

    profile.subjects = sorted(subjects)
    profile.sessions = sorted(sessions)
    profile.tasks = sorted(tasks)
    profile.n_subjects = len(subjects)
    profile.n_sessions = len(sessions)
    profile.n_tasks = len(tasks)

    # ── Modality breakdown ────────────────────────────────────────────────
    modality_map: dict[str, ModalityBreakdown] = {}
    total_bytes = 0

    # Build sets for companion coverage
    # Events: "events" suffix + ".tsv" extension
    events_keys: set[tuple] = set()   # (subject, session, task, run)
    # Channels: "channels" suffix + ".tsv"
    channels_keys: set[tuple] = set()
    # Sidecars: ".json" extension, non-directory, non-dataset_description
    sidecar_keys: set[tuple] = set()  # (subject, session, task, run, suffix-group)

    n_events_files = 0
    n_sidecar_files = 0
    n_derivative_files = 0
    n_signal = 0

    for f in files:
        path = f.path
        size = f.size or 0
        total_bytes += size
        suffix = f.suffix or ""
        ext = f.extension or ""
        modality = f.modality
        subject = f.subject
        session = f.session
        task = f.task
        run = f.run
        entity_key = (subject, session, task, run)

        # Derivative check
        if path.startswith("derivatives/") or "/derivatives/" in path:
            n_derivative_files += 1
            continue  # exclude from coverage metrics

        # Events files
        if suffix == "events" and ext == ".tsv":
            n_events_files += 1
            events_keys.add(entity_key)
            continue

        # Channels files
        if suffix == "channels" and ext == ".tsv":
            channels_keys.add(entity_key)
            continue

        # JSON sidecars (exclude dataset-level JSON)
        if ext == ".json" and subject is not None:
            sidecar_keys.add(entity_key)
            n_sidecar_files += 1
            continue

        # Skip pure metadata files with no modality
        if not modality:
            continue

        n_signal += 1
        if modality not in modality_map:
            modality_map[modality] = ModalityBreakdown(modality=modality)
        mb = modality_map[modality]
        mb.n_files += 1
        mb.total_bytes += size
        if subject:
            mb.n_subjects.add(subject)
        if task:
            mb.tasks.add(task)
        if ext:
            mb.extensions[ext] += 1

        # Subject × task and subject × session matrices
        if subject and task:
            profile.subject_task_matrix.setdefault(subject, [])
            if task not in profile.subject_task_matrix[subject]:
                profile.subject_task_matrix[subject].append(task)
        if subject and session:
            profile.subject_session_matrix.setdefault(subject, [])
            if session not in profile.subject_session_matrix[subject]:
                profile.subject_session_matrix[subject].append(session)

    # ── Coverage metrics ──────────────────────────────────────────────────
    # Compute per-signal-file coverage by matching entity keys
    # Build signal entity key set from signal files
    signal_entity_keys: set[tuple] = set()
    for f in files:
        path = f.path
        if path.startswith("derivatives/") or "/derivatives/" in path:
            continue
        if not f.modality:
            continue
        if f.suffix in {"events", "channels"} or (f.extension or "") in {".json", ".tsv"}:
            continue
        signal_entity_keys.add((f.subject, f.session, f.task, f.run))

    n_with_events = sum(1 for k in signal_entity_keys if k in events_keys)
    n_with_channels = sum(1 for k in signal_entity_keys if k in channels_keys)
    n_with_sidecar = sum(1 for k in signal_entity_keys if k in sidecar_keys)

    denom = len(signal_entity_keys) or 1
    profile.events_coverage = n_with_events / denom
    profile.channels_coverage = n_with_channels / denom
    profile.sidecar_coverage = n_with_sidecar / denom

    profile.n_signal_files = n_signal
    profile.n_events_files = n_events_files
    profile.n_sidecar_files = n_sidecar_files
    profile.n_derivative_files = n_derivative_files
    profile.modality_breakdown = modality_map
    profile.total_size_gb = total_bytes / 1e9
    profile.per_subject_avg_gb = (
        profile.total_size_gb / profile.n_subjects
        if profile.n_subjects else 0.0
    )

    # ── ML readiness ──────────────────────────────────────────────────────
    profile.ml_readiness = _score_ml_readiness(profile)

    # ── Recommendations ───────────────────────────────────────────────────
    profile.recommendations = _build_recommendations(profile)


def _score_ml_readiness(profile: DatasetProfile) -> MLReadinessScore:
    sc = MLReadinessScore()

    # Events/labels (0–30): fraction of signal entity keys with events
    sc.events_score = round(min(30.0, profile.events_coverage * 30.0), 1)

    # Subject count (0–20): log-scaled adequacy
    n = profile.n_subjects
    if n >= 100:
        sc.subjects_score = 20.0
    elif n >= 50:
        sc.subjects_score = 16.0
    elif n >= 20:
        sc.subjects_score = 12.0
    elif n >= 10:
        sc.subjects_score = 8.0
    elif n >= 5:
        sc.subjects_score = 4.0
    else:
        sc.subjects_score = 0.0

    # License (0–15): permissiveness table
    license_str = (profile.dataset_ref.license or "").lower().strip()
    raw_license_score = _PERMISSIVE_LICENSES.get(license_str, 5)
    sc.license_score = round(min(15.0, raw_license_score / 2.0), 1)

    # Modality support (0–15): at least one known ML-supported modality
    supported = _SUPPORTED_MODALITIES & set(profile.modality_breakdown)
    n_supported = len(supported)
    if n_supported >= 3:
        sc.modality_score = 15.0
    elif n_supported == 2:
        sc.modality_score = 12.0
    elif n_supported == 1:
        sc.modality_score = 8.0
    else:
        sc.modality_score = 0.0

    # BIDS structure (0–10): structural metadata completeness
    bids_points = 0.0
    if profile.has_participants_tsv:
        bids_points += 4.0
    if profile.has_dataset_description:
        bids_points += 3.0
    if profile.has_readme:
        bids_points += 1.0
    if profile.n_sessions > 0:
        bids_points += 1.0
    if profile.n_tasks > 0:
        bids_points += 1.0
    sc.structure_score = min(10.0, bids_points)

    # Companion coverage (0–10): average of events + channels + sidecar coverage
    avg_companion = (
        profile.events_coverage * 4.0
        + profile.channels_coverage * 3.0
        + profile.sidecar_coverage * 3.0
    )
    sc.companion_score = round(min(10.0, avg_companion), 1)

    sc.total = round(
        sc.events_score
        + sc.subjects_score
        + sc.license_score
        + sc.modality_score
        + sc.structure_score
        + sc.companion_score,
        1,
    )
    return sc


def _build_recommendations(profile: DatasetProfile) -> list[str]:
    recs: list[str] = []

    if profile.events_coverage < 0.5:
        pct = profile.events_coverage * 100
        recs.append(
            f"Only {pct:.0f}% of signal entity keys have an events TSV — "
            "event-aligned windowing will fall back to fixed windows for uncovered files. "
            "Consider requesting or contributing events files for full ML utility."
        )

    if profile.n_subjects < 10:
        recs.append(
            f"Dataset has only {profile.n_subjects} subject(s). "
            "Cross-validation across subjects will produce highly variable estimates. "
            "Consider combining with other datasets using qortex.Dataset.join()."
        )

    license_str = (profile.dataset_ref.license or "").lower().strip()
    if license_str and license_str not in _PERMISSIVE_LICENSES:
        recs.append(
            f"License '{profile.dataset_ref.license}' may restrict redistribution or "
            "commercial use of derived ML artifacts. Verify compliance before publishing models."
        )
    elif not license_str:
        recs.append(
            "No license specified. Assume all-rights-reserved until clarified with the authors."
        )

    unsupported = set(profile.modality_breakdown) - _SUPPORTED_MODALITIES
    if unsupported:
        recs.append(
            f"Modalities {sorted(unsupported)} have no built-in Qortex loader. "
            "Implement a custom qortex.loaders entry-point plugin to load these files."
        )

    if not profile.has_participants_tsv:
        recs.append(
            "No participants.tsv found. Subject-level metadata (age, sex, group) "
            "will not be available as label candidates. "
            "Check if participants.tsv is missing from the manifest or not yet uploaded."
        )

    if profile.sidecar_coverage < 0.5 and profile.n_signal_files > 0:
        recs.append(
            f"JSON sidecar coverage is {profile.sidecar_coverage * 100:.0f}%. "
            "Acquisition parameters (sampling rate, channel types, task description) "
            "may be missing for many files, which will degrade BIDS sidecar inheritance."
        )

    if profile.total_size_gb > 100:
        recs.append(
            f"Dataset is large ({profile.total_size_gb:.1f} GB). "
            "Use SelectionSpec filters (subjects, tasks, modalities) or "
            "max_size_gb to download a representative subset first."
        )

    if profile.n_tasks == 0:
        recs.append(
            "No task entities found in file paths. "
            "This may be a resting-state or structural-only dataset — "
            "ensure your label strategy (e.g. participants.tsv group column) "
            "can supply training labels without event files."
        )

    return recs
