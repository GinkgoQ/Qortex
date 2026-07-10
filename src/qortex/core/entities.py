"""Qortex core data models.

These pydantic models are the shared language across every subsystem.
No module should define ad-hoc dicts for cross-boundary data.
All models are immutable (frozen) by default to prevent accidental mutation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Base ──────────────────────────────────────────────────────────────────────

class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class _Mutable(BaseModel):
    model_config = ConfigDict(frozen=False)


# ── OpenNeuro API layer ───────────────────────────────────────────────────────

class DatasetRef(_Frozen):
    """Lightweight reference to an OpenNeuro dataset."""

    id: str
    name: str | None = None
    description: str | None = None
    doi: str | None = None
    license: str | None = None
    authors: list[str] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    # Raw metadata bag from the API — preserved but not parsed
    raw_metadata: dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.id}" + (f" — {self.name}" if self.name else "")


class SnapshotRef(_Frozen):
    """Reference to one versioned snapshot of a dataset."""

    dataset_id: str
    tag: str
    # Full compound id as returned by the API: "{dataset_id}:{tag}"
    id: str
    doi: str | None = None
    created: datetime | None = None
    file_count: int | None = None
    size: int | None = None
    # Content-addressed hash — stable across re-publishes with identical content.
    # Use for reliable cache invalidation: if hexsha unchanged, file tree is unchanged.
    hexsha: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _infer_id(cls, data: dict) -> dict:
        if "id" not in data and "dataset_id" in data and "tag" in data:
            data = dict(data)
            data["id"] = f"{data['dataset_id']}:{data['tag']}"
        return data

    @property
    def size_gb(self) -> float | None:
        return self.size / 1e9 if self.size is not None else None


# ── BIDS entity layer ─────────────────────────────────────────────────────────

class BIDSEntities(_Frozen):
    """Parsed BIDS key-value entities extracted from a filename."""

    subject: str | None = None     # sub-XX  (value only, no "sub-" prefix)
    session: str | None = None     # ses-XX
    task: str | None = None
    run: str | None = None
    acquisition: str | None = None
    direction: str | None = None
    space: str | None = None
    resolution: str | None = None
    echo: str | None = None
    part: str | None = None
    hemisphere: str | None = None
    density: str | None = None
    processing: str | None = None
    split: str | None = None
    # Any non-standard BIDS entities land here
    extra: dict[str, str] = Field(default_factory=dict)

    def as_dict(self) -> dict[str, str]:
        """Return all non-None standard entities as a flat dict."""
        result: dict[str, str] = {}
        for field_name in self.model_fields:
            if field_name == "extra":
                result.update(self.extra)
                continue
            v = getattr(self, field_name)
            if v is not None:
                result[field_name] = v
        return result


# ── Manifest layer ────────────────────────────────────────────────────────────

class FileRecord(_Frozen):
    """One file entry in an OpenNeuro snapshot manifest."""

    # OpenNeuro API fields
    id: str
    path: str                        # BIDS-relative, e.g. "sub-01/eeg/sub-01_task-rest_eeg.set"
    filename: str                    # basename
    extension: str                   # ".set", ".nii.gz" (compound extensions preserved)
    size: int | None = None          # bytes; None when API does not provide
    urls: list[str] = Field(default_factory=list)
    checksum: str | None = None      # MD5 from ETag when reliable
    is_dir: bool = False

    # BIDS-parsed structure
    datatype: str | None = None      # "eeg" | "anat" | "func" | "meg" | ...
    suffix: str | None = None        # BIDS suffix: "eeg", "bold", "T1w", "events" ...
    modality: str | None = None      # unified modality: "eeg", "meg", "mri", "fmri", ...
    entities: BIDSEntities = Field(default_factory=BIDSEntities)

    # Sidecar / inheritance grouping
    sidecar_group: str | None = None  # stable hash of the inheritance chain key

    @field_validator("path", "filename")
    @classmethod
    def _non_empty_file_identity(cls, value: str) -> str:
        if not value:
            raise ValueError("file path and filename must not be empty")
        return value

    @field_validator("size")
    @classmethod
    def _non_negative_optional_size(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("file size must be >= 0 when provided")
        return value

    @property
    def subject(self) -> str | None:
        return self.entities.subject

    @property
    def session(self) -> str | None:
        return self.entities.session

    @property
    def task(self) -> str | None:
        return self.entities.task

    @property
    def run(self) -> str | None:
        return self.entities.run

    @property
    def size_mb(self) -> float | None:
        return self.size / 1e6 if self.size is not None else None

    @property
    def is_essential(self) -> bool:
        """True for top-level BIDS metadata files always included in downloads."""
        return self.filename in {
            "dataset_description.json",
            "participants.tsv",
            "participants.json",
            "README",
            "CHANGES",
            ".bidsignore",
        }


class CompanionSet(_Frozen):
    """Files required to interpret or load one primary BIDS data file."""

    primary: FileRecord
    sidecars: list[FileRecord] = Field(default_factory=list)
    events: FileRecord | None = None
    channels: FileRecord | None = None
    electrodes: FileRecord | None = None
    coordsystem: FileRecord | None = None
    scans: FileRecord | None = None
    bvec: FileRecord | None = None
    bval: FileRecord | None = None
    eeg_data: FileRecord | None = None
    participants: FileRecord | None = None
    dataset_description: FileRecord | None = None
    extra: list[FileRecord] = Field(default_factory=list)

    @property
    def files(self) -> list[FileRecord]:
        """Return unique companion files, excluding the primary file."""
        out: list[FileRecord] = []
        seen: set[str] = set()
        for file in [
            self.dataset_description,
            self.participants,
            self.scans,
            self.channels,
            self.electrodes,
            self.coordsystem,
            self.events,
            self.bvec,
            self.bval,
            self.eeg_data,
            *self.sidecars,
            *self.extra,
        ]:
            if file is not None and file.path not in seen:
                seen.add(file.path)
                out.append(file)
        return out


class LogicalRecording(_Frozen):
    """A semantic recording unit built from one primary file and companions."""

    id: str
    primary: FileRecord
    companions: CompanionSet
    modality: str | None = None
    datatype: str | None = None
    subject: str | None = None
    session: str | None = None
    task: str | None = None
    run: str | None = None
    has_events: bool = False
    has_label_candidates: bool = False
    has_labels: bool = False
    downloadable: bool = False
    loadable: bool = True
    estimated_bytes: int = 0
    issues: list[str] = Field(default_factory=list)

    @property
    def files(self) -> list[FileRecord]:
        return [self.primary, *self.companions.files]


class ManifestSummary(_Frozen):
    """Aggregate statistics computed from the full file list."""

    subjects: list[str] = Field(default_factory=list)
    sessions: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=list)
    datatypes: list[str] = Field(default_factory=list)
    suffixes: list[str] = Field(default_factory=list)
    file_count: int = 0
    dir_count: int = 0
    total_size: int = 0         # 0 when any file lacks a size
    total_size_known: bool = True
    has_derivatives: bool = False
    has_bidsignore: bool = False
    has_events: bool = False
    has_participants_tsv: bool = False

    @field_validator("file_count", "dir_count", "total_size")
    @classmethod
    def _non_negative_counts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("manifest summary counts and sizes must be >= 0")
        return value

    @property
    def total_size_gb(self) -> float:
        return self.total_size / 1e9

    @property
    def n_subjects(self) -> int:
        return len(self.subjects)


class Manifest(_Mutable):
    """Complete, queryable file manifest for one dataset snapshot."""

    dataset_id: str
    snapshot: str
    doi: str | None = None
    files: list[FileRecord] = Field(default_factory=list)
    summary: ManifestSummary = Field(default_factory=ManifestSummary)
    built_at: datetime = Field(default_factory=_utcnow)

    # ── Internal index (populated lazily on first get_file call) ─────────
    _path_index: dict[str, FileRecord] | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_unique_paths(self) -> "Manifest":
        seen: set[str] = set()
        duplicates: list[str] = []
        for file in self.files:
            if file.path in seen:
                duplicates.append(file.path)
            seen.add(file.path)
        if duplicates:
            sample = ", ".join(duplicates[:5])
            suffix = f" and {len(duplicates) - 5} more" if len(duplicates) > 5 else ""
            raise ValueError(f"manifest contains duplicate file paths: {sample}{suffix}")
        return self

    def _ensure_index(self) -> None:
        if self._path_index is None:
            self._path_index = {f.path: f for f in self.files}

    def rebuild_index(self) -> None:
        """Rebuild the path index. Call after mutating ``self.files``."""
        self._path_index = {f.path: f for f in self.files}

    # ── Convenience queries ───────────────────────────────────────────────

    def filter(
        self,
        *,
        subjects: list[str] | None = None,
        sessions: list[str] | None = None,
        tasks: list[str] | None = None,
        modalities: list[str] | None = None,
        datatypes: list[str] | None = None,
        extensions: list[str] | None = None,
        exclude_dirs: bool = True,
        include_shared: bool = True,
    ) -> list[FileRecord]:
        """Return files matching the given entity filters.

        Parameters
        ----------
        subjects / sessions / tasks:
            Entity value lists. When ``include_shared=True`` (default), files
            without that entity (e.g. root ``participants.tsv``) are also
            included — they are shared across all subjects/sessions/tasks.
            Set ``include_shared=False`` to get strictly per-entity files.
        modalities / datatypes / extensions:
            Strict filters — no null-passthrough. Files without a matching
            modality, datatype, or extension are excluded.
        exclude_dirs:
            Skip directory entries (default True).
        include_shared:
            When True, files whose entity is None pass through subject/session/
            task filters (BIDS root files shared across entities). When False,
            only files whose entity exactly matches the filter value are returned.
        """
        result = self.files
        if exclude_dirs:
            result = [f for f in result if not f.is_dir]
        if subjects:
            s = set(subjects)
            if include_shared:
                result = [f for f in result if f.subject in s or f.subject is None]
            else:
                result = [f for f in result if f.subject in s]
        if sessions:
            s = set(sessions)
            if include_shared:
                result = [f for f in result if f.session in s or f.session is None]
            else:
                result = [f for f in result if f.session in s]
        if tasks:
            s = set(tasks)
            if include_shared:
                result = [f for f in result if f.task in s or f.task is None]
            else:
                result = [f for f in result if f.task in s]
        if modalities:
            s = set(modalities)
            result = [f for f in result if f.modality in s]
        if datatypes:
            s = set(datatypes)
            result = [f for f in result if f.datatype in s]
        if extensions:
            s = set(extensions)
            result = [f for f in result if f.extension in s]
        return result

    def get_file(self, path: str) -> FileRecord | None:
        """Return the FileRecord for a BIDS-relative path, or None. O(1)."""
        self._ensure_index()
        return self._path_index.get(path)

    def has_file(self, path: str) -> bool:
        """Return True if the manifest contains this path. O(1)."""
        self._ensure_index()
        return path in self._path_index

    def estimate_size(self, files: list[FileRecord] | None = None) -> int:
        """Return total bytes for the given files (or all manifest files).

        Files with unknown size contribute 0. Use ``summary.total_size_known``
        to detect whether the total is reliable.
        """
        src = files if files is not None else self.files
        return sum(f.size or 0 for f in src)

    def subjects_with_modality(self, modality: str) -> list[str]:
        """Return sorted BIDS subject IDs (``sub-XX`` form) with this modality."""
        return sorted({
            f"sub-{f.subject}" for f in self.files
            if f.modality == modality and f.subject is not None and not f.is_dir
        })

    def tasks_for_subject(self, subject: str) -> list[str]:
        """Return sorted task names for one subject.

        Accepts both BIDS-prefixed form (``sub-01``) and raw entity value (``01``).
        """
        raw = subject.removeprefix("sub-")
        return sorted({
            f.task for f in self.files
            if f.subject == raw and f.task is not None
        })

    def files_by_suffix(self, suffix: str) -> list[FileRecord]:
        """Return all non-directory files with the given BIDS suffix."""
        return [f for f in self.files if not f.is_dir and f.suffix == suffix]


# ── Planning layer ────────────────────────────────────────────────────────────

class SelectionSpec(_Frozen):
    """User-facing selection parameters — resolved to a file list by the planner."""

    subjects: list[str] | None = None
    sessions: list[str] | None = None
    tasks: list[str] | None = None
    modalities: list[str] | None = None
    datatypes: list[str] | None = None
    include: list[str] | None = None        # glob patterns
    exclude: list[str] | None = None        # glob patterns
    # Exact BIDS-relative paths resolved via set membership — NOT treated as
    # glob patterns.  Use this instead of ``include`` when paths come from the
    # manifest directly (e.g. ``download_paths()``) to avoid misbehaviour with
    # paths containing glob metacharacters (``[``, ``]``, ``*``, ``?``, etc.).
    exact_paths: list[str] | None = None
    include_derivatives: bool = False
    metadata_only: bool = False
    with_companions: bool = True
    event_complete: bool = False
    label_ready: bool = False
    loadable_only: bool = False
    max_size_gb: float | None = None
    conversion_target: str | None = None


class SelectionReason(_Frozen):
    """Human-readable explanation for why a file is in a DownloadPlan."""

    path: str
    reason: str
    source: str = "selector"
    recording_id: str | None = None


class DownloadPlan(_Mutable):
    """Resolved file list + metadata for a pending download."""

    dataset_id: str
    snapshot: str
    target_dir: Path
    selection: SelectionSpec
    files: list[FileRecord] = Field(default_factory=list)
    essential_files: list[FileRecord] = Field(default_factory=list)
    estimated_bytes: int = 0
    warnings: list[str] = Field(default_factory=list)
    selection_reasons: dict[str, list[SelectionReason]] = Field(default_factory=dict)
    recordings: list[LogicalRecording] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("estimated_bytes")
    @classmethod
    def _non_negative_estimated_bytes(cls, value: int) -> int:
        if value < 0:
            raise ValueError("estimated_bytes must be >= 0")
        return value

    @property
    def estimated_gb(self) -> float:
        return self.estimated_bytes / 1e9

    @property
    def n_files(self) -> int:
        return len(self.files)

    def summary(self) -> str:
        lines = [
            f"Dataset : {self.dataset_id}  (snapshot {self.snapshot})",
            f"Target  : {self.target_dir}",
            f"Files   : {self.n_files}",
            f"Size    : {self.estimated_gb:.2f} GB (estimated)",
        ]
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ⚠  {w}")
        return "\n".join(lines)

    def explain(self, limit: int | None = None) -> str:
        """Return a compact explanation of why files are selected."""
        rows: list[str] = []
        files = self.files if limit is None else self.files[:limit]
        for file in files:
            reasons = self.selection_reasons.get(file.path, [])
            if reasons:
                text = "; ".join(r.reason for r in reasons)
            else:
                text = "selected by filters"
            rows.append(f"{file.path}: {text}")
        remaining = len(self.files) - len(files)
        if remaining > 0:
            rows.append(f"... {remaining} more file(s)")
        return "\n".join(rows)


# ── Download result layer ─────────────────────────────────────────────────────

class DownloadRecord(_Frozen):
    """Outcome for a single successfully downloaded (or cache-hit) file."""

    file: FileRecord
    local_path: Path
    bytes_written: int
    elapsed: float            # seconds
    retries: int = 0
    from_cache: bool = False

    @field_validator("bytes_written", "retries")
    @classmethod
    def _non_negative_download_counts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("download counts must be >= 0")
        return value

    @field_validator("elapsed")
    @classmethod
    def _non_negative_elapsed(cls, value: float) -> float:
        if value < 0:
            raise ValueError("elapsed must be >= 0")
        return value


class FailedRecord(_Frozen):
    """Outcome for a file that could not be downloaded."""

    file: FileRecord
    error: str
    attempts: int

    @field_validator("attempts")
    @classmethod
    def _positive_attempts(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("attempts must be > 0")
        return value


class DownloadResult(_Mutable):
    """Aggregated outcome of executing a DownloadPlan."""

    plan: DownloadPlan
    downloaded: list[DownloadRecord] = Field(default_factory=list)
    skipped: list[DownloadRecord] = Field(default_factory=list)    # cache hits
    failed: list[FailedRecord] = Field(default_factory=list)
    bytes_downloaded: int = 0
    elapsed: float = 0.0

    @field_validator("bytes_downloaded")
    @classmethod
    def _non_negative_bytes_downloaded(cls, value: int) -> int:
        if value < 0:
            raise ValueError("bytes_downloaded must be >= 0")
        return value

    @field_validator("elapsed")
    @classmethod
    def _non_negative_result_elapsed(cls, value: float) -> float:
        if value < 0:
            raise ValueError("elapsed must be >= 0")
        return value

    @property
    def success(self) -> bool:
        return len(self.failed) == 0

    @property
    def n_downloaded(self) -> int:
        return len(self.downloaded)

    @property
    def n_skipped(self) -> int:
        return len(self.skipped)

    @property
    def n_failed(self) -> int:
        return len(self.failed)

    def report(self) -> str:
        lines = [
            f"Downloaded : {self.n_downloaded} files ({self.bytes_downloaded / 1e6:.1f} MB)",
            f"Skipped    : {self.n_skipped} files (cache hits)",
            f"Failed     : {self.n_failed} files",
            f"Elapsed    : {self.elapsed:.1f}s",
        ]
        if self.failed:
            lines.append("Failed files:")
            for rec in self.failed:
                lines.append(f"  ✗ {rec.file.path}: {rec.error}")
        return "\n".join(lines)


# ── Parse / Load layer ────────────────────────────────────────────────────────

class SignalRecord(_Mutable):
    """Loaded electrophysiology data (EEG / MEG / iEEG / fNIRS)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    file: FileRecord
    raw: Any                    # mne.io.BaseRaw — held loosely to avoid hard dep
    sfreq: float
    n_channels: int
    duration: float             # seconds
    channel_names: list[str]
    channel_types: list[str]
    events_file: FileRecord | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sfreq", "duration")
    @classmethod
    def _positive_signal_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("signal sampling frequency and duration must be > 0")
        return value

    @field_validator("n_channels")
    @classmethod
    def _positive_n_channels(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("n_channels must be > 0")
        return value

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_channels, int(self.duration * self.sfreq))


class ImageRecord(_Mutable):
    """Loaded neuroimaging volume (MRI / fMRI / PET / DWI)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    file: FileRecord
    img: Any                    # nibabel.Nifti1Image or similar
    shape: tuple[int, ...]
    voxel_size: tuple[float, ...]
    affine: Any
    tr: float | None = None     # repetition time (fMRI)
    n_volumes: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EventsRecord(_Mutable):
    """Loaded events / behavioral TSV file."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    file: FileRecord
    data: Any                   # polars.DataFrame
    columns: list[str]
    n_events: int
    label_column: str | None = None
    label_values: list[Any] = Field(default_factory=list)

    @field_validator("n_events")
    @classmethod
    def _non_negative_n_events(cls, value: int) -> int:
        if value < 0:
            raise ValueError("n_events must be >= 0")
        return value


class SampleRecord(_Mutable):
    """A windowed, ML-ready sample produced by the ETL pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: Any                           # np.ndarray shape (channels, time) or (x, y, z[, t])
    label: Any | None = None
    label_name: str | None = None
    subject: str | None = None
    session: str | None = None
    task: str | None = None
    run: str | None = None
    modality: str = ""
    onset: float | None = None          # seconds from recording start
    duration: float | None = None       # window length in seconds
    sfreq: float | None = None          # for signals
    split: str | None = None            # "train" | "val" | "test"
    provenance: dict[str, Any] = Field(default_factory=dict)


class LabelPolicy(_Frozen):
    """Configuration for extracting labels from events or metadata."""

    source: Literal["events", "participants", "sessions", "scans", "constant"] = "events"
    column: str | None = None
    task: str | None = None
    missing: Literal["drop", "keep", "error"] = "drop"
    positive_values: list[Any] | None = None


class SplitPlan(_Frozen):
    """Leakage-aware split assignment metadata."""

    strategy: str = "subject"
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15
    seed: int = 42
    group_by: str = "subject"
    assignments: dict[str, str] = Field(default_factory=dict)
    class_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    leakage_risks: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_fractions(self) -> "SplitPlan":
        for name, value in {"train": self.train, "val": self.val, "test": self.test}.items():
            if value < 0:
                raise ValueError(f"{name} split fraction must be >= 0")
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split fractions must sum to 1.0, got {total:.6f}")
        return self


class ReadinessFinding(_Frozen):
    """One actionable dataset-readiness finding."""

    severity: Literal["info", "warning", "error"]
    code: str
    message: str
    path: str | None = None
    recording_id: str | None = None
    recommendation: str | None = None


class ReadinessReport(_Mutable):
    """Decision-oriented report for download/load/label/convert/train readiness."""

    dataset_id: str
    snapshot: str
    n_recordings: int = 0
    n_loadable: int = 0
    n_event_complete: int = 0
    n_label_ready: int = 0
    estimated_bytes: int = 0
    findings: list[ReadinessFinding] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_utcnow)

    @field_validator("n_recordings", "n_loadable", "n_event_complete", "n_label_ready", "estimated_bytes")
    @classmethod
    def _non_negative_readiness_counts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("readiness counts and estimated_bytes must be >= 0")
        return value

    @property
    def can_download(self) -> bool:
        return not any(f.severity == "error" and f.code.startswith("download.") for f in self.findings)

    @property
    def can_convert(self) -> bool:
        return not any(f.severity == "error" and f.code.startswith("convert.") for f in self.findings)

    @property
    def score(self) -> float:
        penalty = 0.0
        for finding in self.findings:
            penalty += 12.0 if finding.severity == "error" else 4.0 if finding.severity == "warning" else 0.0
        if self.n_recordings:
            event_gap = 1.0 - (self.n_event_complete / self.n_recordings)
            label_gap = 1.0 - (self.n_label_ready / self.n_recordings)
            penalty += event_gap * 20.0 + label_gap * 20.0
        return max(0.0, round(100.0 - penalty, 1))

    def summary(self) -> str:
        lines = [
            f"Dataset : {self.dataset_id} (snapshot {self.snapshot})",
            f"Score   : {self.score:.1f}/100",
            f"Records : {self.n_recordings} logical recording(s)",
            f"Events  : {self.n_event_complete}/{self.n_recordings} event-complete",
            f"Labels  : {self.n_label_ready}/{self.n_recordings} label-ready",
        ]
        for finding in self.findings:
            prefix = finding.severity.upper()
            location = f" [{finding.path or finding.recording_id}]" if finding.path or finding.recording_id else ""
            lines.append(f"{prefix}: {finding.code}{location}: {finding.message}")
        return "\n".join(lines)


# ── Provenance layer ──────────────────────────────────────────────────────────

class ProvenanceRecord(_Frozen):
    """Attached to every artifact (download, conversion, split, export)."""

    qortex_version: str
    created_at: datetime = Field(default_factory=_utcnow)
    dataset_id: str
    snapshot: str
    doi: str | None = None
    operation: Literal["download", "convert", "split", "export", "validate", "eda"]
    config: dict[str, Any] = Field(default_factory=dict)
    source_files: list[str] = Field(default_factory=list)
    output_path: str | None = None


# ── Conversion layer ──────────────────────────────────────────────────────────

class ConversionResult(_Frozen):
    """Outcome of an ETL conversion pipeline run."""

    output_format: str
    output_path: Path
    n_samples: int
    n_subjects: int
    splits: dict[str, int] = Field(default_factory=dict)  # split_name → count
    elapsed: float
    provenance: ProvenanceRecord
    warnings: list[str] = Field(default_factory=list)
    artifact_manifest: "ArtifactManifest | None" = None

    @field_validator("n_samples", "n_subjects")
    @classmethod
    def _non_negative_conversion_counts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("conversion counts must be >= 0")
        return value

    @field_validator("elapsed")
    @classmethod
    def _non_negative_conversion_elapsed(cls, value: float) -> float:
        if value < 0:
            raise ValueError("elapsed must be >= 0")
        return value


class ArtifactManifest(_Frozen):
    """Machine-readable contract for a converted Qortex artifact."""

    artifact_id: str
    dataset_id: str
    snapshot: str
    doi: str | None = None
    output_format: str
    output_path: str
    n_samples: int
    n_subjects: int
    splits: dict[str, int] = Field(default_factory=dict)
    source_files: list[str] = Field(default_factory=list)
    label_policy: dict[str, Any] = Field(default_factory=dict)
    window_config: dict[str, Any] = Field(default_factory=dict)
    split_config: dict[str, Any] = Field(default_factory=dict)
    data_schema: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)


# ── EDA layer ─────────────────────────────────────────────────────────────────

class ModalitySummary(_Frozen):
    modality: str
    n_files: int
    n_subjects: int
    total_size: int
    extensions: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class DatasetSummary(_Frozen):
    dataset_id: str
    snapshot: str
    doi: str | None = None
    n_files: int
    n_subjects: int
    n_sessions: int
    n_tasks: int
    total_size: int
    modalities: list[str] = Field(default_factory=list)
    has_derivatives: bool = False
    has_events: bool = False

    @field_validator("n_files", "n_subjects", "n_sessions", "n_tasks", "total_size")
    @classmethod
    def _non_negative_dataset_summary_counts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("dataset summary counts and total_size must be >= 0")
        return value


class QualityMetrics(_Frozen):
    bids_score: float = 0.0          # 0–100
    ml_readiness_score: float = 0.0  # 0–100
    loadability_score: float = 0.0   # 0–100
    missing_events_pct: float = 0.0
    missing_sidecar_pct: float = 0.0
    issues: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class EventLabelSummary(_Frozen):
    """Local event-file label distribution summary."""

    path: str
    n_events: int
    label_column: str | None = None
    label_counts: dict[str, int] = Field(default_factory=dict)
    n_missing_labels: int = 0

    @field_validator("n_events", "n_missing_labels")
    @classmethod
    def _non_negative_event_counts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("event counts must be >= 0")
        return value

    @property
    def n_classes(self) -> int:
        return len(self.label_counts)

    @property
    def imbalance_ratio(self) -> float | None:
        counts = [count for count in self.label_counts.values() if count > 0]
        if len(counts) < 2:
            return None
        return round(max(counts) / min(counts), 3)


class EDAReport(_Mutable):
    """Full exploratory analysis report for a dataset."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_id: str
    snapshot: str | None = None
    dataset_path: Path | None = None
    summary: DatasetSummary | None = None
    modality_summaries: dict[str, ModalitySummary] = Field(default_factory=dict)
    event_summaries: list[EventLabelSummary] = Field(default_factory=list)
    quality: QualityMetrics = Field(default_factory=QualityMetrics)
    generated_at: datetime = Field(default_factory=_utcnow)
    # Populated by eda.report
    html: str | None = None
    figures: dict[str, Any] = Field(default_factory=dict)

    def to_html(self, path: str | Path) -> Path:
        if self.html is None:
            raise RuntimeError("Call eda.report.render() before to_html().")
        out = Path(path)
        out.write_text(self.html, encoding="utf-8")
        return out

    def to_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(self.model_dump_json(indent=2, exclude={"html", "figures"}),
                       encoding="utf-8")
        return out


# ── Validation / local index layer ────────────────────────────────────────────

class ValidationIssue(_Frozen):
    """One normalized issue returned by a BIDS validation backend."""

    severity: Literal["error", "warning", "ignored", "info"]
    code: str
    message: str
    path: str | None = None
    line: int | None = None
    column: int | None = None
    evidence: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(_Mutable):
    """Stable Qortex view of a BIDS Validator run."""

    dataset_path: str
    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    command: list[str] = Field(default_factory=list)
    return_code: int | None = None
    validator_version: str | None = None
    elapsed: float = 0.0
    stdout: str = ""
    stderr: str = ""
    generated_at: datetime = Field(default_factory=_utcnow)
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def ignored(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "ignored"]

    @property
    def n_errors(self) -> int:
        return len(self.errors)

    @property
    def n_warnings(self) -> int:
        return len(self.warnings)

    @property
    def n_ignored(self) -> int:
        return len(self.ignored)

    @property
    def score(self) -> float:
        penalty = self.n_errors * 12.0 + self.n_warnings * 3.0
        return max(0.0, round(100.0 - penalty, 1))

    def summary(self) -> str:
        lines = [
            f"Dataset : {self.dataset_path}",
            f"Valid   : {self.valid}",
            f"Score   : {self.score:.1f}/100",
            f"Errors  : {self.n_errors}",
            f"Warnings: {self.n_warnings}",
            f"Ignored : {self.n_ignored}",
        ]
        for issue in self.issues[:20]:
            location = f" [{issue.path}]" if issue.path else ""
            lines.append(
                f"{issue.severity.upper()}: {issue.code}{location}: {issue.message}"
            )
        remaining = len(self.issues) - 20
        if remaining > 0:
            lines.append(f"... {remaining} more issue(s)")
        return "\n".join(lines)

    def to_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return out

    def to_markdown(self, path: str | Path) -> Path:
        out = Path(path)
        lines = [
            "# Qortex Validation Report",
            "",
            f"- Dataset: `{self.dataset_path}`",
            f"- Valid: `{self.valid}`",
            f"- Score: `{self.score:.1f}/100`",
            f"- Errors: `{self.n_errors}`",
            f"- Warnings: `{self.n_warnings}`",
            f"- Ignored: `{self.n_ignored}`",
            "",
            "| Severity | Code | Path | Message |",
            "| --- | --- | --- | --- |",
        ]
        for issue in self.issues:
            message = issue.message.replace("|", "\\|")
            lines.append(
                f"| {issue.severity} | `{issue.code}` | "
                f"`{issue.path or ''}` | {message} |"
            )
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out

    def to_html(self, path: str | Path) -> Path:
        out = Path(path)
        rows = []
        for issue in self.issues:
            rows.append(
                "<tr>"
                f"<td>{escape(issue.severity)}</td>"
                f"<td><code>{escape(issue.code)}</code></td>"
                f"<td><code>{escape(issue.path or '')}</code></td>"
                f"<td>{escape(issue.message)}</td>"
                "</tr>"
            )
        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Qortex Validation Report</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; margin: 32px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; vertical-align: top; }}
    code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Qortex Validation Report</h1>
  <p><strong>Dataset:</strong> <code>{escape(self.dataset_path)}</code></p>
  <p><strong>Valid:</strong> {self.valid} &nbsp; <strong>Score:</strong> {self.score:.1f}/100</p>
  <p><strong>Errors:</strong> {self.n_errors} &nbsp; <strong>Warnings:</strong> {self.n_warnings} &nbsp; <strong>Ignored:</strong> {self.n_ignored}</p>
  <table>
    <thead><tr><th>Severity</th><th>Code</th><th>Path</th><th>Message</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
        out.write_text(html, encoding="utf-8")
        return out


class ValidationDiff(_Mutable):
    """Comparison between two normalized validation reports."""

    before_path: str
    after_path: str
    added: list[ValidationIssue] = Field(default_factory=list)
    resolved: list[ValidationIssue] = Field(default_factory=list)
    persisted: list[ValidationIssue] = Field(default_factory=list)

    @property
    def n_added(self) -> int:
        return len(self.added)

    @property
    def n_resolved(self) -> int:
        return len(self.resolved)

    @property
    def n_persisted(self) -> int:
        return len(self.persisted)

    def summary(self) -> str:
        return "\n".join([
            f"Before    : {self.before_path}",
            f"After     : {self.after_path}",
            f"Added     : {self.n_added}",
            f"Resolved  : {self.n_resolved}",
            f"Persisted : {self.n_persisted}",
        ])


class LocalFileRecord(_Frozen):
    """One file observed in a local BIDS tree."""

    path: str
    size: int
    mtime: float
    is_dir: bool = False
    extension: str | None = None
    entities: dict[str, Any] = Field(default_factory=dict)

    @field_validator("size")
    @classmethod
    def _non_negative_local_size(cls, value: int) -> int:
        if value < 0:
            raise ValueError("local file size must be >= 0")
        return value


class LocalIndexReport(_Mutable):
    """Reconciliation between a local BIDS tree and an OpenNeuro manifest."""

    dataset_path: str
    n_files: int = 0
    n_dirs: int = 0
    missing_remote: list[str] = Field(default_factory=list)
    extra_local: list[str] = Field(default_factory=list)
    size_mismatches: list[str] = Field(default_factory=list)
    indexed_files: list[LocalFileRecord] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_utcnow)

    @property
    def n_missing(self) -> int:
        return len(self.missing_remote)

    @property
    def n_extra(self) -> int:
        return len(self.extra_local)

    @property
    def n_size_mismatches(self) -> int:
        return len(self.size_mismatches)

    @property
    def consistent(self) -> bool:
        return (
            self.n_missing == 0
            and self.n_extra == 0
            and self.n_size_mismatches == 0
        )

    def summary(self) -> str:
        return "\n".join([
            f"Dataset path     : {self.dataset_path}",
            f"Indexed files    : {self.n_files}",
            f"Indexed dirs     : {self.n_dirs}",
            f"Missing remote   : {self.n_missing}",
            f"Extra local      : {self.n_extra}",
            f"Size mismatches  : {self.n_size_mismatches}",
            f"Consistent       : {self.consistent}",
        ])

    def to_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return out


class FilePreview(_Frozen):
    """Small local or remote preview of one dataset file."""

    dataset_id: str
    snapshot: str
    path: str
    source: Literal["local", "remote"]
    bytes_read: int
    truncated: bool = False
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    text: str | None = None
    content_type: str | None = None
    encoding: str = "utf-8"
