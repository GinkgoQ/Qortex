"""Typed contract models for the Qortex NeuroAI runtime.

Every major operation consumes and produces typed contracts.  No anonymous
dicts cross subsystem boundaries.  All contracts are Pydantic models so they
can be serialised to JSON and stored in artifact provenance.

Hierarchy
---------
SourceProfile       — what a source provides
ModelProfile        — what a model expects and produces
InputContract       — formal expectation a model has on its input
OutputContract      — formal schema for model outputs
PreprocessPlan      — ordered transform chain satisfying the input contract
CompatibilityReport — can the source satisfy the model?
PipelineRunReport   — combined runtime + provenance record
ArtifactContract    — what a written artifact guarantees
LatencyReport       — end-to-end timing breakdown
WarningItem         — structured warning attached to any report
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

try:
    from pydantic import BaseModel, Field
    _PYDANTIC = True
except ImportError:
    from dataclasses import field as Field
    BaseModel = object  # fallback shim
    _PYDANTIC = False


# ── Enums ─────────────────────────────────────────────────────────────────────

class EvidenceStatus(str, Enum):
    confirmed  = "confirmed"
    inferred   = "inferred"
    missing    = "missing"
    unknown    = "unknown"
    blocked    = "blocked"

class CompatibilityStatus(str, Enum):
    compatible              = "compatible"
    compatible_with_transforms = "compatible_with_transforms"
    uncertain               = "uncertain"
    incompatible            = "incompatible"

class TransformKind(str, Enum):
    resample          = "resample"
    channel_select    = "channel_select"
    channel_reorder   = "channel_reorder"
    channel_map       = "channel_map"
    bandpass          = "bandpass"
    normalize         = "normalize"
    window            = "window"
    cast_dtype        = "cast_dtype"
    rescale_intensity = "rescale_intensity"
    reorient          = "reorient"
    resample_spatial  = "resample_spatial"
    pad_or_crop       = "pad_or_crop"
    add_batch_dim     = "add_batch_dim"
    add_channel_dim   = "add_channel_dim"
    to_tensor         = "to_tensor"

class Modality(str, Enum):
    eeg       = "eeg"
    meg       = "meg"
    ieeg      = "ieeg"
    fnirs     = "fnirs"
    mri       = "mri"
    fmri      = "fmri"
    dwi       = "dwi"
    pet       = "pet"
    ct        = "ct"
    dicom     = "dicom"
    image     = "image"
    video     = "video"
    timeseries = "timeseries"
    tabular   = "tabular"
    embedding = "embedding"

class AxisConvention(str, Enum):
    RAS    = "RAS"
    LAS    = "LAS"
    LPS    = "LPS"
    spatial_zyx     = "spatial_zyx"      # (Z, Y, X) — DICOM/NIfTI native
    spatial_xyz     = "spatial_xyz"      # (X, Y, Z)
    channels_first  = "channels_first"   # (C, ...) or (B, C, ...)
    channels_last   = "channels_last"    # (..., C) or (B, ..., C)
    time_channels   = "time_channels"    # (T, C)
    channels_time   = "channels_time"    # (C, T)
    batch_channels_time = "batch_channels_time"  # (B, C, T)
    batch_channels_xyz  = "batch_channels_xyz"   # (B, C, Z, Y, X) / (B, C, X, Y, Z)


# ── Warning item ──────────────────────────────────────────────────────────────

class WarningItem(BaseModel if _PYDANTIC else object):
    code: str
    message: str
    severity: Literal["info", "warning", "error"] = "warning"
    evidence: dict[str, Any] = Field(default_factory=dict) if _PYDANTIC else {}
    suggestion: str | None = None

    def __str__(self) -> str:
        parts = [f"[{self.severity.upper()}] {self.code}: {self.message}"]
        if self.suggestion:
            parts.append(f"  → {self.suggestion}")
        return "\n".join(parts)

    if not _PYDANTIC:
        def __init__(self, code, message, severity="warning", evidence=None, suggestion=None):
            self.code = code
            self.message = message
            self.severity = severity
            self.evidence = evidence or {}
            self.suggestion = suggestion

        def model_dump(self):
            return {"code": self.code, "message": self.message,
                    "severity": self.severity, "evidence": self.evidence,
                    "suggestion": self.suggestion}


# ── Internal data abstractions ────────────────────────────────────────────────

class QortexAbstraction(BaseModel if _PYDANTIC else object):
    """Base for all internal data abstractions per AGENT.md §4.5.

    The ``data`` field carries the actual numpy array so source adapters can
    pass both the contract metadata and the raw samples through the pipeline
    in one object.  The runtime engine extracts ``data`` via ``_extract_array()``.
    """
    abstraction_type: str
    shape: tuple[int, ...]
    axes: list[str]
    dtype: str
    units: str | None = None
    source_provenance: dict[str, Any] = Field(default_factory=dict) if _PYDANTIC else {}
    known_limitations: list[str] = Field(default_factory=list) if _PYDANTIC else []
    # Carries the actual numpy array — excluded from JSON serialisation.
    data: Any = Field(default=None, exclude=True) if _PYDANTIC else None

    if not _PYDANTIC:
        def __init__(self, **kwargs):
            self.data = kwargs.pop("data", None)
            for k, v in kwargs.items():
                setattr(self, k, v)
            if not hasattr(self, "source_provenance"):
                self.source_provenance = {}
            if not hasattr(self, "known_limitations"):
                self.known_limitations = []

        def model_dump(self):
            d = self.__dict__.copy()
            d.pop("data", None)  # never serialise raw arrays
            return d


class QortexTimeSeries(QortexAbstraction):
    """(n_channels, n_times) electrophysiology or sensor signal."""
    abstraction_type: str = "timeseries"
    channel_names: list[str] = Field(default_factory=list) if _PYDANTIC else []
    sampling_frequency_hz: float | None = None
    timebase: str | None = None           # "seconds_since_recording_start"
    reference: str | None = None          # "average" | "Cz" | "mastoid"

    if not _PYDANTIC:
        def __init__(self, **kwargs):
            self.channel_names = kwargs.pop("channel_names", [])
            self.sampling_frequency_hz = kwargs.pop("sampling_frequency_hz", None)
            self.timebase = kwargs.pop("timebase", None)
            self.reference = kwargs.pop("reference", None)
            super().__init__(**kwargs)


class QortexVolume(QortexAbstraction):
    """3D or 4D neuroimaging volume."""
    abstraction_type: str = "volume"
    voxel_sizes_mm: tuple[float, ...] | None = None
    affine: list[list[float]] | None = None   # 4×4, serialised as nested list
    coordinate_frame: str | None = None       # "RAS", "LPS", etc.
    tr_s: float | None = None                 # repetition time for fMRI
    n_volumes: int | None = None


class QortexImage(QortexAbstraction):
    """2D medical image."""
    abstraction_type: str = "image"
    pixel_spacing_mm: tuple[float, float] | None = None
    modality_tag: str | None = None  # DICOM Modality tag


class QortexEventTable(QortexAbstraction):
    """Tabular event annotations."""
    abstraction_type: str = "event_table"
    columns: list[str] = Field(default_factory=list) if _PYDANTIC else []
    n_events: int = 0


# ── Source Profile ─────────────────────────────────────────────────────────────

class ChannelSpec(BaseModel if _PYDANTIC else object):
    name: str
    index: int
    unit: str | None = None
    sampling_rate_hz: float | None = None
    channel_type: str | None = None  # "EEG" | "MEG" | "ECG" | ...

    if not _PYDANTIC:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self):
            return self.__dict__.copy()


class SourceProfile(BaseModel if _PYDANTIC else object):
    """Full description of what a data source provides."""

    source_id: str
    source_type: str                      # "local_edf", "bids", "lsl", ...
    path: str | None = None
    modality: Modality | str | None = None
    abstraction: str | None = None        # which QortexAbstraction subtype

    # For signal sources (EEG/MEG/iEEG)
    n_channels: int | None = None
    sampling_rate_hz: float | None = None
    channel_names: list[str] = Field(default_factory=list) if _PYDANTIC else []
    channel_specs: list[ChannelSpec] = Field(default_factory=list) if _PYDANTIC else []
    duration_s: float | None = None

    # For volumetric sources (NIfTI/DICOM)
    spatial_shape: tuple[int, ...] | None = None
    voxel_sizes_mm: tuple[float, ...] | None = None
    n_volumes: int | None = None
    tr_s: float | None = None
    affine: list[list[float]] | None = None
    axis_convention: AxisConvention | str | None = None

    # General
    dtype: str | None = None
    n_subjects: int | None = None
    available_suffixes: list[str] = Field(default_factory=list) if _PYDANTIC else []
    evidence_status: EvidenceStatus = EvidenceStatus.confirmed
    evidence: dict[str, EvidenceStatus | str] = Field(default_factory=dict) if _PYDANTIC else {}
    warnings: list[WarningItem] = Field(default_factory=list) if _PYDANTIC else []
    extra: dict[str, Any] = Field(default_factory=dict) if _PYDANTIC else {}

    if not _PYDANTIC:
        def __init__(self, **kwargs):
            self.channel_names = []
            self.channel_specs = []
            self.available_suffixes = []
            self.evidence_status = EvidenceStatus.confirmed
            self.evidence = {}
            self.warnings = []
            self.extra = {}
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self):
            return {k: v for k, v in self.__dict__.items()}


# ── Model Profile / Input / Output Contracts ──────────────────────────────────

class InputContract(BaseModel if _PYDANTIC else object):
    """What a model formally requires from its input."""

    modality: Modality | str
    axis_convention: AxisConvention | str

    # Signal
    required_channels: list[str] = Field(default_factory=list) if _PYDANTIC else []
    n_channels: int | None = None
    sampling_rate_hz: float | None = None
    window_duration_s: float | None = None

    # Volume
    spatial_shape: tuple[int, ...] | None = None
    voxel_sizes_mm: tuple[float, ...] | None = None
    n_slices: int | None = None

    # Common
    dtype: str = "float32"
    intensity_range: tuple[float, float] | None = None  # expected [min, max]
    batch_size: int | None = None

    # Evidence
    required_metadata: list[str] = Field(default_factory=list) if _PYDANTIC else []
    evidence_status: EvidenceStatus = EvidenceStatus.confirmed

    if not _PYDANTIC:
        def __init__(self, **kwargs):
            self.required_channels = []
            self.required_metadata = []
            self.evidence_status = EvidenceStatus.confirmed
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self):
            return self.__dict__.copy()


class OutputContract(BaseModel if _PYDANTIC else object):
    """Formal schema for model outputs."""

    output_type: str                   # "classification" | "segmentation" | ...
    classes: list[str] = Field(default_factory=list) if _PYDANTIC else []
    n_classes: int | None = None
    output_shape: tuple[int, ...] | None = None
    output_dtype: str = "float32"
    produces_probabilities: bool = True
    axis_convention: AxisConvention | str | None = None
    extra_outputs: list[str] = Field(default_factory=list) if _PYDANTIC else []

    if not _PYDANTIC:
        def __init__(self, **kwargs):
            self.classes = []
            self.extra_outputs = []
            self.produces_probabilities = True
            self.output_dtype = "float32"
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self):
            return self.__dict__.copy()


class ModelProfile(BaseModel if _PYDANTIC else object):
    """Full description of what a model expects and produces."""

    model_id: str
    provider: str                     # "huggingface" | "onnx" | "torch" | "custom"
    revision: str | None = None
    model_hash: str | None = None     # SHA-256 of weights file when available
    task: str | None = None           # "eeg_classification" | "segmentation" | ...
    license: str | None = None
    trusted: bool = False             # explicit trust required for remote code

    input_contract: InputContract | None = None
    output_contract: OutputContract | None = None

    estimated_params: int | None = None
    estimated_memory_mb: float | None = None
    supported_devices: list[str] = Field(default_factory=list) if _PYDANTIC else []
    warnings: list[WarningItem] = Field(default_factory=list) if _PYDANTIC else []

    if not _PYDANTIC:
        def __init__(self, **kwargs):
            self.supported_devices = []
            self.warnings = []
            self.trusted = False
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self):
            return self.__dict__.copy()


# ── Transform descriptor ───────────────────────────────────────────────────────

class TransformDescriptor(BaseModel if _PYDANTIC else object):
    """A single preprocessing transform in the planned chain."""

    kind: TransformKind | str
    required_by: str                  # model contract field that requires this
    params: dict[str, Any] = Field(default_factory=dict) if _PYDANTIC else {}
    reversible: bool = False
    irreversible_reason: str | None = None
    evidence_status: EvidenceStatus = EvidenceStatus.confirmed

    if not _PYDANTIC:
        def __init__(self, **kwargs):
            self.params = {}
            self.reversible = False
            self.evidence_status = EvidenceStatus.confirmed
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self):
            return self.__dict__.copy()

    def summary(self) -> str:
        kind = self.kind.value if hasattr(self.kind, "value") else self.kind
        params = ", ".join(f"{k}={v}" for k, v in self.params.items())
        rev = "reversible" if self.reversible else "irreversible"
        return f"{kind}({params})  [{rev}]  [required_by: {self.required_by}]"


# ── Preprocess Plan ────────────────────────────────────────────────────────────

class PreprocessPlan(BaseModel if _PYDANTIC else object):
    """Ordered, deterministic transform chain satisfying the model input contract.

    Every transform is linked to the model contract field that requires it.
    The plan must be stored in artifact provenance.
    """

    transforms: list[TransformDescriptor] = Field(default_factory=list) if _PYDANTIC else []
    has_destructive_transforms: bool = False
    warnings: list[WarningItem] = Field(default_factory=list) if _PYDANTIC else []
    unknowns: list[str] = Field(default_factory=list) if _PYDANTIC else []

    if not _PYDANTIC:
        def __init__(self, **kwargs):
            self.transforms = []
            self.warnings = []
            self.unknowns = []
            self.has_destructive_transforms = False
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self):
            return self.__dict__.copy()

    def summary(self) -> str:
        if not self.transforms:
            return "PreprocessPlan: no transforms required"
        lines = ["PreprocessPlan:"]
        for i, t in enumerate(self.transforms, 1):
            lines.append(f"  {i}. {t.summary()}")
        if self.warnings:
            lines.append(f"  Warnings: {len(self.warnings)}")
        if self.has_destructive_transforms:
            lines.append("  ⚠ Contains irreversible transforms")
        return "\n".join(lines)


# ── Compatibility Report ───────────────────────────────────────────────────────

class CompatibilityReport(BaseModel if _PYDANTIC else object):
    """Can the source satisfy the model? What transforms are needed?"""

    status: CompatibilityStatus
    source_id: str
    model_id: str

    required_transforms: list[TransformDescriptor] = Field(default_factory=list) if _PYDANTIC else []
    blockers: list[WarningItem] = Field(default_factory=list) if _PYDANTIC else []
    warnings: list[WarningItem] = Field(default_factory=list) if _PYDANTIC else []
    unknowns: list[str] = Field(default_factory=list) if _PYDANTIC else []
    evidence: list[dict[str, Any]] = Field(default_factory=list) if _PYDANTIC else []

    # Detailed dimension checks
    channel_match: EvidenceStatus = EvidenceStatus.unknown
    sampling_rate_match: EvidenceStatus = EvidenceStatus.unknown
    spatial_shape_match: EvidenceStatus = EvidenceStatus.unknown
    dtype_match: EvidenceStatus = EvidenceStatus.unknown
    axis_convention_match: EvidenceStatus = EvidenceStatus.unknown
    memory_estimate_mb: float | None = None

    if not _PYDANTIC:
        def __init__(self, **kwargs):
            self.required_transforms = []
            self.blockers = []
            self.warnings = []
            self.unknowns = []
            self.evidence = []
            self.channel_match = EvidenceStatus.unknown
            self.sampling_rate_match = EvidenceStatus.unknown
            self.spatial_shape_match = EvidenceStatus.unknown
            self.dtype_match = EvidenceStatus.unknown
            self.axis_convention_match = EvidenceStatus.unknown
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self):
            return self.__dict__.copy()

    def summary(self) -> str:
        st = self.status.value if hasattr(self.status, "value") else self.status
        lines = [f"CompatibilityReport: {st.upper()}"]
        lines.append(f"  source={self.source_id}  model={self.model_id}")
        if self.required_transforms:
            lines.append(f"  Required transforms ({len(self.required_transforms)}):")
            for t in self.required_transforms:
                lines.append(f"    • {t.summary()}")
        if self.blockers:
            lines.append(f"  BLOCKERS ({len(self.blockers)}):")
            for b in self.blockers:
                lines.append(f"    ✗ {b.message}")
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    ⚠ {w.message}")
        if self.unknowns:
            lines.append(f"  Unknowns: {', '.join(self.unknowns)}")
        return "\n".join(lines)

    def explain(self) -> list[dict[str, Any]]:
        """Return a structured source-vs-model compatibility explanation."""
        rows: list[dict[str, Any]] = []
        for item in self.evidence:
            row = dict(item)
            row.setdefault("severity", "info")
            rows.append(row)
        for transform in self.required_transforms:
            kind = transform.kind.value if hasattr(transform.kind, "value") else str(transform.kind)
            rows.append({
                "check": transform.required_by,
                "status": "transform_required",
                "transform": kind,
                "params": transform.params,
                "risk": "irreversible" if not transform.reversible else "reversible",
                "evidence_status": (
                    transform.evidence_status.value
                    if hasattr(transform.evidence_status, "value")
                    else str(transform.evidence_status)
                ),
            })
        for blocker in self.blockers:
            rows.append({
                "check": blocker.code,
                "status": "blocked",
                "message": blocker.message,
                "suggestion": blocker.suggestion,
                "severity": blocker.severity,
            })
        for warning in self.warnings:
            rows.append({
                "check": warning.code,
                "status": "warning",
                "message": warning.message,
                "suggestion": warning.suggestion,
                "severity": warning.severity,
            })
        for unknown in self.unknowns:
            rows.append({
                "check": str(unknown),
                "status": "unknown",
                "severity": "warning",
            })
        return rows

    def to_markdown(self) -> str:
        """Render a detailed compatibility table as Markdown."""
        lines = [
            "| Check | Status | Detail |",
            "|---|---|---|",
        ]
        for row in self.explain():
            check = str(row.get("check", ""))
            status = str(row.get("status", ""))
            detail_bits = []
            for key in ("source", "required", "transform", "risk", "message", "suggestion"):
                value = row.get(key)
                if value not in (None, "", []):
                    detail_bits.append(f"{key}={value}")
            lines.append(f"| {check} | {status} | {'; '.join(detail_bits)} |")
        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialize the report and detailed explanation to JSON."""
        import json
        payload = self.model_dump() if hasattr(self, "model_dump") else self.__dict__.copy()
        payload["explanation"] = self.explain()
        return json.dumps(_jsonable(payload), indent=2, ensure_ascii=False)

    @property
    def is_runnable(self) -> bool:
        st = self.status.value if hasattr(self.status, "value") else self.status
        return st in ("compatible", "compatible_with_transforms")


# ── Latency Report ─────────────────────────────────────────────────────────────

class LatencyBreakdown(BaseModel if _PYDANTIC else object):
    source_read_ms: float = 0.0
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    output_write_ms: float = 0.0
    total_ms: float = 0.0

    if not _PYDANTIC:
        def __init__(self, **kwargs):
            self.source_read_ms = kwargs.get("source_read_ms", 0.0)
            self.preprocess_ms = kwargs.get("preprocess_ms", 0.0)
            self.inference_ms = kwargs.get("inference_ms", 0.0)
            self.postprocess_ms = kwargs.get("postprocess_ms", 0.0)
            self.output_write_ms = kwargs.get("output_write_ms", 0.0)
            self.total_ms = kwargs.get("total_ms", 0.0)

        def model_dump(self):
            return self.__dict__.copy()


class LatencyReport(BaseModel if _PYDANTIC else object):
    """End-to-end timing breakdown per AGENT.md §17."""

    n_windows: int = 0
    n_dropped: int = 0
    budget_ms: float | None = None
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    breakdown: LatencyBreakdown = Field(default_factory=LatencyBreakdown) if _PYDANTIC else None
    status: Literal["PASS", "FAIL", "UNKNOWN"] = "UNKNOWN"

    if not _PYDANTIC:
        def __init__(self, **kwargs):
            self.n_windows = 0
            self.n_dropped = 0
            self.budget_ms = None
            self.p50_ms = 0.0
            self.p95_ms = 0.0
            self.p99_ms = 0.0
            self.mean_ms = 0.0
            self.breakdown = LatencyBreakdown()
            self.status = "UNKNOWN"
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self):
            return self.__dict__.copy()

    def summary(self) -> str:
        b = self.breakdown or LatencyBreakdown()
        lines = [
            f"Source read latency:    {b.source_read_ms:>8.1f} ms",
            f"Preprocessing latency:  {b.preprocess_ms:>8.1f} ms",
            f"Inference latency:      {b.inference_ms:>8.1f} ms",
            f"Postprocessing latency: {b.postprocess_ms:>8.1f} ms",
            f"Output write latency:   {b.output_write_ms:>8.1f} ms",
            f"End-to-end p50:         {self.p50_ms:>8.1f} ms",
            f"End-to-end p95:         {self.p95_ms:>8.1f} ms",
            f"End-to-end p99:         {self.p99_ms:>8.1f} ms",
        ]
        if self.budget_ms is not None:
            lines.append(f"Latency budget:         {self.budget_ms:>8.1f} ms")
        lines.append(f"Windows / dropped:      {self.n_windows} / {self.n_dropped}")
        lines.append(f"Status: {self.status}")
        return "\n".join(lines)


# ── Artifact Contract ──────────────────────────────────────────────────────────

class ArtifactContract(BaseModel if _PYDANTIC else object):
    """Provenance and schema contract stored with every Qortex artifact."""

    qortex_version: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    ) if _PYDANTIC else ""
    source_id: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    model_hash: str | None = None
    pipeline_spec_hash: str | None = None
    preprocessing_transforms: list[str] = Field(default_factory=list) if _PYDANTIC else []
    runtime_backend: str | None = None
    device: str | None = None
    output_schema: str | None = None
    output_type: str | None = None
    n_records: int | None = None
    warnings: list[dict[str, Any]] = Field(default_factory=list) if _PYDANTIC else []
    unknowns: list[str] = Field(default_factory=list) if _PYDANTIC else []
    compatibility_status: str | None = None
    leakage_check_applied: bool = False
    split_policy: str | None = None
    seed: int | None = None

    if not _PYDANTIC:
        def __init__(self, **kwargs):
            from datetime import datetime, timezone
            self.created_at = datetime.now(timezone.utc).isoformat()
            self.preprocessing_transforms = []
            self.warnings = []
            self.unknowns = []
            self.leakage_check_applied = False
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self):
            return self.__dict__.copy()


# ── Pipeline run report ───────────────────────────────────────────────────────

class PipelineRunReport(BaseModel if _PYDANTIC else object):
    """Combined result of a pipeline execution."""

    success: bool
    source_profile: SourceProfile | None = None
    model_profile: ModelProfile | None = None
    compatibility_report: CompatibilityReport | None = None
    preprocess_plan: PreprocessPlan | None = None
    latency_report: LatencyReport | None = None
    artifact_contract: ArtifactContract | None = None
    outputs: list[dict[str, Any]] = Field(default_factory=list) if _PYDANTIC else []
    errors: list[str] = Field(default_factory=list) if _PYDANTIC else []
    warnings: list[WarningItem] = Field(default_factory=list) if _PYDANTIC else []
    n_outputs_written: int = 0
    n_windows_processed: int = 0

    if not _PYDANTIC:
        def __init__(self, **kwargs):
            self.outputs = []
            self.errors = []
            self.warnings = []
            self.n_outputs_written = 0
            self.n_windows_processed = 0
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self):
            return self.__dict__.copy()


# ── Additional internal data abstractions ─────────────────────────────────────

from dataclasses import dataclass as _dataclass, field as _dc_field


@_dataclass
class QortexImageSeries:
    """Ordered sequence of 2D images (e.g., video frames, repeated scans)."""
    frames: Any                          # numpy [N, H, W, C] or [N, H, W]
    n_frames: int
    fps: float | None                    # None if not temporal
    frame_timestamps: list | None        # Unix epoch per frame
    shape: tuple                         # (n_frames, H, W, C) or (n_frames, H, W)
    axes: str                            # "nhwc" | "nchw" | "nhw"
    dtype: str                           # "uint8" | "float32" etc.
    units: str                           # "pixel_intensity" | "HU" etc.
    coordinate_frame: str | None         # None for 2D
    provenance: dict = _dc_field(default_factory=dict)


@_dataclass
class QortexVideo:
    """Video data stream with metadata."""
    frames: Any                          # numpy [N, H, W, C]
    n_frames: int
    fps: float
    duration_s: float
    width: int
    height: int
    n_channels: int                      # 1=gray, 3=RGB, 4=RGBA
    codec: str | None
    shape: tuple
    axes: str = "nhwc"
    dtype: str = "uint8"
    units: str = "pixel_intensity"
    provenance: dict = _dc_field(default_factory=dict)


@_dataclass
class QortexEmbeddingTable:
    """Collection of embedding vectors with metadata."""
    vectors: Any                         # numpy [N, D]
    n_items: int
    dimensionality: int
    item_ids: list = _dc_field(default_factory=list)
    model_id: str | None = None
    layer: str | None = None
    dtype: str = "float32"
    axes: str = "nd"
    units: str = "embedding_dim"
    provenance: dict = _dc_field(default_factory=dict)


@_dataclass
class QortexClinicalContext:
    """Clinical metadata context (FHIR/DICOM-derived, PHI-scrubbed)."""
    patient_id: str = ""                 # anonymized/pseudonymized
    study_date: str | None = None        # YYYYMMDD
    modality: str | None = None
    institution: str | None = None
    clinical_indication: str | None = None
    structured_reports: list = _dc_field(default_factory=list)
    measurements: dict = _dc_field(default_factory=dict)
    phi_redacted: bool = True
    provenance: dict = _dc_field(default_factory=dict)


@_dataclass
class QortexStream:
    """Live data stream descriptor (not the data itself)."""
    stream_id: str = ""
    stream_type: str = ""                # "eeg" | "markers" | "video" etc.
    source_type: str = ""                # "lsl" | "brainflow" | "websocket"
    n_channels: int = 0
    sampling_rate_hz: float | None = None
    channel_names: list = _dc_field(default_factory=list)
    channel_units: list = _dc_field(default_factory=list)
    is_live: bool = True
    buffer_size_s: float = 5.0
    provenance: dict = _dc_field(default_factory=dict)


def _jsonable(value: Any) -> Any:
    """Recursively convert contract objects to JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "__dict__"):
        return _jsonable(value.__dict__)
    return str(value)
