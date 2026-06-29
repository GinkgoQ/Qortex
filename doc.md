# Qortex — Code Review Reference

---

## Part 1: Code Review Methodology (Senior-Level)

### Mental Model First

Before reading a single line: understand the system's **contract boundary** — what enters, what exits, and what the invariants are at each boundary. In Qortex: `SourceProfile → CompatibilityReport → PreprocessPlan → PipelineRunReport → ArtifactContract`. Everything else is implementation.

### Review Order (priority-ranked)

**Tier 1 — Contract Layer** (review first, bugs here break everything downstream)
- Data model files: field names, types, defaults, optionality
- Abstract base classes: method signatures, what each must guarantee
- Exception hierarchy: completeness, correct inheritance

**Tier 2 — Decision Engines** (correctness-critical, branching logic)
- Compatibility engine: every check function, every evidence path
- Planner: transform ordering, auto-insert logic
- Spec validator: every field combination, security gates

**Tier 3 — Adapters** (common bugs: wrong field names, missing required fields, wrong type coercions)
- Source adapters: `SourceProfile` construction — `source_type`, `spatial_shape` as tuple, `channel_names` as list
- Model adapters: `ModelProfile`/`InputContract`/`OutputContract` field names
- Output adapters: `_n_written` counter, `write_marker()`, context manager protocol

**Tier 4 — Pipeline Orchestration** (sequencing bugs, missing pass-throughs)
- `Pipeline.check()`: passes `source_profile` and `model_provider` to planner?
- `Pipeline.run()`: model loaded before streaming starts?
- `Pipeline.replay()`: re-probes new source, rebuilds plan?

**Tier 5 — Provenance / Serialization** (silent data loss)
- `ArtifactWriter`: all 9 files written, SHA-256 manifest complete?
- Enum serialization: `.value` not the object itself in JSON output
- `_to_serialisable()`: handles all types recursively?

### What to Look For (field-level bugs)

| Anti-pattern | Why it fails |
|---|---|
| `axes="hwc"` (string) on `QortexAbstraction` | field is `list[str]`; Pydantic v2 rejects; v1 iterates chars |
| `voxel_sizes=` on `QortexVolume` | field is `voxel_sizes_mm` |
| `provenance=` on any `QortexAbstraction` | field is `source_provenance` |
| `sampling_rate_hz=` on `QortexTimeSeries` | field is `sampling_frequency_hz` |
| `source_type` missing on `SourceProfile` | required field, no default — Pydantic raises |
| `spatial_shape=[n,h,w]` (list) | declared as `tuple[int,...]` |
| `self.n_written = 0` in `OutputAdapter` subclass | `n_written` is a data descriptor (`@property`) — must use `self._n_written` |
| `AxisConvention.batch_channels_spatial` | does not exist; use `batch_channels_xyz` |
| `evidence={...}` on `InputContract`/`OutputContract` | no such field; use `evidence_status=EvidenceStatus.X` |
| Dynamic exception class creation | `type("ModelAdapterError", (QortexError,), {})` bypasses real hierarchy |

### Senior-Level Checks Beyond Correctness

**Security**
- Plugin/custom provider must require `trust_remote_code=True` before `importlib` execution
- PHI: DICOM adapter must never write patient tags to `SourceProfile`, logs, or provenance

**Behavioral invariants**
- `probe()` must be header-only — no full data load
- `inspect()` must not download weights
- `CompatibilityEngine` never loads a model
- `PreprocessPlanner` reads only from `CompatibilityReport` — no re-checking

**Transform ordering** (canonical: channel_select=1 → resample=4 → bandpass=5 → normalize=9 → cast=10 → to_tensor=13)
- `resample` before `normalize` (signal energy conservation)
- `channel_select` before `resample` (fewer channels = cheaper computation)
- `cast_dtype` last before tensor conversion
- `to_tensor` conditional on provider — HF/ONNX accept numpy natively

**EvidenceStatus propagation**
- `confirmed` = read directly from file header
- `inferred` = derived (e.g., srate from timestamps diff)
- `missing` = field absent; no-stream early returns
- `unknown` = live stream — not knowable before connection
- Compatibility check: `unknown` → `uncertain`, not blocker

---

## Part 2: Directory Map

```
src/qortex/
├── core/               # Shared foundation: exceptions, entities, config
├── _internal/          # Private utilities: glob, hashing, progress bars
├── neuroai/            # NeuroAI Runtime — the primary system
│   ├── spec.py         # PipelineSpec + sub-specs (YAML ↔ dataclass)
│   ├── contracts.py    # All typed data contracts (SourceProfile, ModelProfile, …)
│   ├── compatibility.py # CompatibilityEngine — source vs model check
│   ├── pipeline.py     # Pipeline facade — check/run/benchmark/replay
│   ├── artifact.py     # ArtifactWriter — 9-file provenance directory
│   ├── benchmark.py    # PipelineProfiler — p50/p95/p99 per-stage timing
│   ├── preprocess/     # PreprocessPlanner + TransformExecutor
│   ├── runtime/        # RuntimeEngine — streaming execution loop
│   ├── sources/        # SourceAdapter implementations (one file per format)
│   ├── models/         # ModelAdapter implementations (one file per provider)
│   └── outputs/        # OutputAdapter implementations (one file per sink)
├── client/             # OpenNeuro GraphQL + HTTP transport
├── catalog/            # Dataset search, index, refresh
├── fetch/              # Download engine (HTTP + DataLad backends)
├── manifest/           # BIDS manifest: build, diff, sidecar, graph
├── parse/              # Modality parsers: EEG, MEG, MRI, fMRI, DWI, PET, fNIRS, iEEG
├── validation/         # BIDS validator integration, diff, cache
├── convert/            # EDF/NIfTI → Parquet/HDF5/Zarr/WebDataset/TFRecord
├── train/              # Framework bridges: HF, PyTorch, Lightning, sklearn, Ray, Dask
├── visualize/          # Modality-specific viewers + visual audit
├── eda/                # Event analysis, signal quality, EDA reports
├── inspect/            # Dataset profiling, label landscape, fitness scoring
├── qc/                 # QC filters and masks
├── cohort/             # Cohort builder, federated subjects, data cards
├── export/             # MONAI + TorchIO integration
├── harmonize/          # Cross-dataset harmonization reporter
├── derivatives/        # BIDS derivatives indexer
├── stream/             # NIfTI + EDF streaming (non-NeuroAI path)
├── runtime/            # BIDSImageDataset, BIDSSignalDataset, epoch loaders
├── indexing/           # Local file index builder
├── lake/               # Data lake layout, mount, registry
├── plan/               # Download planning, locking, selective fetch
└── check/              # Readiness checks (can-train gate)
```

**Key distinction**: `qortex.neuroai` = live inference pipeline. `qortex` (top-level) = dataset discovery, download, conversion, EDA, training prep. They share `qortex.core` only.

---

## Part 3: Core Layer (`core/`)

### `exceptions.py` — Full hierarchy

```
QortexError
├── APIError            status_code: int|None
│   └── RateLimitError  retry_after: float|None
├── AuthError
├── NetworkError
├── DatasetNotFoundError  dataset_id: str
├── SnapshotNotFoundError dataset_id, tag, available: list[str]
├── ManifestError
├── ValidationError
├── ConversionError
├── ReadinessError
├── DownloadError       path, url, reason
├── CacheError
├── ConfigurationError
├── IntegrityError      path, expected, got, check
├── SourceAdapterError  source_type, path
├── ModelAdapterError   model_id, provider: str
├── OutputAdapterError
├── RuntimeExecutionError  stage: str
└── ContractValidationError contract_type, violations
```

Catch all with `except QortexError`. All carry structured context attributes beyond the message string:

```python
try:
    ...
except QortexError as exc:
    print(exc.code)
    print(exc.context)
    print(exc.to_dict())
```

Warnings use `QortexWarning`, `WarningRecord`, and `emit_warning(...)` when a condition is non-fatal but should still be visible in logs/notebooks. Use this for recoverable degradations, optional dependency fallbacks, skipped files, and uncertainty that does not block execution.

### `entities.py` — Shared domain objects

`FileRecord` (path, size, url, checksum, tag, modality), `Manifest` (dataset_id, tag, files, subject map, session map), `DatasetRef` / `SnapshotRef` (lightweight API refs), `DownloadResult`, `ConversionResult`, `ReadinessReport`, `ValidationReport`, `ValidationDiff`, `EDAReport`, `EventLabelSummary`, `SelectionSpec`, `LocalIndexReport`, `FilePreview`.

These are pure data; no logic beyond `__init__` and basic computed properties.

### `config.py` — Single config object

`QortexConfig`: `api_token`, `cache_dir`, `openneuro_endpoint`, `gql_endpoint`, `max_concurrent_downloads`, `max_concurrent_heads`, `max_retries`, retry backoff, timeout fields, integrity flags, and `exclude_derivatives_default`.

Set runtime overrides via:

```python
import qortex

qortex.configure(api_token="...", max_concurrent_downloads=16, cache_dir="~/qortex-cache")
cfg = qortex.get_config()
```

Use `get_config()` for the active singleton. `QortexConfig()` creates a fresh config from defaults/environment and does not inherit runtime overrides. Overrides are validated; invalid fields or invalid values raise `ConfigurationError`. Use `cfg.redacted()` for logs or reports so tokens are never printed.

---

## Part 4: NeuroAI Runtime (`neuroai/`)

### Data Flow (strict one-way)

```
PipelineSpec.from_yaml()
    └── Pipeline.check()
            ├── make_source_adapter() → adapter.probe() → SourceProfile
            ├── make_model_adapter() → adapter.inspect() → ModelProfile
            ├── CompatibilityEngine.check() → CompatibilityReport
            └── PreprocessPlanner.build_plan() → PreprocessPlan
    └── Pipeline.run()
            ├── adapter.load(runtime_spec)           ← weights loaded here
            ├── RuntimeEngine.run()
            │   └── for window in source.stream():
            │           TransformExecutor.apply(window, plan)
            │           model.predict(window) → ModelOutput
            │           TriggerSpec.evaluate(prediction) → bool
            │           for out in outputs: out.write(ModelOutput)
            │           [if trigger] out.write_marker(EventMarkerOutput)
            └── ArtifactWriter.write(artifact_dir)
```

### `spec.py` — `PipelineSpec` and sub-specs

All are `@dataclass`. `from_yaml()` → `from_dict()` → construct. `to_dict()` → `to_yaml()`. `content_hash()` = SHA-256 of canonical JSON (sorted keys).

**Sub-specs:**
- `SourceSpec`: type, path, query (LSL filter), subjects, sessions, modality, suffix, extra. Accepts scalar aliases `subject` and `session`, then normalizes them to lists.
- `WindowSpec`: duration_s, step_s, overlap_frac, tmin, event_aligned, drop_short. Parses `"2s"` / `"500ms"` strings and accepts both `duration_s` / `step_s` and serialized `duration` / `step`.
- `ModelSpec`: provider, id, task, revision, trust_remote_code, extra
- `PreprocessSpec`: mode (auto/explicit/none), allow, deny, normalize, resample, channel_select. `allows(kind)` first enforces mode and boolean gates, then `deny`, then `allow`. `normalize=False`, `resample=False`, and `channel_select=False` are real policy blockers, not documentation-only fields.
- `RuntimeSpec`: device (auto/cpu/cuda/mps), latency_budget_ms, optimize (safe/speed/memory), batch_size, fp16, cache_model
- `OutputSpec`: type, path, stream_name, append, extra
- `TriggerSpec`: when (class, probability_gte, stable_for), emit. `evaluate(prediction_dict)` → bool

`PipelineSpec.validate()` returns `list[str]` errors. Checks: required fields, known providers, plugin security gate, file existence, output types, window timing, batch_size > 0, transform name validity, allow/deny overlap, boolean-gate contradictions, trigger completeness, trigger probability range, and `stable_for > 0`.

`Pipeline.from_yaml()` and `Pipeline.from_dict()` convert parse/validation failures to `ContractValidationError`, preserving all violations in structured context. Review rule: user-facing constructors should not leak raw `TypeError`, `ValueError`, or string-scraped validation output.

Boolean parsing is explicit. `"false"`, `"0"`, `"no"`, and `"off"` are false; `"true"`, `"1"`, `"yes"`, and `"on"` are true. Never use `bool(value)` for YAML/JSON config booleans.

### `contracts.py` — The type system

**Enums:**
- `EvidenceStatus`: confirmed / inferred / missing / unknown / blocked
- `CompatibilityStatus`: compatible / compatible_with_transforms / uncertain / incompatible
- `TransformKind`: resample, channel_select, channel_reorder, channel_map, bandpass, normalize, window, cast_dtype, rescale_intensity, reorient, resample_spatial, pad_or_crop, add_batch_dim, add_channel_dim, to_tensor
- `AxisConvention`: RAS, LAS, LPS, spatial_zyx, spatial_xyz, channels_first, channels_last, time_channels, channels_time, batch_channels_time, batch_channels_xyz (`batch_channels_spatial` does NOT exist)
- `Modality`: eeg, meg, ieeg, fnirs, mri, fmri, dwi, pet, dicom, image, video, tabular, signal, unknown

**Data abstractions** (inherit `QortexAbstraction`):
- `QortexAbstraction`: abstraction_type, shape `tuple[int,...]`, `axes: list[str]`, dtype, units, `source_provenance: dict`, known_limitations, `data: Any` (excluded from JSON)
- `QortexTimeSeries(QortexAbstraction)`: channel_names, `sampling_frequency_hz`, timebase, reference
- `QortexVolume(QortexAbstraction)`: voxel_sizes_mm, affine (4×4 list), coordinate_frame, tr_s, n_volumes
- `QortexImage`, `QortexEventTable`, `QortexImageSeries`, `QortexVideo`, `QortexEmbeddingTable`, `QortexClinicalContext`, `QortexStream`

**Field name rules** (critical — wrong names silently ignored or crash):
```
QortexTimeSeries:   sampling_frequency_hz  NOT sampling_rate_hz
QortexAbstraction:  source_provenance      NOT provenance
QortexAbstraction:  axes: list[str]        NOT axes: str
QortexVolume:       voxel_sizes_mm         NOT voxel_sizes
SourceProfile:      source_type: str       REQUIRED, no default
SourceProfile:      spatial_shape: tuple   NOT list
SourceProfile:      channel_names: list[str]  NOT None
```

**`SourceProfile`** — from `probe()`:
```
source_id, source_type (required), path, modality, abstraction
n_channels, sampling_rate_hz, channel_names: list[str], channel_specs: list[ChannelSpec]
duration_s, spatial_shape: tuple, voxel_sizes_mm: tuple, n_volumes, tr_s, affine
axis_convention, dtype, n_subjects, available_suffixes
evidence_status: EvidenceStatus, evidence: dict[str, EvidenceStatus], warnings, extra
```

**`ModelProfile`** — from `inspect()`:
```
model_id, provider, revision, model_hash, task, license, trusted
input_contract: InputContract|None, output_contract: OutputContract|None
estimated_params: int|None, estimated_memory_mb: float|None
supported_devices: list[str], warnings: list[WarningItem]
```
(NOT: n_parameters, input_shape, output_shape, framework, extra, modality, n_outputs, label_map)

**`InputContract`**:
```
modality, axis_convention,
required_channels: list[str], n_channels: int|None
sampling_rate_hz: float|None, window_duration_s: float|None
spatial_shape: tuple|None, voxel_sizes_mm: tuple|None
dtype: str, intensity_range: tuple|None, batch_size: int|None
required_metadata: list[str], evidence_status: EvidenceStatus
```
(NOT: evidence={})

**`OutputContract`**:
```
output_type, classes: list[str], n_classes: int|None
output_shape: tuple|None, output_dtype: str
produces_probabilities: bool, axis_convention, extra_outputs: dict
```
(NOT: class_labels, evidence)

**`CompatibilityReport`**:
```
status: CompatibilityStatus, source_id, model_id
required_transforms: list[TransformDescriptor]
blockers: list[WarningItem], warnings: list[WarningItem], unknowns: list[str]
evidence: list[dict]
is_runnable: bool  (= status in {compatible, compatible_with_transforms})
summary() → str   (starts with "CompatibilityReport: COMPATIBLE…")
```

**`PreprocessPlan`**: ordered `transforms: list[TransformDescriptor]`, `summary() → str`

**`TransformDescriptor`**: kind, params: dict, reason, reversible: bool, affects_field

**`LatencyReport`**: per-stage breakdowns `(source_read, preprocess, inference, postprocess, output_write)`, each with `p50_ms, p95_ms, p99_ms`. `budget_ms`, `n_windows`, `n_dropped`. `summary()`.

**`PipelineRunReport`**: n_windows, n_ok, n_errors, duration_s, latency_report, artifact_contract

**`ArtifactContract`**: artifact_dir, run_id, pipeline_hash, source_id, model_id, n_outputs

### `compatibility.py` — `CompatibilityEngine`

`check(source, model, preprocess) → CompatibilityReport`

Per-check functions (each appends to `transforms`, `blockers`, `warnings`, `unknowns`, `evidence`):
- `_check_modality` — exact match or known alias
- `_check_channels` — n_channels match; insert `channel_select` if src > req; blocker if src < req
- `_check_sampling_rate` — insert `resample` if mismatch AND preprocess allows; blocker if not allowed
- `_check_spatial` — shape + voxel; insert `resample_spatial` + `pad_or_crop`
- `_check_dtype` — insert `cast_dtype`
- `_check_axis_convention` — insert `reorient` for LPS↔RAS; skip generic warning when `_SPATIAL_FRAMES` pair (handled by `_check_coordinate_frame`)
- `_check_coordinate_frame` — DICOM LPS→RAS specific message + `reorient` transform
- `_check_memory` — estimate vs `estimated_memory_mb`; warning only
- `_check_required_metadata` — missing required fields → blocker

Status logic: any blocker → `incompatible`; any transform → `compatible_with_transforms`; any unknown → `uncertain`; otherwise `compatible`.

`_SPATIAL_FRAMES = {"LPS", "RAS", "LAS", "SPATIAL_ZYX", "SPATIAL_XYZ"}` — prevents duplicate warnings for the same coordinate pair.

### `preprocess/planner.py` — `PreprocessPlanner` + `TransformExecutor`

`build_plan(compat_report, *, window_duration_s, source_profile, model_provider) → PreprocessPlan`

Takes `compat_report.required_transforms`, sorts by `_TRANSFORM_ORDER`, then adds
runtime structural transforms:
- `to_tensor` at the end when absent. It emits numpy for HF/ONNX-style providers and Torch tensors otherwise.
- `window` when `window_duration_s` is present and no window transform is already required.

It does not insert modality heuristics such as DICOM HU normalization or EEG
bandpass by itself. Those transforms must come from the model contract and
`CompatibilityReport`.

`TransformExecutor.apply(data, plan) → data` — applies transforms in order. Each transform is implemented as a pure function operating on numpy arrays or QortexAbstraction objects.

**Canonical transform order:**
```
channel_select=1, channel_map=2, channel_reorder=3
resample=4, resample_spatial=4
bandpass=5, pad_or_crop=6, reorient=7
rescale_intensity=8, normalize=9, cast_dtype=10
add_batch_dim=11, add_channel_dim=12, to_tensor=13, window=14
```

`reorient` performs actual coordinate flip (not a no-op): `data = data[::-1, :, :]` for z-axis LPS→RAS flip.

### `pipeline.py` — `Pipeline` facade

State machine: `_checked: bool`, `_source_profile`, `_model_profile`, `_compat_report`, `_preprocess_plan`, `_model_adapter`, `_source_adapter`.

**`check()`** sequence:
1. `make_source_adapter(spec.source)` → `adapter.probe()` → `_source_profile`
2. `make_model_adapter(spec.model)` → `adapter.inspect()` → `_model_profile`
3. `CompatibilityEngine().check(source, model, preprocess)` → `_compat_report`
4. `PreprocessPlanner().build_plan(report, source_profile=_source_profile, model_provider=spec.model.provider)` → `_preprocess_plan`

**`run(artifact_dir=None)`**: loads model, creates output adapters, runs `RuntimeEngine`, calls `ArtifactWriter.write()` if `artifact_dir` set.

**`replay(source_path, speed=1.0)`**: re-probes new source (re-builds `_source_profile` and `_preprocess_plan`); does not reload model if already loaded.

**`benchmark(n_windows=100)`**: calls `PipelineProfiler` without writing outputs; returns `LatencyReport`.

### `runtime/engine.py` — `RuntimeEngine`

Inner loop:
```python
for data in source.stream():
    data = executor.apply(data, plan)
    output = model.predict(data)
    prediction_dict = output_to_dict(output)
    if trigger and trigger.evaluate(prediction_dict):
        _trigger_streak += 1
        if _trigger_streak >= required_stable:
            for out in outputs:
                out.write_marker(EventMarkerOutput(...))
    else:
        _trigger_streak = 0
    for out in outputs:
        out.write(output)
```

All stages timed by `PipelineProfiler`. Errors per-window → append to `errors`; do not halt loop (fault-tolerant streaming).

### `artifact.py` — `ArtifactWriter`

Writes 9 files on `write(artifact_dir)`:
```
provenance.json           — lineage: spec, source_id, model_id, timestamps, qortex_version
compatibility_report.json — CompatibilityReport.model_dump()
preprocess_plan.json      — PreprocessPlan.model_dump()
runtime_report.json       — PipelineRunReport.model_dump()
latency_report.json       — LatencyReport.model_dump()
warnings.json             — all warnings + unknowns
pipeline.yaml             — PipelineSpec.to_yaml()
artifact_contract.json    — ArtifactContract.model_dump()
artifact_manifest.json    — {file: {sha256, size}} for all 9 files
```

`_to_serialisable(obj)`: recursively converts → None/bool/int/float/str/dict/list. Handles `model_dump()`, `__dict__`, `.value` (Enum).

`_sha256_file(path)` → hex string; used for manifest integrity entries.

### `benchmark.py` — `PipelineProfiler`

`start_*/end_*` pairs per stage: source_read, preprocess, inference, postprocess, output_write.
`commit_window()` — stores `_WindowTiming` and resets current.
`report()` → `LatencyReport` with p50/p95/p99 via `statistics.quantiles`.
`budget_ms` → `LatencyReport.budget_exceeded` = `p95 > budget_ms`.

---

## Part 5: Source Adapters (`neuroai/sources/`)

### `_base.py` — `SourceAdapter` ABC

Required: `probe() → SourceProfile`, `read_batch() → list[QortexData]`, `stream() → Iterator[QortexData]`
Optional: `replay(speed=1.0)` — default calls `stream()` with sleep-based timing

### `_ring_buffer.py` — Lock-free windowing

`get_ring_buffer(n_channels, capacity, window_size, step_size)` — returns Rust extension if available, else Python fallback. `push(array)` / `pop_window()` → `ndarray | None`. Used by LSL and BrainFlow for producer-consumer gap between chunk delivery and window size.

### `_registry.py` — `make_source_adapter(spec, window_spec) → SourceAdapter`

Routes by `spec.type`:
```
local_file / edf / bdf / fif  → LocalFileAdapter
bids                           → BIDSSourceAdapter
dicom / dicom_folder           → DICOMFolderAdapter
dicomweb                       → DICOMWebAdapter
nwb                            → NWBAdapter
xdf                            → XDFAdapter
lsl                            → LSLSourceAdapter
brainflow                      → BrainFlowAdapter
image / img                    → ImageVideoAdapter
video                          → ImageVideoAdapter
```
Auto-detection: if `type` omitted, routes by extension (`.nwb`, `.xdf`, `.edf`, image/video exts).

### `local.py` — `LocalFileAdapter`

Formats: EDF/BDF/FIF/SET/VHDR (via MNE), NIfTI (nibabel), Parquet/CSV/TSV (Polars).
- `_probe_signal()` → MNE raw info; no data load (`preload=False`)
- `_probe_volume()` → nibabel header only
- `_stream_windowed()` → sliding windows via `raw.get_data(tmin, tmax)`, respects `step_s`, `overlap_frac`, `drop_short`
- `_load_volume()` → `nib.as_closest_canonical()` — always reorients to RAS

Returns `QortexTimeSeries` with `axes=["channels", "times"]`, `sampling_frequency_hz`, `source_provenance`.

### `bids.py` — `BIDSSourceAdapter`

`probe()` reads `dataset_description.json`, probes first signal file (MNE raw info) + first NIfTI (nibabel). Discovers subjects by `sub-*` dirs, modalities by subdir names.
`_collect_target_files()` — respects `spec.modality`, `spec.subjects`, `spec.suffix`.
`stream()` delegates to `LocalFileAdapter` per file.

PHI: reads only BIDS sidecar (never DICOM tags). No PHI in `SourceProfile`.

### `dicom.py` — `DICOMFolderAdapter`

Groups `.dcm` files by `SeriesInstanceUID`. Sorts by `InstanceNumber` or `ImagePositionPatient.z`.
Builds 4×4 affine from `ImageOrientationPatient`, `ImagePositionPatient`, `PixelSpacing`, `SliceThickness`.
Applies `RescaleSlope * pixel + RescaleIntercept` → Hounsfield units.

PHI redaction: `PatientName`, `PatientID`, `PatientBirthDate`, `PatientSex`, `PatientAge`, `PatientAddress`, `ReferringPhysicianName`, `InstitutionName` — never written anywhere. `extra["phi_redacted"] = True` flag confirms.

Returns `QortexVolume(axes=["z","y","x"], coordinate_frame="patient_lps", units="HU")`.
`SourceProfile.axis_convention = AxisConvention.spatial_zyx`.

### `dicomweb.py` — `DICOMWebAdapter`

QIDO-RS for metadata, WADO-RS for pixel data. Auth: bearer or basic from `spec.extra["auth"]`.
URL pattern: `{base}/studies/{study_uid}/series/{series_uid}/instances`.

Returns `QortexVolume(axes=["z","y","x"], coordinate_frame="patient_lps")`.

### `nwb.py` — `NWBAdapter`

Opens with `pynwb.NWBHDF5IO`. Finds `ElectricalSeries` in acquisition group.
NWB stores `[T, Ch]` → transposes to `[Ch, T]`.
`_get_srate()` tries `.rate`, `.sampling_rate` attrs, then infers from `timestamps` diffs.
`_get_channel_names()` reads from `electrodes["label"]`.

Returns `QortexTimeSeries(axes=["channels", "times"], sampling_frequency_hz, source_provenance)`.

### `xdf.py` — `XDFAdapter`

`pyxdf.load_xdf(select_streams=[])` for fast header-only probe.
`_select_streams()` filters by `spec.query["type"]` and/or `spec.query["name"]` (case-insensitive substring).
XDF stores `[T, Ch]` → transposes. `_extract_channel_names()` parses `info.desc.channels.channel.label`.
`replay(speed)` computes `sleep_s = win_size / srate / speed`.

Returns `QortexTimeSeries(axes=["channels", "time"], sampling_frequency_hz, source_provenance)`.

### `lsl.py` — `LSLSourceAdapter`

`pylsl.resolve_streams(wait_time=5.0)`. `_filter_streams()` by type/name.
`read_batch()` collects for `spec.extra["duration_s"]` seconds via `pull_chunk`.
`stream()` uses ring buffer: `push(chunk.T)` → `pop_window()`.
`replay()` warns "live source" and falls back to `stream()`.

### `brainflow.py` — `BrainFlowAdapter`

`probe()` uses `BoardShim.get_eeg_channels/get_sampling_rate/get_eeg_names` without opening session.
`stream()` opens session, loops `time.sleep(win_dur/4)` + `board.get_board_data(win_samples)`, uses ring buffer.
Default board: `BoardIds.SYNTHETIC_BOARD` — safe for testing without hardware.

### `image.py` — `ImageVideoAdapter`

Images: PIL/Pillow. Videos: OpenCV.
`_probe_image()` → reads width/height/mode from single file without decoding pixels.
`_stream_video()` — batches `win_frames = int(duration_s * fps)` frames.
`_load_image()` → `QortexVolume(axes=["h","w","c"], source_provenance)`.
`_load_video()` → `QortexVolume(axes=["n","h","w","c"], source_provenance)`.

---

## Part 6: Model Adapters (`neuroai/models/`)

### `_base.py` — `ModelAdapter` ABC + `ModelOutput`

Required: `inspect()→ModelProfile`, `required_input()→InputContract`, `output_schema()→OutputContract`, `load(runtime)→None`, `predict(batch)→ModelOutput`.

`ModelOutput` fields: output_type, raw, class_name, class_index, probabilities: dict[str,float], bbox, mask, embedding, regression_value, metadata.

### `_registry.py` — `make_model_adapter(spec) → ModelAdapter`

Routes by `spec.provider`. Unknown provider → `ModelAdapterError`.

### `huggingface.py` — `HuggingFaceModelAdapter`

`inspect()` → `AutoConfig.from_pretrained(id, revision=...)` (no weights). Extracts labels from `config.id2label`.
`load()` → `AutoModel.from_pretrained(id, revision=...)`. Raises real `ModelAdapterError` (not dynamic class).
`predict()` → forward pass; applies softmax for classification.

### `onnx.py` — `ONNXModelAdapter`

`inspect()` → `onnx.load_model()` (reads graph, no inference session created).
`load()` → `onnxruntime.InferenceSession(providers=...)`. Provider selection: `cuda:N` → `CUDAExecutionProvider`.
`predict()` → `session.run(output_names, {input_name: float32_array})`.

### `torch.py` — `TorchModelAdapter`

`inspect()` → SHA-256 hash of file, infers n_params from `.parameters()` if loadable without weights.
`load()` → `torch.jit.load()` for TorchScript; `torch.load(weights_only=False)` for `.pt`. Applies `.half()` for fp16+cuda.
`predict()` → 1D output → softmax → `ClassificationOutput`; multi-dim → argmax → segmentation.

`ModelProfile`: `estimated_params=n_params` (NOT `n_parameters`). No `extra={}` field.
`InputContract`: `evidence_status=EvidenceStatus.inferred` (NOT `evidence={}`).
`OutputContract`: `output_type=..., n_classes=None` (NOT `class_labels={}, evidence={}`).

### `braindecode.py` — `BraindecodelAdapter`

Loads `config.json` from HF Hub. Maps model names to braindecode classes:
```
eegnet → EEGNetv4
shallowfbcspnet → ShallowFBCSPNet
deepfbcspnet → Deep4Net
eegconformer → EEGConformer
tidnet → TIDNet
```
Unknown → `AutoModel.from_pretrained()`. Input always `[batch, channels, time]`. Applies softmax.

`evidence_status = confirmed if n_channels else unknown`.

### `monai.py` — `MONAIBundleAdapter`

Bundle resolution: local dir → local ZIP (extract to tempdir) → MONAI Hub download.
`inspect()` reads `configs/metadata.json` + `configs/inference.json`.
`load()` → `monai.bundle.ConfigParser` for network, `torch.load` for weights.
`predict()` → `monai.inferers.sliding_window_inference(roi_size=(96,96,96))` for volumes.

`AxisConvention.batch_channels_xyz` (NOT `batch_channels_spatial`).

### `ultralytics.py` — `UltralyticsModelAdapter`

YOLO always loads weights at init. Task dispatch: detect→`DetectionOutput`, segment→`SegmentationOutput`, classify→`ClassificationOutput`.
Input: numpy CHW float → HWC uint8 (YOLO expects BGR uint8).
Box parsing: `result.boxes.xyxy`, `.conf`, `.cls`.

`spatial_shape=(640,640)` (H,W only, not including channel dim). `AxisConvention.batch_channels_xyz`.

### `plugin.py` — `PluginModelAdapter`

Security gate: `trust_remote_code` must be True else raises `ModelAdapterError` before any file access.
`importlib.util.spec_from_file_location` → validates `QortexPlugin` class has all required methods.
All calls wrapped in try/except → structured `ModelAdapterError`.

---

## Part 7: Output Adapters (`neuroai/outputs/`)

### `_base.py` — `OutputAdapter` ABC

```python
_n_written: int = 0        # class-level default
n_written: @property       # data descriptor — blocks instance assignment
write_marker(marker) → None  # default no-op
__enter__/__exit__ → open/close
```

**Critical**: subclasses must use `self._n_written += 1`, never `self.n_written += 1`.

### `types.py` — Canonical output dataclasses

`ClassificationOutput`: class_name, class_index, confidence, probabilities, label_map
`DetectionOutput`: boxes: list[BoundingBox], class_names, confidences. `BoundingBox`: x1,y1,x2,y2,conf,cls
`SegmentationOutput`: mask: ndarray, class_labels: dict[int,str], n_classes
`RegressionOutput`: value, unit
`EmbeddingOutput`: vector, dimension, model_layer
`TimeSeriesPredictionOutput`: values, timestamps_s, channel_names
`EventMarkerOutput`: event_type, label, confidence, timestamp_s, metadata
`VolumePredictionOutput`: volume: ndarray, affine, class_labels, label_type
`ReportOutput`: title, findings: list[str], confidence: float (NOT str)

### `_registry.py` — `make_output_adapter(spec) → OutputAdapter`

Routes by `spec.type`:
```
jsonl / json_lines   → JSONLOutputAdapter
parquet              → ParquetOutputAdapter
csv                  → CSVOutputAdapter
lsl_marker / lsl     → LSLMarkerOutputAdapter
nifti / nii          → NIfTIOutputAdapter
dicom_seg            → DICOMSegOutputAdapter
dicom_sr             → DICOMSROutputAdapter
bids / bids_derivative → BIDSOutputAdapter
coco / coco_json     → COCOOutputAdapter
yolo / yolo_txt      → YOLOOutputAdapter
websocket / ws       → WebSocketOutputAdapter
http / webhook       → HTTPOutputAdapter
overlay              → OverlayOutputAdapter
```

### Adapter matrix

| Adapter | write_marker | special |
|---|---|---|
| `jsonl_out.py` | yes — appends `{"type":"marker",...}` | streaming; flush per write |
| `parquet_out.py` | no | batches in memory; writes on close |
| `csv_out.py` | no | `csv.DictWriter`; header on open |
| `lsl_out.py` | yes — pushes string chunk | `pylsl.StreamOutlet`; `stream_name` from spec |
| `nifti_out.py` | no | writes NIfTI per volume prediction; requires nibabel |
| `dicom_seg_out.py` | no | DICOM-SEG structured report |
| `dicom_sr_out.py` | no | DICOM-SR free text / structured |
| `bids_out.py` | no | BIDS derivative: `derivatives/qortex/sub-X/...` |
| `coco_out.py` | no | accumulates; writes single JSON on close |
| `yolo_out.py` | no | one `.txt` per image; `cls cx cy w h` format |
| `websocket_out.py` | no | `websockets.connect()`; async in thread |
| `http_out.py` | no | POST JSON per prediction; retries on 429/5xx |
| `overlay_out.py` | no | PIL overlay; writes PNG per frame |

All adapters: `_n_written` counter, context manager protocol, `open()`/`close()` explicit.

---

## Part 8: Dataset / OpenNeuro Layer

### `client/`

**`transport.py`** — `SyncTransport` + `AsyncTransport`. `httpx.Client` with retry/backoff on `RETRYABLE_CODES` (408,429,500,502,503,504,522,524). `_build_ssl_context()` tries `truststore` first (system CA store), falls back to `ssl.create_default_context()`. `USER_AGENT = f"qortex/{__version__}"`. Raises `RateLimitError(retry_after=float)` on 429.

**`auth.py`** — `resolve_token()` checks env `QORTEX_TOKEN` → `get_config().api_token` → raises `AuthError`.

**`graphql.py`** — `OpenNeuroClient`. All queries use GraphQL variables (not string interpolation). Methods:
- `get_dataset / get_dataset_rich` → `DatasetRef` / `RichDatasetInfo`
- `get_snapshots / get_snapshot / get_latest_snapshot`
- `get_files(dataset_id, tag)` → `(SnapshotRef, list[dict])`
- `search_datasets / search_datasets_rich`

Rich metadata: `snapshot.summary` gives subjects/sessions/tasks/modalities + `subjectMetadata` (age, sex, group) without downloading any files. `snapshot.hexsha` for cache invalidation.

**`remote.py`** — async download helpers used by fetch engine.

### `catalog/`

**`search.py`** — `DatasetQuery(modalities, tasks, subjects_min/max, age_range, license, ...)` → `PagedResults`. `facets()` → available modality/task values. `live_search()` — streams results.

**`index.py`** — catalog index (local SQLite or in-memory) for fast query without hitting API.

**`refresh.py`** — periodic catalog refresh; delta sync via `hexsha` comparison.

### `fetch/`

**`engine.py`** — `FetchEngine.download(manifest, dest, *, include, exclude, resume)`. Async HTTP download with progress bars. Calls `apply_include_exclude()` for file filtering.

**`backends/`** — `_base.py` = `BackendBase` ABC. `http.py` = direct HTTPS from OpenNeuro CDN. `datalad.py` = DataLad-based backend (optional; slower but version-controlled).

**`cache.py`** — cache dir management; content-addressed by SHA-256; skip re-download if checksum matches.

### `manifest/`

**`builder.py`** — builds `Manifest` from API file list or local directory scan.
**`bids.py`** — BIDS-specific manifest logic: sidecar association, entity extraction.
**`diff.py`** — `ManifestDiff`: new, removed, changed files between two manifests.
**`sidecar.py`** — sidecar JSON read/write; inheritance (dataset → subject → file level).
**`graph.py`** — DAG of file relationships (e.g., fieldmap→bold intendedFor).

### `parse/`

Each file handles one modality. All inherit `_base.py` → `ModalityParser`.
`parse(path, sidecar) → ModalityData` — reads file + sidecar, returns typed object.

| File | Modality | Library | Returns |
|---|---|---|---|
| `eeg.py` | EEG | MNE | `EEGData` (epochs, events, channels) |
| `meg.py` | MEG | MNE | `MEGData` |
| `ieeg.py` | iEEG | MNE | `iEEGData` + electrode coordinates |
| `fnirs.py` | fNIRS | MNE-NIRS | `fNIRSData` |
| `mri.py` | sMRI | nibabel | `MRIData` |
| `fmri.py` | fMRI | nibabel | `fMRIData` (BOLD + TR) |
| `dwi.py` | DWI | nibabel | `DWIData` (bvals, bvecs) |
| `pet.py` | PET | nibabel | `PETData` |
| `behavior.py` | events | Polars | `BehavioralData` |

`_mne_utils.py` — shared MNE helpers: montage loading, channel type classification.
`_registry.py` — `parse(path) → ModalityData` dispatcher by extension.

### `validation/`

**`bids_validator.py`** — wraps `bids-validator` CLI or `pybids`. Returns `ValidationReport(errors, warnings)`.
**`diff.py`** — validates that a local copy matches the remote manifest.
**`cache.py`** — caches validation results by dataset + checksum; avoids re-running.

---

## Part 9: Supporting Layers

### `convert/` — EDF/NIfTI → ML formats

**`pipeline.py`** — `ConversionPipeline(manifest, format, window_duration, overlap, splits)`. Chains windows → format converters.
**`windows.py`** — sliding window generation from signal data; respects `step_s`, `overlap_frac`.
**`splits.py`** — train/val/test split by subject (not by file — prevents leakage).
**`provenance.py`** — records conversion parameters, source checksums, output checksums.

**`formats/`**:
- `parquet.py` — Polars `write_parquet()`; column: `channel_data` as list
- `hdf5.py` — h5py; one dataset per channel
- `zarr.py` — zarr v2; chunked by window
- `webdataset.py` — `.tar` shards; `{key}.npy` + `{key}.json` per sample
- `tfrecord.py` — TFRecord for TensorFlow
- `huggingface.py` — `datasets.Dataset.from_dict()` → push to Hub

### `train/` — Framework bridges

All inherit `_base.py → TrainingBridge`. `prepare(manifest, split) → framework_dataset`.

| File | Framework | Returns |
|---|---|---|
| `huggingface.py` | HF Trainer | `datasets.Dataset` |
| `torch.py` | PyTorch | `torch.utils.data.Dataset` |
| `lightning.py` | Lightning | `LightningDataModule` |
| `braindecode.py` | Braindecode | `BaseConcatDataset` |
| `sklearn.py` | sklearn | `(X, y)` numpy arrays |
| `tensorflow.py` | TF | `tf.data.Dataset` |
| `ray.py` | Ray Train | `ray.data.Dataset` |
| `dask.py` | Dask | `dask.dataframe.DataFrame` |

### `visualize/` — Modality viewers

**`_dispatch.py`** — `visualize(data)` dispatches by type.
**`timeseries.py`** — EEG butterfly plot; channel offset display; event markers.
**`volume.py`** — 3-plane NIfTI viewer (axial/sagittal/coronal).
**`fmri.py`** — BOLD time series + activation overlay.
**`dwi.py`** — FA map + tractography glyphs.
**`dicom.py`** — DICOM stack browser (series → slice navigation).
**`overlay.py`** — segmentation mask overlay on anatomical.
**`surface.py`** — cortical surface mesh + scalar values.
**`_audit.py`** — `VisualAuditReport`: grid of random samples for QC.
**`_html.py`** — HTML report template for notebook and static export.
**`_asset.py`** — asset bundling (CSS, JS) for standalone HTML.
**`_colors.py`** — colormap registry; anatomical label color maps.

### `eda/` — Exploratory data analysis

**`quality.py`** — signal quality metrics: flatline detection, amplitude range, impedance proxy.
**`events.py`** — event table analysis: label distribution, inter-trial intervals, missing events.
**`summary.py`** — dataset-level summary: subjects, sessions, modalities, total recording time.
**`report.py`** — `EDAReport` assembly; calls quality + events + summary.
**`plots.py`** — EDA-specific plots: label histograms, quality heatmaps.

### `inspect/` — Dataset fitness

**`dataset.py`** — `DatasetInspector.profile()` → `DatasetProfile` (n_subjects, modalities, tasks, completeness).
**`label_landscape.py`** — `LabelLandscape`: which labels exist, which are usable (min N per class).
**`selector.py`** — `DatasetSelector.rank(goal: ResearchGoal)` → sorted `list[DatasetFitness]`.
**`signal_budget.py`** — estimates total signal (hours × channels × Hz) per modality.

### `qc/` — Quality control

**`filter.py`** — `QCFilter(masks)` — apply QC masks to filter subjects/files from manifest.
`QCMask` — binary mask per subject with reason field.

### `cohort/` — Multi-dataset cohorts

**`builder.py`** — `CohortBuilder.add_dataset(...).build()` → `CohortManifest` (merged manifest across datasets).
**`card.py`** — `DataCard`: summary card for a cohort (demographics, modalities, label distribution).
**`federated.py`** — `FederatedCohort`: cross-institution subjects where data stays remote; only metadata aggregated.

### `harmonize/` — Cross-dataset harmonization

**`reporter.py`** — `HarmonizationReporter.report(cohort)` → `HarmonizationReport`. Checks: channel naming consistency, sampling rate distribution, coordinate system alignment, label vocabulary overlap.

### `export/` — Framework-specific neuroimaging export

**`monai.py`** — `MONAIExporter.export(manifest, dest)` → MONAI Dataset-compatible directory with JSON split files.
**`torchio.py`** — `TorchIOExporter.export(manifest, dest)` → `tio.Subject` objects with correct spatial metadata.

### `derivatives/` — BIDS derivatives

**`indexer.py`** — `DerivativeIndexer.index(bids_root)` → maps derivatives (`derivatives/`) back to source subjects; supports `qortex` namespace derivatives.

### `stream/` — Direct file streaming (non-pipeline)

**`edf.py`** — `EDFStreamer(path)` → yields epochs; uses MNE `read_raw_edf`.
**`nifti.py`** — `NiftiStreamer(path)` → yields volume slices or TR volumes.
**`_cache.py`** — per-streamer LRU cache to avoid repeated file opens.

### `runtime/` — ML dataset wrappers

**`loader.py`** — `BIDSImageDataset(torch.Dataset)`, `BIDSSignalDataset`, `BIDSEpochDataset`.
**`epochs.py`** — `MONAIDictBuilder` (MONAI dict-transforms compatible), `TorchEEGBridge`.

### `indexing/local.py` — `LocalIndexReport`

Scans local dir for BIDS-like structure; builds `LocalIndexReport` without network access. Used for offline workflows.

### `lake/` — Data lake

**`layout.py`** — defines directory layout for the Qortex data lake (versioned by dataset + tag).
**`mount.py`** — mount/unmount lake volumes.
**`registry.py`** — tracks which datasets are in the lake.

### `plan/` — Download planning

**`planner.py`** — `DownloadPlanner.plan(manifest, include, exclude, resume)` → `DownloadPlan` with file list, estimated bytes, skippable files.
**`selector.py`** — file selection DSL (modality, subject, session, task filters).
**`lock.py`** — file lock for concurrent download prevention.

### `check/readiness.py` — Can-train gate

`ReadinessChecker.check(manifest) → ReadinessReport`. Gates: min subjects, min label distribution, no leakage risk, BIDS validity, signal budget estimate.

---

## Part 10: `_internal/` — Private Utilities

### `glob.py` — Gitignore-style BIDS glob

`glob_filter(all_paths, patterns) → dict[pattern, set[str]]`

Rules:
- Bare pattern (no `/`) → MATCHBASE (match basename at any depth)
- Leading `/` → anchored to root, disable basename match
- `*` and `**` do NOT match dotfiles (gitignore semantics)
- Tries `wcmatch` (full GLOBSTAR support); falls back to stdlib `fnmatch`

`apply_include_exclude(all_files, include, exclude) → (kept, included_set, excluded_set)` — used by fetch engine for selective download.

`find_close_matches(pattern, all_paths)` — `difflib.get_close_matches()` for error suggestions.

### `hashing.py` — Integrity utilities

`md5_file(path) → str`, `sha256_file(path) → str` — synchronous, 64 KiB chunks.
`StreamingHasher(algorithm)` — `.update(bytes)` + `.hexdigest()` + `.copy()` — for in-flight verification during async download.
`feed_existing_file_async(path, hasher)` — feeds already-downloaded file into hasher for resume verification.
`parse_etag_md5(etag) → str|None` — extracts plain MD5 from S3 ETag; rejects multipart ETags (`-N` suffix).

### `progress.py` — Progress reporting

All user output goes through here (never `print()` or direct `tqdm` import in other modules).
`bytes_bar(total, desc)` — total bytes, `unit="B"`, `unit_scale=True`, `unit_divisor=1024`.
`file_bar(total, desc, initial)` — per-file bar with resume support (`initial=` bytes already downloaded).
`count_bar(total, desc)` — item-count bar.
`msg(text, emoji, fallback)` — `tqdm.write()` safe; detects UTF-8 stdout for emoji output.
`spinner(desc)` — context manager for indeterminate tasks.

---

## Part 11: Public API Surface (`__init__.py`)

### `qortex` (top level)
```python
from qortex import Dataset, Artifact, configure, get_config
from qortex import QortexError
from qortex import DatasetQuery, facets, live_search
from qortex import DatasetInspector, DatasetSelector
from qortex import HarmonizationReporter, MONAIExporter, TorchIOExporter
from qortex import CohortBuilder, FederatedCohort
from qortex import NiftiStreamer, EDFStreamer
from qortex import BIDSImageDataset, BIDSSignalDataset, BIDSEpochDataset
from qortex import visualize
```

`Dataset` is the primary object — wraps all download/convert/eda/check operations.

### `qortex.neuroai`
```python
from qortex.neuroai import Pipeline, check
from qortex.neuroai import PipelineSpec, SourceSpec, ModelSpec, WindowSpec
from qortex.neuroai import SourceProfile, ModelProfile, InputContract, OutputContract
from qortex.neuroai import CompatibilityReport, CompatibilityStatus, EvidenceStatus
from qortex.neuroai import PreprocessPlan, TransformKind, AxisConvention
from qortex.neuroai import ArtifactWriter, CompatibilityEngine, PreprocessPlanner
from qortex.neuroai import SourceAdapter, ModelAdapter, OutputAdapter
from qortex.neuroai import ClassificationOutput, DetectionOutput, SegmentationOutput
```

`qortex.neuroai.check(source, model, task, provider)` — convenience one-liner; internally builds a `Pipeline`.

---

## Quick Reference: Invariants to Know Cold

```
SourceProfile.source_type         required (str, no default)
QortexTimeSeries.sampling_frequency_hz  NOT .sampling_rate_hz
QortexAbstraction.source_provenance     NOT .provenance
QortexAbstraction.axes                  list[str], NOT str
QortexVolume.voxel_sizes_mm             NOT .voxel_sizes
SourceProfile.spatial_shape             tuple, NOT list
OutputAdapter._n_written                increment _n_written, never n_written
AxisConvention.batch_channels_xyz       batch_channels_spatial does not exist
InputContract.evidence_status           NOT evidence={}
CompatibilityReport.summary()           returns "CompatibilityReport: COMPATIBLE..."
PreprocessSpec.allows(kind)             checks deny → allow list → mode
TriggerSpec.evaluate(dict)              prediction dict, not ModelOutput object
ArtifactWriter                          9 files including manifest with SHA-256
_SPATIAL_FRAMES                         {"LPS","RAS","LAS","SPATIAL_ZYX","SPATIAL_XYZ"}
PHI redaction                           DICOM adapter only; confirmed by extra["phi_redacted"]
plugin security gate                    trust_remote_code=True required before any import
```
