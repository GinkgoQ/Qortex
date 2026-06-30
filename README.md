# Qortex

Qortex is a Python library for working with OpenNeuro BIDS datasets. Its core value is a decision layer on top of the remote file manifest: it reads what the dataset contains — subjects, companions, labels, signal hours — before you transfer a byte, then guides you toward the smallest useful download.

It has two main surfaces: the **Dataset workflow** for OpenNeuro → ML artifact pipelines, and the **NeuroAI runtime** for source → model → output inference pipelines on local or streaming neuro data.

## Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[eeg,mri,dwi,validation,torch,sklearn]"
```

| Group | Packages added |
|---|---|
| `eeg` | mne, mne-bids |
| `mri` | nibabel, nilearn |
| `dwi` | dipy |
| `validation` | (requires `bids-validator` on PATH — external CLI) |
| `torch` | torch |
| `sklearn` | scikit-learn |
| `visual-all` | plotly, kaleido, matplotlib, PIL |
| `neuroai` | onnxruntime, pynwb, pyxdf, pylsl |

## Quick Start

```python
from pathlib import Path
from qortex import Dataset, Artifact

ds = Dataset("ds000001")

# Before downloading anything, inspect the manifest and check usability.
print(ds.doctor().to_text())
print(ds.label_landscape().summary())

# Plan the exact files needed for a first training batch.
plan = ds.minimum(goal="first-batch")
print(plan.to_text())

# Download only metadata and event tables (not raw imaging data).
ds.download_metadata(output_dir=Path("data/ds000001-meta"))

# Verify content before converting.
print(ds.content_status("data/ds000001-meta").to_text())
print(ds.can_train(local_path="data/ds000001-meta").to_text())

# Convert events/behavioral tables to a Parquet artifact.
result = ds.convert(
    output_dir=Path("artifacts/ds000001"),
    output_format="parquet",
    split_strategy="subject",
)

# Reopen for training.
artifact = Artifact.open("artifacts/ds000001")
X_train, y_train = artifact.sklearn(split="train")
train_ds = artifact.torch(split="train")
```

CLI equivalents:

```bash
qortex doctor ds000001
qortex minimum ds000001 --goal first-batch
qortex metadata ds000001 --download --output-dir data/ds000001-meta
qortex can-train ds000001 --local-path data/ds000001-meta
qortex convert ds000001 --output-dir artifacts/ds000001 --format parquet
```

## How It Works

**Manifest-first.** `Dataset("ds000001")` makes no network call. The manifest is fetched lazily on first use and cached in memory. Structural decisions — subject count, companion coverage, size estimates, label candidate status — come from the manifest and small sidecar files. No imaging data required.

**Companion-aware planning.** BIDS files have companions: a BOLD run needs its JSON sidecar and events.tsv; a DWI needs bval/bvec. The planner tracks these relationships and includes companions automatically in every plan.

**Decision reports.** `doctor()`, `can_train()`, `minimum()`, `first_batch()`, and `leakage_check()` return structured reports with three states: `possible`, `uncertain`, `not_possible`. They explain what evidence is missing, not just what the answer is.

**Evidence states.** Each field in a compatibility or readiness report carries an `EvidenceStatus`: `confirmed` (read from file), `inferred` (derived from manifest), `missing` (expected but absent), `unknown` (not checkable without download).

## Dataset Workflow

### Remote inspection without downloading

The OpenNeuro API and CDN expose enough to answer most pre-download questions:

```python
ds = Dataset("ds000117")

# Demographics from API — no TSV downloaded.
df = ds.participants()
print(df["age"].mean())

# Events table from CDN — no bulk download.
events = ds.events(subject="01", task="facerecognition")

# JSON sidecars, BIDS-merged across all inheritance paths.
meta = ds.sidecar("sub-01/meg/sub-01_task-facerecognition_meg.fif")
print(meta["SamplingFrequency"])

# NIfTI shape and TR from 352 bytes — not the full 38 MB file.
info = ds.nifti_info("sub-01/func/sub-01_task-facerecognition_bold.nii.gz")
print(info)  # 4D fMRI 64×64×33×208  vox=3.00×3.00×4.05mm  TR=2.000s

# Class balance, ISI jitter, cross-subject consistency — all events files fetched concurrently.
landscape = ds.label_landscape()
print(landscape.summary())

# Total achievable windows from remote sidecars + NIfTI headers.
budget = ds.signal_budget()
print(budget.estimate_windows(window_duration_s=2.0, overlap=0.5))
```

### Dataset fitness ranking

```python
from qortex.inspect import ResearchGoal, DatasetSelector

goal = ResearchGoal(
    modality="eeg",
    task_keywords=["motor", "imagery"],
    min_subjects=20,
    min_n_classes=2,
    min_trials_per_class=50,
    license_must_be_open=True,
    max_size_gb=5.0,
)

for fit in DatasetSelector().find(goal, limit=10):
    print(fit.summary_line())
```

`DatasetSelector` escalates through three tiers: local catalog → OpenNeuro API → remote events fetch. It only goes deeper when cheaper tiers return enough candidates.

### Decision reports

| Report | Question it answers |
|---|---|
| `doctor()` | Is this dataset usable? What evidence supports or blocks it? |
| `minimum(goal=...)` | What exact files are the smallest valid download for this goal? |
| `can_train(target=...)` | Are labels confirmed? What split policy is safe? What leakage risks exist? |
| `first_batch(artifact_path=...)` | Can Qortex read training rows now, or what download is needed first? |
| `content_status(local_path=...)` | Are local files complete? Any zero-byte, pointer-like, or manifest-mismatched files? |
| `leakage_check(artifact_path)` | Does this artifact leak subjects or source files across splits? |

```python
print(ds.doctor().to_text())
print(ds.minimum(goal="label-check").to_text())
print(ds.can_train(local_path="data/ds000001-meta", target="trial_type").to_text())
print(leakage_check("artifacts/ds000001").to_text())
```

### Selective download

```python
# Dry-run plan: files, companions, size, and reason per file.
plan = ds.plan(
    subjects=["01", "02"],
    modalities=["fmri"],
    event_complete=True,
    max_size_gb=2.0,
)
print(plan.summary())
print(plan.explain(limit=20))

ds.download_paths(plan.files, output_dir="data/ds000001")
```

Plan filters: `subjects`, `sessions`, `tasks`, `modalities`, `datatypes`, `include`/`exclude` globs, `metadata_only`, `with_companions`, `event_complete`, `label_ready`, `loadable_only`, `max_size_gb`, `conversion_target`.

### Conversion and artifacts

```python
result = ds.convert(
    output_dir=Path("artifacts/ds000001"),
    output_format="parquet",    # zarr, hdf5, webdataset, huggingface, tfrecord
    split_strategy="subject",   # random, stratified_subject
    shard_size=1000,
)

artifact = Artifact.open("artifacts/ds000001")
X_train, y_train = artifact.sklearn(split="train")
train_ds = artifact.torch(split="train")
```

Conversion writes a provenance record and `artifact_manifest.json` alongside the data shards. `Artifact.open()` reads those to reconstruct split counts, subject lists, and source tracking.

| Format | Status |
|---|---|
| Parquet | Scenario-tested; default path |
| Zarr / HDF5 / WebDataset / HuggingFace / TFRecord | Basic writers; limited roundtrip coverage |

### Visual QC

```python
ds = Dataset("ds000001", data_dir="data/ds000001")
report = ds.visualize(subjects=["01", "02"], suffixes=["T1w", "bold"])
report.to_html("reports/visual-audit.html")

# Modality-specific QC panels.
from qortex import visualize
visualize.visualize("sub-01/func/sub-01_task-rest_bold.nii.gz", mode="qc").to_html("bold-qc.html")
visualize.overlay_mask("T1w.nii.gz", "brain_mask.nii.gz").to_html("mask.html")

comparison = visualize.compare_masks("T1w.nii.gz", "pred.nii.gz", "truth.nii.gz", exact=True)
print(comparison.provenance["dice_exact"])
```

The visual audit HTML report shows manifest completeness, a subject × suffix coverage matrix, file cards (searchable and filterable), and prioritized action items.

### Catalog search

```python
import qortex

qortex.refresh_catalog(max_pages=5)

results = qortex.search(
    query="auditory",
    modality="eeg",
    min_subjects=20,
    limit=10,
)
for row in results:
    print(row["dataset_id"], row["score"], row["tasks"])
```

Catalog uses DuckDB when available, SQLite otherwise. Search filters: free text, `modality`, `task`, `author`, `license`, `min_subjects`, `max_size_gb`, `has_events`, `has_derivatives`.

## NeuroAI Runtime

The NeuroAI runtime runs source → model → output pipelines on local files, BIDS datasets, DICOM, NWB, XDF, LSL streams, BrainFlow boards, images, and video. It checks source-model compatibility before loading weights, builds a documented preprocessing plan, then runs and records a structured artifact.

### Pipeline YAML

```yaml
name: eeg_classifier
source:
  type: bids
  path: data/ds004130
  modality: eeg
  subject: "01"
window:
  duration_s: 4.0
  step_s: 2.0
preprocessing:
  mode: auto
  allow: [resample, channel_select, cast_dtype, to_tensor]
  deny: []
  normalize: false
  resample: true
  channel_select: true
model:
  provider: huggingface
  id: braindecode/EEGNet
  task: eeg_classification
outputs:
  - type: jsonl
    path: predictions.jsonl
  - type: lsl_marker
    stream_name: qortex_markers
runtime:
  device: cpu
  batch_size: 4
  num_workers: 2
  fp16: false
  cache_model: true
  latency_budget_ms: 50.0
artifact:
  failure_policy: strict   # strict | warn
```

The YAML loader accepts both singular and plural BIDS selectors (`subject` or
`subjects`, `session` or `sessions`). Window timing accepts `duration_s` /
`step_s` or their serialized forms `duration` / `step`, including strings such
as `"2s"` and `"500ms"`.

Preprocessing is contract-driven. Qortex does not add normalization, bandpass, or
image intensity scaling because a modality usually benefits from it. The
compatibility engine only plans transforms required by the model input contract,
and it respects `allow`, `deny`, and the boolean gates above. If a required
transform such as `cast_dtype`, `resample`, `reorient`, or `channel_select` is
denied, the compatibility report becomes `incompatible` with a structured
blocker instead of silently applying the transform.

Model contracts can also declare `required_transforms`,
`preferred_transforms`, and `forbidden_transforms`. Required transforms are
merged into the executable `PreprocessPlan` and must be allowed by the pipeline
policy. Unsupported axis convention mismatches are blockers unless Qortex can
emit a concrete `transpose_axes` transform. Spatial shape mismatches can use
`resample_spatial` only when the target shape is concrete and SciPy is
available; otherwise the pipeline fails early instead of pretending the
transform exists.

```bash
qortex neuroai check pipeline.yaml --markdown
qortex neuroai run pipeline.yaml --artifact-dir artifacts/run_001
qortex neuroai validate-artifact artifacts/run_001
qortex neuroai benchmark pipeline.yaml --windows 100
qortex neuroai suggest-models data.edf --task classification --json
```

### Python API

```python
from qortex.neuroai import Pipeline
from qortex.neuroai import validate_artifact

pipe = Pipeline.from_yaml("pipeline.yaml")

# Check compatibility without loading weights.
report = pipe.check()
print(report.summary())
# → CompatibilityReport: COMPATIBLE_WITH_TRANSFORMS

# Inspect what preprocessing will be applied and why.
for t in pipe.plan_preprocessing().transforms:
    print(f"  {t.kind.value}: {t.required_by}")

if report.is_runnable:
    run = pipe.run(artifact_dir="artifacts/run_001")
    print(run.latency_report.summary())
    print(run.outputs)  # prediction and marker record counts per adapter
    validation = validate_artifact("artifacts/run_001")
    print(validation.summary())

bench = pipe.benchmark(n_windows=50)
print(bench.summary())   # per-window p50/p95/p99, batch p50/p95/p99, throughput

pipe.replay("recording.xdf", speed=2.0)
```

### Sources

| Type | Class | Notes |
|---|---|---|
| Local file (EDF/BDF/FIF) | `LocalFileAdapter` | MNE-based |
| BIDS dataset | `BIDSSourceAdapter` | Profiles supported recording files through the local source adapter, records BIDS entities, and reports cross-record header consistency |
| DICOM folder | `DICOMFolderAdapter` | Groups by SeriesInstanceUID, affine from header |
| DICOMweb | `DICOMWebAdapter` | QIDO-RS metadata + WADO-RS pixels |
| NWB | `NWBAdapter` | `ElectricalSeries` via `pynwb` |
| XDF | `XDFAdapter` | Stream selection by type or name |
| LSL stream | `LSLSourceAdapter` | Real-time; ring buffer |
| BrainFlow board | `BrainFlowAdapter` | Any board via board_id + params |
| Image / video | `ImageVideoAdapter` | PIL for images, OpenCV for video |

### Models

| Provider | Class | Notes |
|---|---|---|
| `huggingface` | `HuggingFaceModelAdapter` | Native `transformers.pipeline` tasks only; non-native tensor tasks must use ONNX/Torch/Braindecode/MONAI/plugin |
| `onnx` | `ONNXModelAdapter` | ONNX Runtime; CPU and CUDA EP |
| `torch` / `torchscript` | `TorchModelAdapter` | `.pt` or `.ts`; FP16 on CUDA |
| `braindecode` | `BrainDecodeAdapter` | EEGNet, ShallowFBCSPNet, etc. |
| `monai` | `MONAIBundleAdapter` | MONAI bundles with config/spec-driven ROI size, overlap, activation, argmax/threshold, and label map |
| `ultralytics` | `UltralyticsAdapter` | YOLOv8 detect / segment / classify |
| `plugin` / `custom` | `CustomPluginAdapter` | Load any `.py` implementing `QortexPlugin` |

### Outputs

| Type | What it writes |
|---|---|
| `jsonl` | One JSON line per prediction window |
| `parquet` | Columnar predictions with metadata |
| `csv` | Analytics-friendly CSV with probabilities, metadata, trigger state, source, pipeline hash, and compact summaries for array payloads |
| `lsl_marker` | LSL marker stream (`pylsl`) |
| `nifti` | NIfTI mask with affine (`nibabel`) |
| `dicom_seg` | DICOM-SEG object (`highdicom`) |
| `dicom_sr` | SR MeasurementReport (`highdicom`) |
| `bids` | BIDS derivative directory |
| `coco` | COCO JSON |
| `yolo` | Per-image `.txt` with normalized boxes |
| `overlay` | Annotated images with boxes/masks/labels drawn on source frames |
| `websocket` | JSON payload over WebSocket |
| `http` | JSON POST with retry and auth |

The `overlay` adapter accepts a `source_image` key in the metadata dict (numpy `[H, W]` or `[H, W, C]`) and renders detection boxes, segmentation masks, or classification labels onto it using Pillow or OpenCV.

### Artifact directory

When `artifact_dir` is passed to `pipe.run()`, file-backed outputs are routed
under `artifact_dir/outputs/` and the manifest hashes both sidecars and
prediction files:

```
artifacts/run_001/
  provenance.json           pipeline spec, source, model, timestamps
  compatibility_report.json source-model check results
  preprocess_plan.json      transforms applied, why each was needed
  runtime_report.json       window counts, error counts, output counts
  latency_report.json       p50/p95/p99 per stage
  warnings.json             non-fatal issues during the run
  pipeline.yaml             spec copy
  artifact_contract.json    hash, schema, provenance summary
  artifact_manifest.json    SHA-256 of every file below
  outputs/
    predictions.jsonl
    predictions.csv
```

Artifact writing is strict by default when `artifact_dir` is requested. If
`ArtifactWriter` fails, `Pipeline.run()` raises instead of returning a run that
cannot be reproduced. Use `artifact.failure_policy: warn` only for exploratory
runs where a missing artifact is acceptable.

Validate the completed run before sharing it or using it as a downstream
evidence object:

```python
from qortex.neuroai import validate_artifact

report = validate_artifact("artifacts/run_001", strict=True)
print(report.summary())
print(report.to_markdown())
```

The validator checks required sidecars, `artifact_manifest.json` SHA-256 and
size entries, JSONL prediction records, trigger marker records, CSV schema,
Parquet metadata, NIfTI mask geometry, COCO/YOLO structure, DICOM output
headers when `pydicom` is installed, and runtime/output-count consistency from
`runtime_report.json`.

### Ring buffer

For real-time sources (LSL, BrainFlow), Qortex uses a lock-free ring buffer. A Rust extension (`src/qortex_rs/`) is included for performance-critical use; the Python fallback handles all cases where Rust is not built.

```python
from qortex.neuroai.sources._ring_buffer import get_ring_buffer, batch_window

buf = get_ring_buffer(n_channels=64, capacity=2048, window_size=512, step_size=256)
buf.push(chunk)               # [Ch, N] float32
window = buf.pop_window()     # [Ch, 512] or None

# Offline windowing:
windows = batch_window(data, window_size=512, step_size=256)
```

Build the Rust extension (optional — Python fallback is always available):

```bash
cd src/qortex_rs && maturin develop --release
```

## Implementation Status

| Subsystem | Status |
|---|---|
| Core contracts/config/exceptions | Production-oriented; validated config overrides, structured exceptions, Qortex warnings, and entity invariants |
| Manifest access + semantic graph | Production; scenario-tested on `ds000001` |
| Remote preview + metadata-first workflows | Production |
| Decision reports (doctor, minimum, can-train, etc.) | Production; scenario-tested |
| Companion-aware planning + selective download | Production |
| Catalog ingestion and search | Production; scenario-tested |
| Local index reconciliation | Production |
| EDA + readiness scoring | Useful; score weighting evolving |
| Parquet conversion | Useful; scenario-tested |
| Zarr / HDF5 / WebDataset / HuggingFace / TFRecord writers | Basic; roundtrip coverage limited |
| Torch adapter | Useful; basic batching, no advanced collation |
| Visual QC (manifest audit, fMRI, DWI, overlays, masks) | Useful; scenario-tested |
| BIDS Validator wrapper | Useful; does not fabricate results |
| EEG / MEG / MRI / DWI / PET loaders | Real optional-dependency loaders; need fixture coverage |
| NeuroAI sources (DICOM, NWB, XDF, LSL, BrainFlow, image) | Implemented; DICOM has PHI redaction + affine; integration tests pending |
| NeuroAI models (HF, ONNX, Torch, MONAI, Ultralytics, Braindecode) | Implemented; ONNX supports explicit output decoders; Braindecode requires confirmed dimensions |
| NeuroAI outputs (JSONL, Parquet, CSV, NIfTI, DICOM-SEG, BIDS, COCO, YOLO, overlay, HTTP) | Implemented; overlay renders bounding boxes and masks on source images |
| NeuroAI compatibility engine | Implemented; checks modality, channels, sampling rate, spatial shape, dtype, voxel spacing, coordinate frame, fMRI TR, denied-transform blockers, and detailed `explain()` exports |
| NeuroAI preprocessing planner/executor | Implemented; contract-driven transform planning, strict critical-transform failures, real named-channel selection, no hidden modality heuristics |
| NeuroAI runtime batching + metadata | Implemented; `batch_size` batches windows and output metadata carries source/window details |
| NeuroAI artifact integrity | Implemented; artifact runs route file outputs into `artifact_dir/outputs` and recursively hash outputs plus sidecars |
| NeuroAI trigger system | Implemented; fires structured EventMarkerOutput to all output adapters |
| NeuroAI ring buffer (Python + Rust) | Implemented; Rust is optional |
| Dashboard | Experimental entrypoint; not a product |

## Current Limitations

- Decision reports require local event files to confirm labels. Remote manifest alone produces `uncertain` states.
- `Dataset.validate()` requires `bids-validator` on PATH. Qortex raises a clear error if it's missing.
- Torch and sklearn adapters expect Parquet artifacts.
- Event-aligned windowing exists in `Dataset.convert(event_aligned=True)` but lacks broad real-signal test coverage.
- DICOM-SEG and DICOM-SR writers fall back to `.npy` / `.json` if `highdicom` API changes break compatibility.

## Project Structure

| Package | Role |
|---|---|
| `qortex.client` | OpenNeuro GraphQL client |
| `qortex.catalog` | Local DuckDB/SQLite catalog |
| `qortex.manifest` | Manifest models, BIDS parsing, semantic graph |
| `qortex.plan` | Selection planning |
| `qortex.fetch` | Download backends and cache |
| `qortex.check` | Readiness analysis |
| `qortex.eda` | Summaries, QC, HTML reports |
| `qortex.parse` | Modality loaders |
| `qortex.convert` | Windowing, splitting, artifact writers |
| `qortex.artifact` | Artifact access |
| `qortex.train` | ML framework adapters |
| `qortex.indexing` | Local BIDS index and reconciliation |
| `qortex.validation` | BIDS Validator wrapper |
| `qortex.neuroai` | Source → model → output inference pipelines |
| `qortex.cli` | CLI |

## Tests

The repo includes a real scenario suite (no mocks) under `test/`:

```bash
python test/run_all.py

# Override dataset:
QORTEX_REAL_TEST_DATASET=ds000001 QORTEX_REAL_TEST_SNAPSHOT=1.0.0 python test/run_all.py
```

The suite shares one metadata download across all stages. Stages 0–20 cover: import, manifest models, planning, preview, download, EDA, conversion, readiness, loaders, windowing, local index, catalog, CLI, Dataset facade, decision workflows, remote inspection, label landscape, signal budget, and dataset selection.

## CLI Reference

```bash
qortex search           search local catalog
qortex inspect          fetch and summarize a manifest
qortex metadata         list or download metadata files
qortex preview          preview first rows of a remote/local file
qortex plan             compute a download plan (dry run)
qortex download         download dataset or subset
qortex doctor           usability report
qortex minimum          smallest valid download for a goal
qortex can-train        label and split feasibility
qortex first-batch      read first rows or plan needed
qortex content-status   check local file completeness
qortex leakage-check    check split leakage in artifact
qortex validate         run BIDS Validator
qortex local-index      index local BIDS tree
qortex eda              EDA report + optional HTML
qortex convert          convert dataset to ML artifact
qortex visualize        render one local file (--mode qc for QC panels)
qortex visual-audit     manifest-aware visual audit
qortex fmri-qc          BOLD QC (tSNR, stability, events)
qortex dwi-qc           DWI QC (b0, shells, gradient sphere)
qortex compare-masks    prediction vs ground-truth mask comparison
qortex artifact-visualize  inspect artifact samples and splits
qortex catalog-refresh  ingest OpenNeuro metadata into catalog
qortex neuroai check    probe source + model compatibility
qortex neuroai run      run inference pipeline
qortex neuroai validate-artifact  verify NeuroAI run artifact integrity
qortex neuroai benchmark  latency benchmark
qortex neuroai replay   replay recorded session
qortex neuroai suggest-models  rank compatible models for a source
```
