## Proposed library: `Qortex`

**Short description**

`Qortex` is a standalone Python data platform for OpenNeuro/BIDS datasets: it discovers datasets, builds pre-download manifests, validates BIDS structure, selectively downloads data, parses neuroimaging/electrophysiology files, performs EDA/QC, converts data into fast ML formats, and exposes training-ready interfaces for PyTorch, TensorFlow, Hugging Face Datasets, Ray, Dask, and cloud/object storage.

It should be independent from `openneuro-py` at the core level. `openneuro-py` is GPL-3.0-only and mainly exposes download/login functionality, so copying or deeply depending on it may constrain licensing and scope. Its latest PyPI release is `2026.4.1`, described as “A Python client for OpenNeuro,” with basic CLI/Python download support. ([PyPI][1]) ([PyPI][1])

---

# 1. Strategic position

`Qortex` should not be “another OpenNeuro downloader.”

It should be:

```text
OpenNeuro → Manifest → Validation → Download Plan → Local Data Lake → Parser → EDA/QC → ETL → ML/DL Training
```

The real contribution:

> A unified ML-ready OpenNeuro/BIDS data layer with fast metadata indexing, selective retrieval, validation, EDA, conversion, dashboarding, and training-framework adapters.

Why this is a valid gap:

- OpenNeuro provides a repository, GraphQL API, Git/DataLad access, snapshots, file trees, and DOIs, but not a complete Python ML/data-engineering workflow. OpenNeuro’s API exposes dataset/snapshot metadata and recursive file trees, while snapshots and files can be queried through GraphQL. ([docs.openneuro.org][2]) ([docs.openneuro.org][2])
- OpenNeuro datasets are git-annex repositories created with DataLad, and snapshots are represented as Git tags. ([docs.openneuro.org][3])
- The official OpenNeuro CLI can upload/download datasets and uses DataLad/git-annex mechanisms for larger annexed files, but it is Deno-based and not a complete Python ETL/ML platform. ([docs.openneuro.org][4]) ([docs.openneuro.org][4])
- PyBIDS works with local BIDS datasets and supports querying, modeling, and reports, but it is not an OpenNeuro-first discovery/download/ML export system. ([bids-standard.github.io][5])
- EEGDash is a close recent competitor for public neurophysiology and explicitly targets OpenNeuro/NEMAR EEG/MEG/iEEG/fNIRS/EMG ML access; therefore, the new contribution must be broader, multimodal, ETL/dashboard/reproducibility-first, and not limited to electrophysiology. ([arXiv][6])

---

# 2. Core design principle

Build an independent orchestration layer, but use mature domain libraries as optional backends.

| Layer                      | Own implementation | Optional backend                                                   |
| -------------------------- | ------------------ | ------------------------------------------------------------------ |
| OpenNeuro GraphQL client   | Yes                | none                                                               |
| Manifest/index engine      | Yes                | DuckDB, Polars, PyArrow                                            |
| Download planner           | Yes                | own HTTP backend, DataLad backend, optional `openneuro-py` backend |
| BIDS validation wrapper    | Yes wrapper        | BIDS Validator                                                     |
| BIDS query layer           | Yes wrapper/schema | PyBIDS                                                             |
| EEG/MEG/iEEG/fNIRS loading | unified API        | MNE-BIDS, MNE                                                      |
| MRI/fMRI/PET loading       | unified API        | NiBabel, Nilearn                                                   |
| EEG DL adapters            | unified API        | Braindecode, MOABB, EEGDash                                        |
| ETL/export                 | Yes                | Arrow, Parquet, Zarr, WebDataset, Hugging Face Datasets            |
| Dashboard                  | Yes                | Streamlit/Panel/FastAPI                                            |
| Training adapters          | Yes                | PyTorch, TensorFlow, Lightning, Ray, Dask                          |

MNE-BIDS is the correct backend for BIDS-compatible MEG, EEG, fNIRS, and iEEG workflows because it reads/writes BIDS-compatible datasets through MNE-Python and supports the BIDS-specified formats for those modalities. ([mne.tools][7])

NiBabel is the correct low-level neuroimaging backend because it reads/writes NIfTI, GIFTI, CIFTI-2, FreeSurfer formats, and other neuroimaging formats, with NumPy access and lazy loading. ([nipy.org][8])

Nilearn should be used for higher-level brain volume/surface analysis, statistical tooling, decoding, connectivity, and scikit-learn-style neuroimaging ML. ([nilearn.github.io][9])

---

# 3. Proposed package identity

## Working name

```text
Qortex
```

Alternative names:

```text
neuroflow
bidsflow
openneuro-lake
neurobids-flow
```

Best name for clarity:

```text
Qortex
```

Best name for broader future scope:

```text
neurobids-flow
```

## One-line description

```text
A standalone Python platform for turning OpenNeuro BIDS datasets into validated, queryable, ML-ready data lakes.
```

## Public API target

```python
from openneuro_flow import Dataset

ds = Dataset("ds000246", snapshot="1.0.0")

ds.inspect()
ds.validate()
ds.plan(include=["sub-0001/anat"]).summary()
ds.download(include=["sub-0001/anat"])
ds.eda().to_html("report.html")
ds.convert(format="zarr", modality="eeg")
train_ds = ds.to_torch(modality="eeg", task="rest")
```

---

# 4. Full feature list

## 4.1 OpenNeuro access layer

### Features

- Native OpenNeuro GraphQL client
- Dataset lookup by ID
- Snapshot lookup by tag/version
- Latest snapshot resolution
- Draft access where allowed
- File-tree crawling
- Recursive file listing
- File size retrieval
- File ID/hash retrieval
- DOI and dataset description retrieval
- API token support
- Environment variable token support
- Anonymous public access where supported
- Retry/backoff/rate-limit handling
- Request cache
- Typed API responses

### Objects

```python
OpenNeuroClient
DatasetRef
SnapshotRef
FileRecord
DatasetDescription
OpenNeuroAuth
```

### Why own this layer

OpenNeuro’s GraphQL API already exposes enough metadata to build a manifest-first system, including dataset information, snapshot descriptions, file IDs, filenames, sizes, directory flags, and recursive file trees. ([docs.openneuro.org][2]) ([docs.openneuro.org][2])

---

## 4.2 Catalog and discovery

### Features

- Search datasets by:
  - dataset ID
  - name
  - modality
  - task
  - number of subjects
  - sessions
  - file count
  - total size
  - BIDS datatype
  - keywords
  - DOI
  - license
  - publication metadata where available

- Local searchable catalog cache
- Catalog refresh
- Dataset ranking
- Dataset similarity search
- Dataset recommendation for ML tasks
- Export catalog to:
  - CSV
  - JSON
  - Parquet
  - DuckDB

### Output example

```python
catalog.search(
    modality="eeg",
    task="rest",
    min_subjects=20,
    max_size_gb=100,
)
```

### Internal storage

Use:

```text
DuckDB + Parquet + Polars
```

Reason:

- DuckDB: fast local analytical queries
- Parquet: portable metadata storage
- Polars: fast dataframe operations
- PyArrow: interoperability with ML/data tooling

---

## 4.3 Manifest engine

### Features

- Pre-download dataset inspection
- File tree table
- Subject/session/task/modality extraction
- Dataset size estimate
- File count estimate
- Datatype distribution
- Extension distribution
- Large-file detection
- Annexed/non-annexed detection where available
- Missing expected files
- BIDS entity extraction from filenames
- Sidecar matching:
  - `.json`
  - `.tsv`
  - `.bvec`
  - `.bval`
  - events files
  - channels files
  - electrodes files
  - coordsystem files

- Snapshot lock metadata
- Manifest diff between snapshots
- Manifest export

### API

```python
manifest = ds.manifest()

manifest.subjects()
manifest.sessions()
manifest.tasks()
manifest.modalities()
manifest.files.to_polars()
manifest.estimate_size(include=["sub-*/eeg/**"])
manifest.diff(other_snapshot="1.0.1")
manifest.to_json("manifest.json")
```

### Output schema

```text
dataset_id
snapshot
doi
file_id
path
filename
extension
size_bytes
directory
annexed
subject
session
task
run
modality
datatype
suffix
entities
sidecar_group
```

---

## 4.4 Download planner

### Features

- Dry-run mode
- Size estimate before download
- Include/exclude expressions
- Subject/session/task/modality selection
- File-extension selection
- Metadata-only download
- Raw-only download
- Derivatives-only download
- Exclude derivatives by default option
- Include `.bidsignore` automatically
- Storage budget check
- Duplicate file detection
- Resume interrupted downloads
- Parallel download
- Global rate-limit/backoff
- Per-file retry policy
- Per-dataset retry policy
- Checksum/hash verification
- Download provenance
- Download lockfile
- Failed-file report
- Partial failure recovery
- Cache-aware skip
- Drop/delete local content while keeping manifest

### API

```python
plan = ds.plan(
    subjects=["sub-001", "sub-002"],
    modalities=["eeg"],
    exclude=["derivatives/**"],
    include_metadata=True,
)

plan.summary()
plan.estimate_size()
plan.save("download.lock")
plan.execute("./data")
```

### Download backends

```text
native-http
datalad
git-annex
openneuro-cli-subprocess
openneuro-py-optional
```

Default should be `native-http` for independence.

DataLad should be optional because it gives powerful partial retrieval and provenance. DataLad can clone OpenNeuro datasets as lightweight repositories and retrieve only selected paths with `datalad get`, which avoids downloading full datasets unnecessarily. ([handbook.datalad.org][10]) ([handbook.datalad.org][10])

---

## 4.5 Local cache and data lake

### Features

- Global cache directory
- Dataset-specific cache
- Snapshot-specific cache
- Content-addressed file cache
- File presence tracking
- Integrity check
- Cache pruning
- Cache migration
- Offline mode
- Local object-store mode
- Symlink/hardlink/copy materialization
- Project-level dataset mount
- Data lake registry

### Layout

```text
~/.cache/Qortex/
  catalog.duckdb
  datasets/
    ds000246/
      snapshots/
        1.0.0/
          manifest.parquet
          download.lock
          files/
          reports/
          exports/
```

### Lockfile

```yaml
dataset_id: ds000246
snapshot: 1.0.0
doi: 10.18112/openneuro.ds000246.v1.0.0
created_at: ...
selection:
  subjects: [sub-0001]
  modalities: [meg]
  include: [...]
  exclude: [...]
files:
  - path: sub-0001/meg/...
    size: ...
    hash: ...
    status: present
```

---

## 4.6 BIDS validation layer

### Features

- Run BIDS Validator
- Parse JSON validation output
- Include `.bidsignore`
- Validate full dataset
- Validate partial subset
- Validate metadata-only subset
- Classify:
  - errors
  - warnings
  - ignored files
  - missing files
  - sidecar problems
  - inheritance issues

- ML-readiness score
- Validation report:
  - JSON
  - HTML
  - Markdown
  - dashboard view

- Validation caching
- Validation diff between snapshots
- Validation diff before/after repair

The BIDS Validator is the official compliance tool as a web app, command-line utility, and JavaScript/TypeScript library. ([bids-validator.readthedocs.io][11])

---

## 4.7 Quality control and ML-readiness scoring

### Features

Dataset-level scores:

- BIDS compliance score
- loadability score
- metadata completeness
- event-label availability
- modality consistency
- sampling-rate consistency
- missing-file risk
- train/test split risk
- subject leakage risk
- class imbalance
- size feasibility
- preprocessing feasibility
- license/reuse clarity
- citation/provenance completeness

Example score:

```text
BIDS score: 92/100
Loadability score: 78/100
ML-readiness score: 71/100
Risk: missing event labels for 18% of runs
```

### Real contribution

This is one of the strongest research contributions: a public “OpenNeuro ML-readiness index.”

EEGDash makes a similar argument for neurophysiology: metadata-compliant datasets may still fail to load, and ML reuse requires loadability/compliance metadata, format repair, windowing, and evaluation. ([arXiv][6])

---

## 4.8 Parser layer

### Unified modality API

```python
data = ds.load(
    subject="sub-001",
    session="ses-01",
    task="rest",
    modality="eeg",
)
```

### Supported modalities

| Modality          | Backend                                |
| ----------------- | -------------------------------------- |
| EEG               | MNE-BIDS / MNE                         |
| MEG               | MNE-BIDS / MNE                         |
| iEEG              | MNE-BIDS / MNE                         |
| fNIRS             | MNE-BIDS / MNE                         |
| EMG if present    | MNE/Braindecode/EEGDash-style adapters |
| MRI/anat          | NiBabel                                |
| fMRI/func         | NiBabel + Nilearn                      |
| DWI               | NiBabel + DIPY optional                |
| PET               | NiBabel + Nilearn optional             |
| Behavioral/events | Polars/Pandas                          |
| Phenotype         | Polars/Pandas                          |
| Derivatives       | PyBIDS + modality-specific loaders     |

### Output abstraction

```python
SignalRecord
ImageRecord
EventsRecord
MetadataRecord
SubjectRecord
SampleRecord
```

### Lazy loading

- Do not load full files by default.
- Metadata first.
- Signal/image data lazy until explicitly requested.
- Chunked reads when possible.
- Memory mapping for large arrays.
- Zarr/Arrow output for repeated access.

---

## 4.9 EDA and reporting

### Dataset overview

- dataset name
- snapshot
- DOI
- number of files
- total size
- subjects
- sessions
- tasks
- modalities
- datatypes
- suffixes
- derivatives presence

### Metadata EDA

- missing values
- inconsistent JSON sidecars
- inconsistent task names
- event-file coverage
- channel-file coverage
- scan-file coverage
- participant metadata summary
- phenotype table summary

### Signal EDA

For EEG/MEG/iEEG/fNIRS:

- sampling frequencies
- channel counts
- channel types
- recording durations
- bad/missing channels
- event distribution
- label distribution
- trial/window counts
- signal amplitude summaries
- PSD summary
- missing event labels
- annotation coverage

### Image EDA

For MRI/fMRI/PET:

- image shapes
- voxel sizes
- affine consistency
- TR distribution
- number of volumes
- mask availability
- spatial orientation
- header summaries
- missing sidecars
- file-size outliers

### Reports

```python
report = ds.eda()
report.to_html("eda.html")
report.to_markdown("eda.md")
report.to_json("eda.json")
```

---

## 4.10 Fast ETL and conversion

### Input

```text
BIDS raw files
BIDS derivatives
OpenNeuro subset
local BIDS dataset
DataLad dataset
```

### Output formats

| Format                 | Use                                           |
| ---------------------- | --------------------------------------------- |
| Parquet                | metadata, events, phenotype, tabular features |
| Arrow                  | fast columnar interchange                     |
| Zarr                   | chunked signals/images, cloud-friendly        |
| HDF5                   | scientific compatibility                      |
| NumPy memmap           | local fast training                           |
| WebDataset             | large-scale streaming DL                      |
| Hugging Face Datasets  | sharing/training                              |
| TFRecord               | TensorFlow                                    |
| Torch tensor cache     | PyTorch                                       |
| NWB optional           | neuroscience interoperability                 |
| BIDS derivative output | standards-preserving processed outputs        |

### ETL features

- windowing for signals
- event-aligned windows
- fixed-length windows
- overlapping windows
- subject-level splits
- session-level splits
- task-level splits
- class-balanced splits
- leakage-safe split generation
- normalization
- resampling
- filtering hooks
- artifact/QC hooks
- feature extraction
- chunked writes
- parallel conversion
- resumable conversion
- conversion lockfile
- conversion report

### API

```python
ds.convert(
    modality="eeg",
    output_format="zarr",
    window_seconds=10,
    overlap=0.5,
    target="events",
    split="subject",
)
```

---

## 4.11 Training adapters

### PyTorch

```python
train_ds = ds.to_torch(
    modality="eeg",
    split="train",
    target="events",
    lazy=True,
)
```

Features:

- `torch.utils.data.Dataset`
- `IterableDataset`
- streaming mode
- map-style mode
- collators
- transforms
- batch samplers
- distributed sampler
- subject-balanced sampler
- class-balanced sampler

### PyTorch Lightning

```python
dm = ds.to_lightning(
    modality="eeg",
    batch_size=32,
    num_workers=8,
)
```

### TensorFlow

```python
tfds = ds.to_tfdata(modality="fmri")
```

### Hugging Face Datasets

```python
hf_ds = ds.to_huggingface(format="arrow")
```

### Braindecode

Braindecode is PyTorch-native and already provides deep-learning workflows for EEG/ECoG/MEG, with MNE integration, dataset support through MOABB/EEGDash, and Zarr-backed dataset sharing. ([braindecode.org][12]) ([braindecode.org][12])

`Qortex` should expose compatible outputs:

```python
windows = ds.to_braindecode_windows(...)
```

### MOABB

MOABB is specialized for reproducible EEG/BCI benchmarks, with many EEG datasets, standard evaluations, MNE/scikit-learn integration, and benchmark results. ([moabb.neurotechx.com][13])

`Qortex` should not replace MOABB. It should export compatible datasets/evaluations where possible.

---

## 4.12 Dashboard

### Dashboard pages

| Page               | Features                                                      |
| ------------------ | ------------------------------------------------------------- |
| Catalog            | search OpenNeuro datasets                                     |
| Dataset            | summary, DOI, snapshot, modalities                            |
| Manifest           | file tree, size, subjects, tasks                              |
| Download planner   | include/exclude, size estimate, selected files                |
| Download monitor   | progress, retries, failed files, speed                        |
| Cache              | stored datasets, snapshots, disk use                          |
| Validation         | BIDS errors/warnings, `.bidsignore`, ML-readiness             |
| EDA                | event labels, channels, image shapes, duration, class balance |
| ETL                | conversion jobs, output formats, throughput                   |
| Training readiness | splits, leakage checks, adapters                              |
| Export             | HF, S3, GCS, Azure, local                                     |
| Provenance         | lockfiles, citations, dataset DOI, snapshot                   |

### Dashboard backend

```text
FastAPI service + DuckDB metadata + local cache + Streamlit/Panel frontend
```

### CLI command

```bash
Qortex dashboard
```

---

## 4.13 Monitoring and observability

### Features

- progress bars
- structured logs
- JSON logs
- download speed
- retry count
- failed file count
- skipped file count
- cache-hit count
- validation error count
- ETL throughput
- memory usage
- disk usage
- pipeline stage timing
- job resume
- crash recovery
- report generation

### Output

```python
result = ds.download(...)

result.downloaded_files
result.skipped_files
result.failed_files
result.bytes_downloaded
result.elapsed_seconds
result.report()
```

---

## 4.14 Export and platform connectors

### Local exports

- CSV
- JSON
- Parquet
- Arrow
- Zarr
- HDF5
- WebDataset
- TFRecord
- PyTorch tensors
- Hugging Face Datasets

### Cloud/object storage

- S3
- GCS
- Azure Blob
- MinIO
- local filesystem
- SSH/SFTP optional

Use `fsspec` abstraction.

### ML/experiment platforms

- Hugging Face Hub
- MLflow
- Weights & Biases Artifacts
- DVC
- LakeFS optional
- Ray Data
- Dask
- Spark optional

---

## 4.15 Dataset conversion and interoperability

### Convert OpenNeuro/BIDS to

```text
BIDS subset
BIDS derivative
Parquet metadata lake
Zarr signal/image lake
Hugging Face Dataset
PyTorch Dataset
TensorFlow Dataset
WebDataset shards
```

### Convert between representations

```python
ds.export("hf", repo_id="user/ds000246-eeg")
ds.export("zarr", path="s3://bucket/ds000246.zarr")
ds.export("webdataset", path="./shards")
ds.export("bids-subset", path="./subset")
```

### Important rule

Never destroy BIDS provenance. All converted datasets should keep:

```text
dataset_id
snapshot
doi
source_file
source_hash
subject
session
task
run
modality
BIDS entities
conversion_config
library_version
```

---

## 4.16 Repair and normalization layer

### Safe repairs

- normalize path metadata
- resolve missing `.bidsignore`
- parse BIDS entities
- validate sidecar inheritance
- generate derived manifest
- fix local symlink/materialization issues
- warn on missing file contents
- detect broken downloads
- detect inconsistent sampling frequency
- detect event/label mismatch

### Unsafe repairs

Do not silently rewrite scientific metadata.

For unsafe repairs:

```text
detect → report → suggest → require explicit user confirmation
```

---

## 4.17 Benchmark suite

### Benchmarks

- metadata query speed
- manifest generation speed
- file-tree parsing speed
- download planning speed
- selective download savings
- download throughput
- retry behavior
- BIDS validation time
- EDA report generation time
- ETL throughput
- Zarr/Parquet write speed
- training dataloader throughput
- memory usage
- loadability success rate

### Benchmark datasets

Use small/medium/large OpenNeuro datasets across:

```text
anat
func
dwi
eeg
meg
ieeg
fnirs
pet
derivatives
```

---

# 5. Implementation plan, not timeline

## Step 1: Define the universal internal schema

Create the “universal language” first.

Core entities:

```text
Dataset
Snapshot
Manifest
FileRecord
BIDSFile
Subject
Session
Task
Run
Modality
SidecarGroup
EventTable
SignalRecord
ImageRecord
SampleRecord
SplitRecord
ConversionRecord
ValidationReport
EDAReport
DownloadPlan
DownloadResult
ProvenanceRecord
```

This prevents the project from becoming a collection of scripts.

---

## Step 2: Implement native OpenNeuro GraphQL client

Required functions:

```python
client.get_dataset(dataset_id)
client.get_snapshots(dataset_id)
client.get_snapshot(dataset_id, tag)
client.get_latest_snapshot(dataset_id)
client.get_files(dataset_id, tag, recursive=True)
client.get_description(dataset_id, tag)
client.get_doi(dataset_id, tag)
```

Do not copy `openneuro-py` code because of license constraints. Use OpenNeuro API behavior directly.

---

## Step 3: Build manifest engine

Input:

```text
dataset_id + snapshot
```

Output:

```text
manifest.parquet
manifest.json
manifest.duckdb table
```

Processing:

1. Query recursive file tree.
2. Normalize file paths.
3. Parse BIDS entities.
4. Detect subject/session/task/run/modality.
5. Group sidecars.
6. Estimate size.
7. Build modality/task/subject summaries.
8. Save manifest.

---

## Step 4: Build selection and planning system

Input:

```text
manifest + selection query
```

Selection query examples:

```python
subjects=["sub-001"]
sessions=["ses-01"]
tasks=["rest"]
modalities=["eeg"]
include=["sub-*/eeg/**"]
exclude=["derivatives/**"]
```

Output:

```text
DownloadPlan
selected files
estimated bytes
required metadata files
warnings
```

Rules:

- Always include essential metadata.
- Always include `.bidsignore` if present.
- Warn if selected subset may not validate as full BIDS.
- Support dry-run before download.

---

## Step 5: Build independent downloader

Required features:

- async HTTP client
- shared connection pool
- global concurrency limit
- HEAD-free mode where API hashes/sizes are enough
- retry on 408/429/500/502/503/504/522/524
- per-file retries
- global backoff when server is stressed
- resume partial files
- checksum verification
- size verification
- structured result
- failed file report
- lockfile update
- crash recovery

Avoid known `openneuro-py` flaws:

- do not create one HTTP client per file
- do not print success before async jobs finish
- do not rely only on ETag
- do not crash whole download on one retryable file failure
- do not omit `.bidsignore`

---

## Step 6: Build validation wrapper

Processing:

1. Run BIDS Validator.
2. Capture machine-readable output.
3. Parse errors/warnings.
4. Map issues back to manifest files.
5. Classify severity.
6. Generate validation report.
7. Compute BIDS compliance score.
8. Compute ML-readiness score.

Output:

```python
ValidationReport
```

---

## Step 7: Build local BIDS index

Use PyBIDS where useful, but wrap it in your own API.

Processing:

1. Open local dataset.
2. Create BIDS layout.
3. Query files/entities.
4. Merge PyBIDS index with OpenNeuro manifest.
5. Detect local/remote mismatch.
6. Detect missing downloaded files.
7. Export local index to DuckDB/Parquet.

---

## Step 8: Build modality loaders

Implement loaders as plugins.

```text
openneuro_flow.loaders.eeg
openneuro_flow.loaders.meg
openneuro_flow.loaders.ieeg
openneuro_flow.loaders.fnirs
openneuro_flow.loaders.mri
openneuro_flow.loaders.fmri
openneuro_flow.loaders.dwi
openneuro_flow.loaders.pet
openneuro_flow.loaders.behavior
```

Each loader must implement:

```python
inspect()
load()
lazy_load()
to_numpy()
to_dataframe()
to_sample_records()
```

---

## Step 9: Build EDA/QC engine

Inputs:

```text
manifest + local index + loaded metadata + optional sampled data
```

Processing:

- summarize dataset
- summarize subject/session/task coverage
- summarize modality coverage
- summarize events
- summarize channel/image metadata
- detect outliers
- detect missing metadata
- detect leakage risks
- produce figures/tables
- export report

Output:

```python
EDAReport
```

---

## Step 10: Build ETL engine

Inputs:

```text
local BIDS files + manifest + parser + conversion config
```

Processing:

- load metadata
- load signals/images lazily
- create samples/windows
- assign labels
- split safely
- chunk arrays
- write output
- write provenance
- validate output

Output:

```text
Parquet/Arrow/Zarr/WebDataset/HF/Torch/TFRecord
```

---

## Step 11: Build ML adapters

Implement adapters after ETL schema is stable.

Adapters:

```python
to_torch()
to_iterable_torch()
to_lightning()
to_tfdata()
to_huggingface()
to_ray()
to_dask()
to_sklearn()
to_braindecode()
```

Each adapter must preserve:

```text
source dataset
snapshot
subject/session/task
modality
label source
preprocessing config
split policy
```

---

## Step 12: Build dashboard over existing APIs

Do not build dashboard first.

Dashboard should call stable core APIs:

```text
Catalog API
Manifest API
Download API
Validation API
EDA API
ETL API
Cache API
```

Backend:

```text
FastAPI
DuckDB
local cache registry
```

Frontend:

```text
Streamlit or Panel first
custom frontend later if needed
```

---

## Step 13: Build CLI

CLI commands:

```bash
Qortex search
Qortex inspect ds000246
Qortex plan ds000246 --subject sub-0001 --modality eeg
Qortex download ds000246
Qortex validate ./data/ds000246
Qortex eda ./data/ds000246
Qortex convert ./data/ds000246 --format zarr
Qortex export ./data/ds000246 --to hf
Qortex dashboard
Qortex cache list
Qortex cache prune
```

---

## Step 14: Build testing strategy

Test types:

- unit tests for schema/entity parsing
- unit tests for GraphQL query parsing
- mocked OpenNeuro API tests
- small real dataset integration tests
- partial download tests
- interrupted download tests
- validation tests
- parser tests per modality
- ETL roundtrip tests
- dashboard smoke tests
- ML adapter tests
- performance regression tests

Use small public datasets first. Do not require huge datasets in CI.

---

# 6. Recommended package modules

```text
openneuro_flow/
  __init__.py
  client/
    graphql.py
    auth.py
    retries.py
  catalog/
    search.py
    index.py
    registry.py
  manifest/
    builder.py
    schema.py
    bids_entities.py
    diff.py
  planning/
    selector.py
    planner.py
    lockfile.py
  download/
    native.py
    datalad.py
    cache.py
    result.py
  validation/
    bids_validator.py
    report.py
    scoring.py
  indexing/
    pybids_index.py
    local_index.py
  loaders/
    base.py
    eeg.py
    meg.py
    ieeg.py
    fnirs.py
    mri.py
    fmri.py
    dwi.py
    pet.py
    behavior.py
  eda/
    summary.py
    quality.py
    plots.py
    report.py
  etl/
    windows.py
    conversion.py
    zarr.py
    parquet.py
    webdataset.py
    hf.py
  ml/
    torch.py
    lightning.py
    tensorflow.py
    sklearn.py
    ray.py
    dask.py
    braindecode.py
  dashboard/
    app.py
    api.py
  cli/
    app.py
  provenance/
    lock.py
    citation.py
    audit.py
  tests/
```

---

# 7. Minimal first implementation scope

For the first real version, implement only this:

```text
Native GraphQL client
Manifest builder
Selection planner
Native downloader
Cache registry
BIDS validation wrapper
Basic EDA report
Parquet/Zarr export
PyTorch adapter
Dashboard
```

Supported modalities first:

```text
EEG
MEG
iEEG
fNIRS
anat MRI
fMRI
events/behavioral metadata
```

Do not start with every modality equally. Build a plugin system, then add modality-specific support.

---

# 8. Full feature checklist

## Access

- [ ] OpenNeuro GraphQL client
- [ ] API token auth
- [ ] environment variable auth
- [ ] public dataset access
- [ ] private dataset access where authorized
- [ ] snapshot resolver
- [ ] recursive file tree
- [ ] DOI extraction
- [ ] dataset description extraction

## Catalog

- [ ] local dataset catalog
- [ ] search by modality
- [ ] search by task
- [ ] search by subject count
- [ ] search by size
- [ ] search by keywords
- [ ] export catalog

## Manifest

- [ ] file manifest
- [ ] subject/session/task parser
- [ ] modality parser
- [ ] size estimator
- [ ] sidecar grouping
- [ ] snapshot diff
- [ ] manifest export

## Download

- [ ] dry-run
- [ ] selective download
- [ ] resumable download
- [ ] parallel download
- [ ] retry/backoff
- [ ] checksum verification
- [ ] failed file report
- [ ] structured result
- [ ] cache registry
- [ ] lockfile

## Validation

- [ ] BIDS Validator wrapper
- [ ] `.bidsignore` handling
- [ ] JSON report parsing
- [ ] HTML report
- [ ] ML-readiness score
- [ ] validation diff

## EDA/QC

- [ ] dataset summary
- [ ] modality summary
- [ ] subject/session/task coverage
- [ ] event distribution
- [ ] channel summary
- [ ] image metadata summary
- [ ] missing metadata report
- [ ] leakage checks
- [ ] class imbalance report

## Parsing

- [ ] PyBIDS integration
- [ ] MNE-BIDS integration
- [ ] NiBabel integration
- [ ] Nilearn integration
- [ ] event/phenotype parser
- [ ] lazy loading

## ETL

- [ ] Parquet export
- [ ] Arrow export
- [ ] Zarr export
- [ ] WebDataset export
- [ ] Hugging Face export
- [ ] TFRecord export
- [ ] Torch tensor cache
- [ ] conversion report

## ML/DL

- [ ] PyTorch Dataset
- [ ] PyTorch IterableDataset
- [ ] Lightning DataModule
- [ ] TensorFlow Dataset
- [ ] sklearn dataframe/matrix export
- [ ] Braindecode adapter
- [ ] Ray Data adapter
- [ ] Dask adapter

## Dashboard

- [ ] catalog browser
- [ ] manifest viewer
- [ ] download planner
- [ ] progress monitor
- [ ] validation viewer
- [ ] EDA explorer
- [ ] ETL monitor
- [ ] cache viewer
- [ ] export panel

## Provenance

- [ ] DOI tracking
- [ ] snapshot tracking
- [ ] file hash tracking
- [ ] conversion config
- [ ] split config
- [ ] citation helper
- [ ] reproducibility lockfile

---

# 9. Strongest contribution statement

The contribution should be framed as:

> `Qortex` turns OpenNeuro from a BIDS data archive into a complete ML-ready neurodata platform: manifest-first discovery, selective validated access, quality scoring, EDA, fast ETL, reproducible conversion, and direct integration with modern ML/DL training stacks.

That is a real contribution because the current ecosystem is fragmented:

```text
OpenNeuro = repository
openneuro-py = downloader
DataLad = versioned retrieval
PyBIDS = local BIDS query
MNE-BIDS = electrophysiology BIDS I/O
NiBabel/Nilearn = neuroimaging I/O/analysis
Braindecode/MOABB/EEGDash = mostly EEG/MEG/BCI/deep-learning ecosystem
```

`Qortex` becomes the missing integrated layer:

```text
repository → validated local lake → fast ML-ready data
```

[1]: https://pypi.org/project/openneuro-py/ "openneuro-py · PyPI"
[2]: https://docs.openneuro.org/api.html "API Examples - OpenNeuro documentation"
[3]: https://docs.openneuro.org/git.html "Git access to OpenNeuro datasets - OpenNeuro documentation"
[4]: https://docs.openneuro.org/packages/openneuro-cli.html "OpenNeuro command line interface - OpenNeuro documentation"
[5]: https://bids-standard.github.io/pybids/ "Welcome to pybids’s documentation! — PyBIDS 0.22.0 documentation"
[6]: https://arxiv.org/abs/2606.16041?utm_source=chatgpt.com "EEGDash: An open-source platform for machine learning on public neurophysiological data"
[7]: https://mne.tools/mne-bids/stable/index.html "MNE-BIDS — MNE-BIDS 0.19.0 documentation"
[8]: https://nipy.org/nibabel/ "NiBabel — NiBabel 0.1.0.dev1+gabe5e8e0c documentation"
[9]: https://nilearn.github.io/stable/index.html "Nilearn"
[10]: https://handbook.datalad.org/en/latest/usecases/openneuro.html "OpenNeuro Quickstart Guide: Accessing OpenNeuro datasets via DataLad — The DataLad Handbook"
[11]: https://bids-validator.readthedocs.io/en/latest/ "The BIDS Validator — BIDS Validator  documentation"
[12]: https://braindecode.org/stable/index.html "Braindecode — Decode raw EEG, ECoG and MEG with deep learning — Braindecode 1.5.1 documentation"
[13]: https://moabb.neurotechx.com/docs/index.html "MOABB - Mother of all BCI Benchmarks — MOABB Documentation"
