# Qortex

Qortex by GinkgoQ is a production-oriented Python library for working with
OpenNeuro and BIDS datasets. It provides a real dataset workflow: discover a
dataset, inspect its manifest, preview metadata, plan selective downloads,
download only what is needed, analyze readiness, summarize labels, convert local
tables/events into ML-ready artifacts, and open those artifacts through ML
adapters. It also includes decision-first workflows for answering practical
questions such as "what is the smallest real download?", "can I train from this
dataset?", "can I inspect the first batch?", and "does this local/artifact state
look safe?".

Qortex is not a thin file downloader. Its core contribution is a semantic layer
over OpenNeuro manifests: files are interpreted as BIDS entities, logical
recordings, companion sets, event tables, sidecars, and ML provenance units.

## Install

From this checkout:

```bash
python -m pip install -e .
```

Optional extras are available for modality-specific loading and ML frameworks:

```bash
python -m pip install -e ".[eeg,mri,dwi,validation,torch,sklearn]"
```

The official BIDS Validator is an external CLI. `Dataset.validate()` and
`qortex validate` use it when `bids-validator` is installed on `PATH`; Qortex
does not fabricate validation results when the CLI is missing.

## Simple Usage

```python
from pathlib import Path

from qortex import Artifact, Dataset

ds = Dataset("ds000001")

# 1. Inspect a real OpenNeuro dataset without downloading full neuroimaging data.
info = ds.info()
print(info)

# 2. Preview real metadata remotely.
rows = ds.first_rows("participants.tsv", n=5)
description = ds.preview("dataset_description.json", max_bytes=4096)

# 3. Plan a selective download before transferring bytes.
plan = ds.plan(
    subjects=["01"],
    modalities=["fmri"],
    event_complete=True,
)
print(plan.summary())
print(plan.explain(limit=10))

# 4. Download only metadata, sidecars, and event tables.
metadata_dir = Path("data/ds000001-metadata")
ds.download_metadata(output_dir=metadata_dir)

# 5. Analyze readiness and labels from the local metadata.
doctor = ds.doctor(local_path=metadata_dir)
can_train = ds.can_train(local_path=metadata_dir)
print(doctor.to_text())
print(can_train.to_text())

eda = ds.eda(local_path=metadata_dir, output_html=Path("eda.html"))
print(eda.quality.ml_readiness_score)

# 6. Convert downloaded local tables/events into a Parquet artifact.
result = ds.convert(
    output_dir=Path("artifacts/ds000001-events"),
    output_format="parquet",
    split_strategy="subject",
)
print(result.n_samples, result.splits)

# 7. Reopen the artifact later.
artifact = Artifact.open("artifacts/ds000001-events")
print(artifact.summary())
```

CLI equivalents:

```bash
qortex metadata ds000001 --limit 20
qortex preview ds000001 participants.tsv --rows 5
qortex plan ds000001 --subjects 01 --modalities fmri
qortex doctor ds000001
qortex minimum ds000001 --goal first-batch
qortex can-train ds000001 --local-path data/ds000001-metadata
qortex first-batch --dataset ds000001
qortex metadata ds000001 --download --output-dir data/ds000001-metadata
qortex local-index data/ds000001-metadata
qortex validate data/ds000001-metadata --markdown-output validation.md
```

## Core Capabilities

| Area | What Qortex Provides |
| --- | --- |
| OpenNeuro access | Native GraphQL client for datasets, snapshots, metadata, and file manifests |
| Dataset facade | One high-level `Dataset` object for manifest, selection, download, readiness, EDA, validation, indexing, conversion, and adapters |
| Manifest model | Typed `Manifest`, `FileRecord`, `BIDSEntities`, and dataset summary models |
| BIDS parsing | Subject/session/task/run/entity extraction, datatype detection, suffix detection, compound extension handling |
| Semantic graph | `LogicalRecording` objects with primary files and required companions |
| Companion closure | Events, JSON sidecars, channels, electrodes, coordsystem, scans, participants, dataset description, bvec, and bval support |
| Selection planner | Explainable dry-run plans with filters, companion expansion, size estimates, and per-file reasons |
| Selective download | Full dataset, metadata-only, exact-path, subject/session/task/modality/datatype filters |
| Remote preview | Bounded remote/local previews for JSON, TSV, and CSV without full dataset download |
| Local indexing | Local BIDS tree indexing and reconciliation against the OpenNeuro manifest |
| Validation | Official BIDS Validator wrapper, typed report, cache, JSON/Markdown/HTML export, report diff |
| Readiness | Download, event, label, loadability, split, and conversion readiness reports |
| Decision workflows | `doctor`, `minimum`, `can-train`, `first-batch`, `content-status`, `leakage-check`, and reproducible recipes |
| EDA/QC | Dataset summaries, modality summaries, coverage, quality scores, event-label summaries, HTML reports |
| Loaders | Lazy plugin registry and modality-aware loaders for behavior, EEG, MEG, iEEG, fNIRS, MRI, fMRI, DWI, and PET |
| Conversion | Loader resolution, optional windowing, split assignment, writer dispatch, provenance, artifact manifests |
| Artifacts | Reopen converted artifacts, inspect sample/split/source metadata, hand off to adapters |
| ML adapters | Basic artifact bridges for Torch, Lightning, sklearn, TensorFlow, HuggingFace, Ray, Dask, and Braindecode; maturity varies by framework |
| Catalog | Normalized OpenNeuro catalog ingestion, deep file-summary digestion, structured facets, and weighted local search backed by DuckDB or SQLite fallback |
| Remote inspection | Demographics from API (no TSV download), NIfTI header extraction via 352-byte Range request, concurrent remote events fetch, label landscape with imbalance and ISI jitter, signal budget estimation from sidecars |
| Dataset fitness ranking | `ResearchGoal` + `DatasetSelector` multi-dimensional scoring: modality, task, subjects, license, class balance, signal hours, community engagement |
| CLI | Search, inspect, metadata, preview, plan, download, decision workflows, validate, local-index, EDA, convert, cache, login |

## Implementation Maturity

Qortex intentionally separates what is deep today from what is present as an
integration path. This avoids treating module existence as production maturity.

| Subsystem | Current maturity | Honest status |
| --- | --- | --- |
| OpenNeuro manifest access | Strong | Used by the real scenario suite against `ds000001`; handles current GraphQL schema |
| Metadata-first preview | Strong | Remote bounded JSON/TSV/CSV preview and first-row inspection are core, tested workflows |
| Semantic recording graph | Strong | Builds logical recordings and companion closures from real manifests |
| Companion-aware planning | Strong | Exact-path, metadata-only, and semantic companion plans are exercised on real OpenNeuro data |
| Metadata/event download | Strong | Real metadata/event downloads are used by the scenario suite |
| Decision-first workflow layer | Useful | `doctor`, `minimum`, `can-train`, `first-batch`, `content-status`, `leakage-check`, recipe read/write, and CLI commands are implemented and scenario-tested on real OpenNeuro metadata |
| Catalog ingestion/search | Useful | Paged OpenNeuro metadata ingestion, per-dataset deep file-summary digestion, normalized modality/task/author/keyword/file-summary tables, facets, and structured search are scenario-tested |
| Local index reconciliation | Useful | Detects missing remote files, extra local files, size mismatches; optional PyBIDS path exists |
| Readiness scoring | Useful but evolving | Produces structured reports; score accounting is shown, but deeper weighting/explainability is future work |
| Event/table EDA | Useful | Summarizes real local event labels, counts, classes, imbalance, and HTML reports |
| Behavior/events loader | Useful | Loads real BIDS TSV/CSV tables and emits `SampleRecord` rows |
| EEG/MEG/iEEG/fNIRS loaders | Real optional-dependency loaders | Use MNE/MNE-BIDS-style loading paths; require real modality fixtures and broader scenario coverage |
| MRI/fMRI/DWI/PET loaders | Real optional-dependency loaders | Use NiBabel/domain dependencies and expose shape/metadata/sample records; need deeper roundtrip coverage |
| Parquet writer | Useful | Real event/table artifact generation is scenario-tested; stores numeric arrays and structured row JSON |
| Zarr/HDF5/WebDataset/HuggingFace/TFRecord writers | Basic | Writers exist with optional dependencies, but need formal roundtrip tests and larger-data validation |
| Torch adapter | Basic/useful | Opens Parquet artifacts, supports split filtering and iterable mode; not yet advanced batching/collation |
| sklearn adapter | Basic | Works for numeric signal bytes; event/table-only artifacts need dataframe workflows instead of `(X, y)` |
| TensorFlow/HuggingFace/Ray/Dask/Braindecode adapters | Basic integration paths | Modules exist, but should not be treated as production-grade adapters yet |
| Validation wrapper | Useful | Runs official `bids-validator` when installed; exports typed reports; does not fabricate results |
| Dashboard | Experimental | Entrypoint exists; full dashboard product is future work |
| Real scenario suite | Useful integration coverage | Uses real OpenNeuro data and shared metadata download; not a substitute for unit/contract tests |

## Compatibility Matrix

| Feature | Remote manifest only | Metadata-only local tree | Full local BIDS tree | Converted artifact |
| --- | --- | --- | --- | --- |
| Dataset summary | Yes | Yes | Yes | Via artifact summary |
| Metadata preview | Yes | Yes | Yes | No |
| Label status | Candidate only | Confirmed for downloaded event tables | Confirmed when local events are present | Confirmed from artifact metadata/rows |
| Companion planning | Yes | Yes | Yes | No |
| EDA | Manifest-level | Metadata/event-level | Best available local EDA | Artifact-level adapters only |
| BIDS validation | No | Possible but will report missing raw files | Yes, with official CLI | No |
| Conversion | No | Event/table conversion | Loader-dependent full conversion | No |
| First ML batch | No | Possible for table artifacts | Possible when loaders/adapters support modality | Possible for compatible artifacts |
| Leakage checks | Plan/local risk only | Plan/local risk only | Source/derivative risk checks | Subject/source/derivative split checks |

## Public Python API

### `qortex.Dataset`

| Method | Purpose |
| --- | --- |
| `Dataset(dataset_id, snapshot=None, token=None, data_dir=None)` | Bind work to one OpenNeuro dataset and optional snapshot |
| `manifest(force_refresh=False)` | Fetch and cache a structured OpenNeuro manifest |
| `info()` | Return compact dataset statistics and metadata |
| `files(...)` | Filter manifest files by subject, session, task, modality, datatype, extension, or metadata-only flag |
| `metadata_files()` | Return essential metadata, sidecars, and lightweight table files |
| `plan(...)` | Build an explainable `DownloadPlan` without downloading |
| `select(...)` | Alias for `plan(...)` |
| `download(...)` | Download a full or filtered dataset subset |
| `download_metadata(...)` | Download only metadata, sidecars, events, and lightweight tables |
| `download_paths(paths, ...)` | Download exact manifest paths with optional companion expansion |
| `preview(path, ...)` | Preview a remote or local JSON/TSV/CSV file using a bounded byte range |
| `first_rows(path, n=5, ...)` | Return first rows from a remote/local TSV or CSV file |
| `preview_metadata(...)` | Preview several metadata/sidecar/table files |
| `check(...)` | Return a decision-oriented `ReadinessReport` |
| `doctor(local_path=None)` | Return a high-level usability report with status, findings, and next actions |
| `minimum(goal="first-batch", ...)` | Plan the smallest real download for `label-check`, `first-batch`, `validation`, or `metadata` |
| `can_train(...)` | Report possible/uncertain/not possible training status, label status, split policy, and leakage risks |
| `first_batch(...)` | Read first artifact rows or return the smallest plan needed to produce a first batch |
| `content_status(local_path=None)` | Check local files for completeness, pointer-like content, and manifest mismatches |
| `validate(...)` | Run official BIDS Validator and return a typed `ValidationReport` |
| `index_local(...)` | Index a local BIDS tree and reconcile it with the manifest |
| `eda(...)` | Run EDA/QC and optionally write an HTML report |
| `inspect()` | Fetch full metadata, ML readiness score, modality breakdown, and recommendations — no download |
| `participants(prefer_api=True)` | Return demographics as a Polars DataFrame from API or remote participants.tsv |
| `events(subject, session, task, run)` | Fetch a remote events TSV as a Polars DataFrame — no download |
| `sidecar(path)` | Fetch and BIDS-merge all JSON sidecars for a file path — no download |
| `nifti_info(path)` | Extract NIfTI shape, TR, and voxel sizes via a 352-byte HTTP Range request |
| `label_landscape(...)` | Concurrently fetch all events files and analyze class balance, ISI jitter, and cross-subject consistency |
| `signal_budget(...)` | Estimate total signal hours and achievable windows from remote sidecars and NIfTI headers |
| `convert(...)` | Convert downloaded local data into an ML artifact |
| `torch_dataset(...)` | Open a converted Parquet artifact as a Torch dataset |
| `lightning_datamodule(...)` | Open a converted artifact as a Lightning DataModule |
| `sklearn_arrays(...)` | Open a converted artifact as `(X, y)` arrays |

### Module-Level API

| Function / Class | Purpose |
| --- | --- |
| `qortex.configure(...)` | Override cache, endpoints, concurrency, retry, auth, and integrity settings |
| `qortex.get_config()` | Read active configuration |
| `qortex.search(...)` | Search the local catalog |
| `qortex.refresh_catalog(...)` | Ingest paged OpenNeuro metadata into the local catalog |
| `qortex.refresh_catalog_dataset(...)` | Ingest and return one dataset profile, optionally with recursive file-summary digestion |
| `qortex.Artifact.open(path)` | Open a converted Qortex artifact |
| `Artifact.summary()` | Return artifact ID, dataset, snapshot, format, samples, subjects, and split counts |
| `Artifact.torch(...)` | Open a Parquet artifact for Torch |
| `Artifact.sklearn(...)` | Open a Parquet artifact for sklearn |
| `qortex.content_status(path, manifest=None)` | Inspect local content independently of a `Dataset` object |
| `qortex.leakage_check(artifact_path)` | Check converted artifacts for split leakage by subject/source and derivative-source risk |
| `qortex.Recipe` | Typed workflow recipe model |
| `qortex.write_recipe(recipe, path)` / `qortex.read_recipe(path)` | Persist and reload reproducible workflow recipes |

## Selection and Planning

Qortex planning is structural and companion-aware. It reasons from typed BIDS
entities and manifest graph relationships instead of isolated filename
shortcuts.

Supported controls:

- `subjects`
- `sessions`
- `tasks`
- `modalities`
- `datatypes`
- `include` and `exclude` glob patterns
- `include_derivatives`
- `metadata_only`
- `with_companions`
- `event_complete`
- `label_ready`
- `loadable_only`
- `max_size_gb`
- `conversion_target`

A `DownloadPlan` contains:

- Dataset ID and snapshot
- Target directory
- Selected files
- Essential metadata files
- Estimated bytes and GB
- Warnings
- Logical recordings involved in the plan
- Per-file `SelectionReason` entries
- `summary()` and `explain()` methods

Example:

```python
from qortex import Dataset

ds = Dataset("ds000001")
plan = ds.plan(
    subjects=["01"],
    modalities=["fmri"],
    event_complete=True,
    max_size_gb=1.0,
)

print(plan.summary())
print(plan.explain(limit=20))
```

## Metadata-First Workflows

Qortex can inspect and triage datasets before a large neuroimaging download.

```python
from qortex import Dataset

ds = Dataset("ds000001")

metadata = ds.metadata_files()
rows = ds.first_rows("participants.tsv", n=5)
description = ds.preview("dataset_description.json", max_bytes=4096)

dry_run = ds.download_metadata(output_dir="data/ds000001-meta", dry_run=True)
print(dry_run.plan.summary())
```

This is useful for:

- Checking participants and phenotype columns
- Inspecting event labels
- Reviewing dataset description, license, authors, and DOI
- Estimating whether a full or selective download is worthwhile
- Building real tests and examples without downloading gigabytes of raw data

## Decision-First Workflows

Qortex exposes practical workflow decisions as first-class APIs and CLI
commands. These reports do not guess silently: they use `possible`,
`uncertain`, and `not_possible` states and explain the missing evidence.

```python
from qortex import Dataset, leakage_check

ds = Dataset("ds000001")

doctor = ds.doctor()
print(doctor.to_text())

minimum = ds.minimum(goal="label-check", output_dir="data/ds000001-meta")
print(minimum.to_text())

train = ds.can_train(local_path="data/ds000001-meta", target="trial_type")
print(train.to_text())

first = ds.first_batch(artifact_path="artifacts/ds000001-events", limit=5)
print(first.to_text())

content = ds.content_status("data/ds000001-meta")
print(content.to_text())

leakage = leakage_check("artifacts/ds000001-events")
print(leakage.to_text())
```

Implemented decision reports:

| Report | What It Answers |
| --- | --- |
| `DoctorReport` | Is the dataset usable, what is proven, what is uncertain, and what should the user do next? |
| `MinimumPlanReport` | What exact files are the smallest useful real download for label checking, first batch, validation, or metadata inspection? |
| `CanTrainReport` | Can supervised training start now, are labels confirmed or only candidates, what split policy is safe, and what leakage risks exist? |
| `FirstBatchReport` | Can Qortex read first rows from a converted artifact, or what download plan is required before that can happen? |
| `ContentStatusReport` | Does a local tree contain zero-byte files, pointer-like files, missing manifest files, extra files, or size mismatches? |
| `LeakageReport` | Does an artifact leak subjects or source files across splits, or include derivative-source samples that need review? |
| `Recipe` | A shareable JSON workflow description for reproducible Qortex decisions and downloads |

## Readiness and EDA

`Dataset.check()` returns a `ReadinessReport` with:

- Logical recording count
- Loadable recording count
- Event-complete count
- Label-ready count
- Estimated bytes
- Score
- Structured findings with severity, code, path, recording ID, and recommendation

`Dataset.eda()` returns an `EDAReport` with:

- Dataset summary
- Per-modality summaries
- Quality metrics
- BIDS essentials checks
- Event coverage checks
- Local event-label distributions
- Class imbalance ratios
- Optional HTML report

Example:

```python
from qortex import Dataset

ds = Dataset("ds000001")
ds.download_metadata(output_dir="data/ds000001-meta")

readiness = ds.check(local_path="data/ds000001-meta", conversion_target="sklearn")
print(readiness.summary())

report = ds.eda(
    local_path="data/ds000001-meta",
    output_html="reports/ds000001-eda.html",
)
print(report.quality.ml_readiness_score)
```

## Validation and Local Indexing

Qortex separates local-file reconciliation from official BIDS validation.

Local indexing:

```python
from qortex import Dataset

ds = Dataset("ds000001")
index = ds.index_local("data/ds000001-meta", use_pybids=False)
print(index.summary())
```

Official BIDS validation:

```python
from qortex import Dataset

ds = Dataset("ds000001")
report = ds.validate(
    local_path="data/ds000001",
    output_json="validation.json",
    refresh_cache=True,
)
report.to_markdown("validation.md")
report.to_html("validation.html")
```

If `bids-validator` is not installed, Qortex raises a clear validation error.
The real staged scenario suite reports this dependency state explicitly and
does not invent validator results.

## Conversion and Artifacts

The conversion pipeline can convert local downloaded files into ML-friendly
artifacts. Current conversion behavior is strongest for BIDS behavioral/events
tables and for modality loaders whose optional dependencies are installed.

Implemented conversion features:

- Loader registry discovery
- Per-file loader resolution
- Local file existence checks
- Optional fixed-window segmentation for signal samples
- Subject-aware splits
- Random splits
- Stratified subject splits when labels are available
- Writer dispatch
- Provenance record writing
- Artifact manifest writing
- Source-file tracking from actually loaded samples
- Sample, subject, and split counts
- Non-numeric behavioral/table row serialization in Parquet artifacts

Supported writer targets:

| Format | Status |
| --- | --- |
| Parquet | Useful/default path; real event/table artifact generation is scenario-tested |
| Zarr | Basic writer; requires optional dependencies and more roundtrip coverage |
| HDF5 | Basic writer; requires optional dependencies and more roundtrip coverage |
| WebDataset | Basic writer; needs larger-data and reader-side validation |
| HuggingFace datasets | Basic writer; requires optional dependencies and more roundtrip coverage |
| TFRecord | Basic writer; requires TensorFlow and more roundtrip coverage |

Example:

```python
from pathlib import Path

from qortex import Artifact, Dataset

ds = Dataset("ds000001")
ds.download_metadata(output_dir="data/ds000001-meta")

result = ds.convert(
    output_dir=Path("artifacts/ds000001-events"),
    output_format="parquet",
    split_strategy="subject",
    shard_size=1000,
)

print(result.n_samples)
print(result.splits)

artifact = Artifact.open("artifacts/ds000001-events")
print(artifact.summary())
```

## Supported Modalities and Loaders

Qortex has loader modules for these BIDS data categories:

| Modality | Loader Scope | Current Maturity |
| --- | --- |
| Behavior | BIDS `.tsv` and `.csv` event, participant, session, scan, and behavior tables | Useful and real-scenario tested |
| EEG | MNE-compatible EEG recordings | Real optional-dependency loader; needs broader fixture coverage |
| MEG | MNE-compatible MEG recordings | Real optional-dependency loader; needs broader fixture coverage |
| iEEG | MNE-compatible iEEG recordings | Real optional-dependency loader; needs broader fixture coverage |
| fNIRS | SNIRF and MNE-compatible fNIRS/NIRS recordings | Real optional-dependency loader; needs broader fixture coverage |
| MRI | Anatomical image files through imaging dependencies | Real optional-dependency loader; needs broader fixture coverage |
| fMRI | Functional/perfusion images and metadata | Real optional-dependency loader; needs broader fixture coverage |
| DWI | Diffusion images with bvec/bval companions | Real optional-dependency loader; needs broader fixture coverage |
| PET | PET imaging files and metadata helpers | Real optional-dependency loader; needs broader fixture coverage |

Loader discovery is lazy and fault-isolated. Missing optional dependencies for
one modality do not prevent importing Qortex or using metadata-first workflows.

## Advanced Remote Inspection (Zero-Download)

Qortex exposes a layer of analysis that requires no local download at all. It
uses HTTP Range requests, GraphQL API metadata, and concurrent CDN fetches to
extract meaningful signal from OpenNeuro datasets in seconds.

### Participant demographics from the API

```python
from qortex import Dataset

ds = Dataset("ds000117")
df = ds.participants()        # Polars DataFrame: participant_id, age, sex, group
print(df["age"].mean())       # API response — no TSV downloaded
print(df.filter(df["sex"] == "M"))
```

The API's `snapshot.summary.subjectMetadata` field returns per-subject age,
sex, and group. Qortex falls back to fetching the CDN URL of participants.tsv
only when the API returns no demographics.

### Remote events inspection

```python
df = ds.events(subject="01", task="facerecognition")
print(df.head())              # onset, duration, trial_type — fetched from CDN
```

### Remote sidecar inspection (BIDS inheritance)

```python
meta = ds.sidecar("sub-01/meg/sub-01_task-facerecognition_meg.fif")
print(meta["SamplingFrequency"])   # merged from 11 candidate sidecar paths
print(meta["RepetitionTime"])
```

### NIfTI header from 352 bytes

```python
info = ds.nifti_info("sub-01/func/sub-01_task-facerecognition_bold.nii.gz")
print(info)
# 4D fMRI 64×64×33×208  vox=3.00×3.00×4.05mm  TR=2.000s
```

For `.nii.gz` files, Qortex fetches the first 64 KB of compressed bytes and
decompresses in-memory to extract the 352-byte NIfTI-1 header. A 38 MB fMRI
volume yields full shape, TR, and voxel info using under 64 KB of network I/O.

### Label landscape analysis

```python
landscape = ds.label_landscape()
print(landscape.summary())
# Events files: 128/128 fetched
# Classes: 5  Total events: 12,800
# Imbalance: 1.08x (balanced)
# ISI jitter: task-face CV=0.03 (fixed-rate)
# Cross-subject consistency: 96.1%

print(landscape.imbalance_severity)   # "balanced"
print(landscape.recommendations)
```

`label_landscape()` concurrently fetches all events TSVs using an async batch
with configurable concurrency (default 24), then computes:

- Trial type frequencies and per-subject profiles
- Class imbalance ratio and severity (balanced / moderate / severe / critical)
- Inter-stimulus interval jitter coefficient of variation per task
- Cross-subject consistency: fraction of subjects with the full global class set
- Actionable ML recommendations

### Signal budget estimation

```python
budget = ds.signal_budget()
print(budget.estimate_windows(window_duration_s=2.0, overlap=0.5))
# {'meg': 183200, 'eeg': 42100}

plan = budget.minimum_download_for_n_windows(target=10000, window_s=2.0)
print(plan)
# {'subjects_needed': 4, 'windows_achieved': 11200}
```

`signal_budget()` fetches JSON sidecars for all signal files concurrently and
extracts `SamplingFrequency`, `RecordingDuration`, `EEGChannelCount`, TR, and
discard volumes. For fMRI files missing TR or volume counts in the sidecar, it
falls back to fetching the NIfTI header (352 bytes) automatically.

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

selector = DatasetSelector()
ranking = selector.find(goal, limit=10)
for fit in ranking:
    print(fit.summary_line())

# ✓ ds004362         score=91.2/100 [A]  modality=1.00 | subject_count=0.90 | ...
# ✓ ds003490         score=74.6/100 [B]  modality=1.00 | ...
```

`DatasetSelector` works in three lazily-escalating tiers:

| Tier | Trigger | What it checks |
| --- | --- | --- |
| 1 — Catalog | Always | Modality, subject count, size, license (local, fast) |
| 2 — API | Default | Full metadata, engagement, BIDS version, demographics |
| 3 — Remote events | Optional (`tier3_events=True`) | Class count, imbalance, trials-per-class via `LabelLandscape` |

Each `DatasetFitness` result carries a per-dimension breakdown with score,
weight, observed value, target value, and grade (A–F), plus a `report()` method
for full transparency.

## Catalog Search

Qortex includes a normalized local catalog index for OpenNeuro discovery. The
ingester stores dataset metadata, latest snapshot metadata, authors,
modalities, tasks, derived keywords, and optional recursive file-summary
digests. Deep ingestion counts file extensions, BIDS datatypes, suffixes, event
files, derivative files, primary files, and metadata files without downloading
the raw dataset content.

```python
import qortex

qortex.configure(cache_dir=".qortex-cache")
qortex.refresh_catalog(max_pages=1, progress=False)

results = qortex.search(
    query="auditory",
    modality="eeg",
    min_subjects=20,
    limit=10,
)
for row in results:
    print(row["dataset_id"], row["name"], row["score"], row["tasks"])

profile = qortex.refresh_catalog_dataset("ds000001", include_file_summary=True)
print(profile["n_event_files"], profile["file_summaries"][:5])
```

Supported catalog search filters:

- Free-text query over dataset ID, name, description, DOI, authors, modalities,
  tasks, keywords, and license
- `modality`
- `task`
- `author`
- `license`
- `min_subjects`
- `max_size_gb`
- `has_events`
- `has_derivatives`
- `limit` and `offset`

Catalog profiles expose:

- Dataset ID, name, DOI, license, authors, latest snapshot, subjects, sessions,
  tasks, modalities, files, and bytes
- Raw OpenNeuro metadata/description JSON retained for downstream inspection
- `has_events`, `has_derivatives`, event-file count, derivative-file count,
  primary-file count, metadata-file count
- File-summary facets by extension, datatype, and suffix

The catalog uses DuckDB when available and falls back to SQLite.

## CLI Commands

| Command | Purpose |
| --- | --- |
| `qortex search` | Search the local OpenNeuro catalog |
| `qortex inspect` | Fetch and summarize a dataset manifest |
| `qortex metadata` | List or download metadata-only files |
| `qortex preview` | Preview first rows/text of a remote/local file |
| `qortex plan` | Compute a download plan without downloading |
| `qortex download` | Download a dataset or selected subset |
| `qortex doctor` | Explain dataset usability, uncertainty, findings, and next actions |
| `qortex minimum` | Compute the smallest real download for label check, first batch, validation, or metadata |
| `qortex can-train` | Decide whether supervised training is possible, uncertain, or blocked |
| `qortex first-batch` | Print first artifact rows or the required first-batch download plan |
| `qortex content-status` | Check local files, pointer-like content, and manifest mismatches |
| `qortex leakage-check` | Check converted artifacts for subject/source split leakage |
| `qortex make-recipe` | Write a reusable workflow recipe JSON |
| `qortex run-recipe` | Load a recipe and run its minimum download decision |
| `qortex validate` | Run official BIDS Validator and normalize its report |
| `qortex local-index` | Index a local BIDS tree and optionally reconcile with a saved manifest |
| `qortex eda` | Run EDA and optionally write HTML |
| `qortex convert` | Convert a downloaded dataset into an ML artifact |
| `qortex cache` | Inspect and manage local cache/registry state |
| `qortex login` | Save or remove an OpenNeuro API token |
| `qortex catalog-refresh` | Refresh the local OpenNeuro catalog |
| `qortex catalog-profile` | Print or refresh one digested catalog dataset profile |
| `qortex dashboard` | Launch the Streamlit dashboard entrypoint when dashboard extras are installed |

## Real Scenario Suite

The repository includes a no-pytest real scenario suite under `test/`.

Run all scenarios:

```bash
python test/run_all.py
```

The runner uses real OpenNeuro data. By default it uses `ds000001`, and you can
override this with:

```bash
QORTEX_REAL_TEST_DATASET=ds000001 QORTEX_REAL_TEST_SNAPSHOT=1.0.0 python test/run_all.py
```

The suite covers:

| Stage | Workflow |
| --- | --- |
| `0_import_config` | Installed package import and configuration |
| `1_manifest_models` | Real OpenNeuro manifest and semantic graph |
| `2_selection_planning` | Real primary-file selection and companion closure |
| `3_remote_preview_project` | Remote metadata preview without full download |
| `4_download_specific_parts_project` | Metadata-only and exact-path dry-run plans |
| `5_eda_events` | One real metadata download, EDA, event-label summaries |
| `6_conversion_artifact` | Real event/table Parquet artifact generation |
| `7_readiness_report_project` | Readiness score and accounting |
| `8_behavior_loader_project` | Real BIDS events loader behavior |
| `9_window_split_project` | Real event samples and subject-safe split allocation |
| `10_local_index_validation_cache_project` | Local index, validation dependency handling, report exports |
| `11_catalog_project` | Real catalog refresh and search |
| `12_cli_project` | Installed CLI against real metadata |
| `13_dataset_facade_project` | High-level `Dataset` facade workflow |
| `14_live_openneuro_metadata_project` | Live manifest and metadata smoke check |
| `15_decision_workflows_project` | Real doctor, minimum, can-train, first-batch, content-status, and recipe workflow |
| `16_catalog_ingestion_project` | Deep catalog ingestion, task/event search, facets, and file-summary digestion |
| `17_remote_inspection_project` | `participants()`, `events()`, `sidecar()`, `nifti_info()` — zero-download intelligence |
| `18_label_landscape_project` | Concurrent remote events fetch, class balance, ISI jitter, cross-subject consistency |
| `19_signal_budget_project` | Remote sidecar + NIfTI header acquisition params, window estimates, minimum subset planning |
| `20_dataset_selector_project` | `ResearchGoal` + `DatasetSelector` fitness scoring against known and catalog-searched datasets |

`test/run_all.py` shares one real metadata download across downstream stages so
the suite does not redownload the same event/sidecar files for every project.

## Current Boundaries

Qortex is strict about what it can prove.

- Decision-first commands are implemented, but they report uncertainty when
  local evidence is missing instead of treating remote manifests as proof.
- Semantic Atlas-style search over dataset meaning and tasks is still future
  work; current catalog search is structured/local-catalog based.
- Confirmed label readiness requires local event-file inspection.
- `Dataset.validate()` requires the external official `bids-validator` CLI.
- Torch and sklearn convenience adapters currently expect Parquet artifacts.
- Metadata-first workflows are mature; full raw neuroimaging conversion depends
  on optional modality libraries and loader maturity.
- Event-aligned windowing is exposed through `Dataset.convert(event_aligned=True, window_tmin=...)`,
  but still needs broader real-signal scenario coverage before it can be considered production-grade.
- Dashboard modules exist, but the dashboard is not yet a complete product
  surface.
- Some advanced platform features are intentionally future work: DataLad
  backend, cloud export, deep signal/image EDA, full repair workflow, advanced
  samplers/collators, and benchmark harnesses.

## Project Structure

| Package | Role |
| --- | --- |
| `qortex.client` | OpenNeuro API, auth, and transport |
| `qortex.catalog` | Local searchable dataset catalog |
| `qortex.manifest` | Manifest building, BIDS parsing, graph, sidecars, diffs |
| `qortex.plan` | Selection and download planning |
| `qortex.fetch` | Download backends, cache, and execution |
| `qortex.check` | Readiness analysis |
| `qortex.eda` | Summaries, quality metrics, reports, plots |
| `qortex.parse` | Modality loaders and loader registry |
| `qortex.convert` | Windowing, splitting, provenance, artifact writers |
| `qortex.artifact` | Converted artifact access |
| `qortex.train` | ML framework adapters |
| `qortex.indexing` | Local BIDS indexing and reconciliation |
| `qortex.validation` | BIDS Validator wrapper, cache, report diff |
| `qortex.lake` | Local data lake layout and registry |
| `qortex.cli` | Command-line interface |

## Roadmap

Near-term work should preserve the current manifest-first architecture and add
depth in these areas:

- Deepen decision workflows with richer split diagnostics, session/run leakage
  detection, class-imbalance thresholds, expected batch-shape contracts, and
  recipe execution logs.
- Map validation issues back to `FileRecord` and `LogicalRecording`.
- Extend readiness scoring with license, citation, class imbalance, split risk,
  and local storage feasibility.
- Add deep signal/image EDA: sampling-rate, duration, channel stats, PSD,
  image shape, voxel size, affine, TR, and outliers.
- Wire event-aligned windows into high-level conversion.
- Add normalization, resampling, filtering, feature extraction, resumable
  conversion, and conversion lockfiles.
- Add DataLad/git-annex retrieval as an optional backend.
- Add cache pruning, offline mode, integrity audits, and materialization
  policies.
- Add cloud/object-store export through `fsspec`.
- Add unified adapter methods such as `to_torch`, `to_tfdata`,
  `to_huggingface`, `to_ray`, `to_dask`, and `to_braindecode`.
- Add advanced dataloader utilities: transforms, collators, distributed
  samplers, subject-balanced samplers, and class-balanced samplers.
- Build dashboard pages on the stable core APIs.
- Add benchmarks for manifest speed, planning speed, download throughput,
  conversion throughput, dataloader throughput, memory use, and loadability.
