# Qortex

Qortex by GinkgoQ is a production-oriented Python library for working with
OpenNeuro and BIDS datasets. It provides a real dataset workflow: discover a
dataset, inspect its manifest, preview metadata, plan selective downloads,
download only what is needed, analyze readiness, summarize labels, convert local
tables/events into ML-ready artifacts, and open those artifacts through ML
adapters.

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
readiness = ds.check(local_path=metadata_dir, conversion_target="sklearn")
print(readiness.summary())

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
| EDA/QC | Dataset summaries, modality summaries, coverage, quality scores, event-label summaries, HTML reports |
| Loaders | Lazy plugin registry and modality-aware loaders for behavior, EEG, MEG, iEEG, fNIRS, MRI, fMRI, DWI, and PET |
| Conversion | Loader resolution, optional windowing, split assignment, writer dispatch, provenance, artifact manifests |
| Artifacts | Reopen converted artifacts, inspect sample/split/source metadata, hand off to adapters |
| ML adapters | Torch, Lightning, sklearn, TensorFlow, HuggingFace, Ray, Dask, and Braindecode modules |
| Catalog | Local OpenNeuro catalog refresh and search backed by DuckDB or SQLite fallback |
| CLI | Search, inspect, metadata, preview, plan, download, validate, local-index, EDA, convert, cache, login |

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
| `validate(...)` | Run official BIDS Validator and return a typed `ValidationReport` |
| `index_local(...)` | Index a local BIDS tree and reconcile it with the manifest |
| `eda(...)` | Run EDA/QC and optionally write an HTML report |
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
| `qortex.Artifact.open(path)` | Open a converted Qortex artifact |
| `Artifact.summary()` | Return artifact ID, dataset, snapshot, format, samples, subjects, and split counts |
| `Artifact.torch(...)` | Open a Parquet artifact for Torch |
| `Artifact.sklearn(...)` | Open a Parquet artifact for sklearn |

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
| Parquet | Implemented, default tabular artifact format |
| Zarr | Implemented writer, requires optional dependencies |
| HDF5 | Implemented writer, requires optional dependencies |
| WebDataset | Implemented writer |
| HuggingFace datasets | Implemented writer, requires optional dependencies |
| TFRecord | Implemented writer, requires TensorFlow |

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

| Modality | Loader Scope |
| --- | --- |
| Behavior | BIDS `.tsv` and `.csv` event, participant, session, scan, and behavior tables |
| EEG | MNE-compatible EEG recordings |
| MEG | MNE-compatible MEG recordings |
| iEEG | MNE-compatible iEEG recordings |
| fNIRS | SNIRF and MNE-compatible fNIRS/NIRS recordings |
| MRI | Anatomical image files through imaging dependencies |
| fMRI | Functional/perfusion images and metadata |
| DWI | Diffusion images with bvec/bval companions |
| PET | PET imaging files and metadata helpers |

Loader discovery is lazy and fault-isolated. Missing optional dependencies for
one modality do not prevent importing Qortex or using metadata-first workflows.

## Catalog Search

Qortex includes a local catalog index for OpenNeuro discovery.

```python
import qortex
from qortex.catalog.refresh import refresh

qortex.configure(cache_dir=".qortex-cache")
refresh(max_pages=1, progress=False)

results = qortex.search(modality="eeg", min_subjects=20, limit=10)
for row in results:
    print(row["dataset_id"], row["name"], row["n_subjects"])
```

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
| `qortex validate` | Run official BIDS Validator and normalize its report |
| `qortex local-index` | Index a local BIDS tree and optionally reconcile with a saved manifest |
| `qortex eda` | Run EDA and optionally write HTML |
| `qortex convert` | Convert a downloaded dataset into an ML artifact |
| `qortex cache` | Inspect and manage local cache/registry state |
| `qortex login` | Save or remove an OpenNeuro API token |
| `qortex catalog-refresh` | Refresh the local OpenNeuro catalog |
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

`test/run_all.py` shares one real metadata download across downstream stages so
the suite does not redownload the same event/sidecar files for every project.

## Current Boundaries

Qortex is strict about what it can prove.

- Confirmed label readiness requires local event-file inspection.
- `Dataset.validate()` requires the external official `bids-validator` CLI.
- Torch and sklearn convenience adapters currently expect Parquet artifacts.
- Metadata-first workflows are mature; full raw neuroimaging conversion depends
  on optional modality libraries and loader maturity.
- Event-aligned windowing exists at the lower level but is not fully exposed
  through high-level `Dataset.convert()` configuration.
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
