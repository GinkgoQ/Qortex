# Data Model

Qortex has a layered data model. Each layer adds more specific information as you move through the pipeline.

## RichDatasetInfo

`RichDatasetInfo` is the primary metadata object returned by `OpenNeuroClient.get_dataset_rich()`. It covers every field visible on an OpenNeuro dataset page without downloading or iterating the file tree.

Key fields:

| Field | Description |
|-------|-------------|
| `id`, `name`, `doi` | Dataset identity |
| `license`, `authors`, `senior_author` | Attribution |
| `modalities`, `tasks`, `species` | Scientific context |
| `study_domain`, `study_design` | Study type |
| `grant_funder`, `grant_id` | Funding |
| `associated_paper_doi` | Linked publication |
| `created`, `publish_date` | Timestamps |
| `engagement` | `DatasetEngagement` — views, downloads, stars, followers |
| `latest_snapshot_summary` | `SnapshotSummary` — subjects, sessions, tasks, file counts, demographics |
| `data_processed` | True if preprocessed data is included |

```python
from qortex.client import OpenNeuroClient

with OpenNeuroClient() as client:
    info = client.get_dataset_rich("ds008039")

snap = info.latest_snapshot_summary
print(snap.n_subjects, snap.total_size_gb, snap.bids_version)
print(snap.funding, snap.references, snap.ethics_approvals)
print(snap.demographics_dataframe())   # Polars DataFrame: age, sex, group
```

## DatasetEngagement

`DatasetEngagement` tracks community activity for a dataset on the OpenNeuro platform.

```python
info.engagement.views             # integer
info.engagement.downloads         # integer
info.engagement.stars             # integer
info.engagement.followers         # integer
info.engagement.popularity_score  # composite 0–100 score
```

## SnapshotSummary

`SnapshotSummary` is the API-level BIDS summary for one snapshot. It is returned by `get_dataset_rich()` (as `latest_snapshot_summary`) and by `get_snapshot_summary()`. It does not require a file tree fetch.

```python
snap.tag                # "1.2.0"
snap.n_subjects         # len(snap.subjects)
snap.subjects           # ["sub-01", "sub-02", ...]
snap.sessions           # ["ses-01", "ses-02"]
snap.tasks              # ["rest", "nback"]
snap.modalities         # ["MRI"]
snap.total_files        # integer
snap.total_size_gb      # float
snap.bids_version       # "1.8.0"
snap.license            # "CC0"
snap.funding            # ["NIH R01 MH123456", ...]
snap.references         # ["https://doi.org/...", ...]
snap.ethics_approvals   # ["IRB approval #2024-001"]
snap.how_to_acknowledge # citation string
snap.data_processed     # bool
snap.subject_demographics  # list[SubjectDemographic]
snap.demographics_dataframe()  # Polars DataFrame
snap.age_stats()        # {"n": 42, "mean": 28.4, "min": 18, "max": 65, ...}
```

## Dataset

`Dataset` is the top-level facade. It holds a dataset ID, an optional snapshot tag, and an optional local data directory. It is cheap to create — no network calls happen at construction time.

```python
from qortex import Dataset

ds = Dataset("ds004130")                        # no network call yet
ds = Dataset("ds004130", snapshot="1.2.0")      # pin a snapshot
ds = Dataset("ds004130", data_dir=Path("/data")) # override default cache
```

The manifest is fetched lazily on the first call to `ds.manifest()`, `ds.download()`, `ds.doctor()`, or any other method that needs it.

## Manifest

A `Manifest` is the result of fetching the remote file tree and parsing it into typed records. It contains:

- `dataset_id` and `snapshot` tag
- `doi` (dataset DOI)
- `files` — a list of `FileRecord` objects
- `summary` — a `ManifestSummary` with pre-computed counts

```python
manifest = ds.manifest()
print(manifest.summary.n_subjects)     # integer
print(manifest.summary.modalities)     # list of strings
print(manifest.summary.has_events)     # bool
print(manifest.summary.total_size)     # bytes
```

The manifest is cached in memory. Repeated calls to `ds.manifest()` return the cached copy unless `force_refresh=True` is passed.

## FileRecord

A `FileRecord` represents one file in the manifest. It has:

- `path` — BIDS-relative path (e.g., `sub-01/func/sub-01_task-rest_bold.nii.gz`)
- `size` — file size in bytes (may be None for some endpoints)
- `urls` — list of CDN and S3 URLs for this file
- `subject`, `session`, `task`, `run`, `datatype`, `suffix` — parsed BIDS entities
- `extension` — file extension (`.nii.gz`, `.tsv`, etc.)
- `is_dir` — True for directory entries

FileRecords are read-only. Filtering happens through `manifest.filter()` or `ds.files()`.

## SelectionSpec

A `SelectionSpec` describes which files to include in a download plan. It combines:

- Entity filters: `subjects`, `sessions`, `tasks`, `modalities`, `datatypes`
- Glob patterns: `include` and `exclude` lists
- Semantic filters: `event_complete`, `label_ready`, `loadable_only`
- Size limit: `max_size_gb`

The planner resolves a SelectionSpec against a Manifest to produce a `DownloadPlan`.

## DownloadPlan

A `DownloadPlan` lists the exact files to download, with target paths and companion files included. It is produced by `DownloadPlanner.plan()` before any transfer happens.

```python
plan = ds.plan(subjects=["01", "02"], suffixes=["bold"])
print(len(plan.files))          # file count
print(plan.target_dir)          # destination path
print(plan.size_gb)             # GB
```

## DatasetProfile

`DatasetProfile` is the result of `DatasetInspector.inspect()`. It builds on the file tree manifest to produce a full structural breakdown with ML readiness scoring.

```python
profile = ds.inspect()
print(profile.n_subjects)           # from manifest
print(profile.events_coverage)      # fraction with events.tsv
print(profile.ml_readiness.grade)   # "A"–"F"
print(profile.rich_info)            # RichDatasetInfo (if available)
```

## Artifact

An `Artifact` is the output of `ConversionPipeline.run()`. It lives on local disk at a directory containing an `artifact_manifest.json` and one subdirectory per split (`train/`, `val/`, `test/`). Each split contains Parquet (or Zarr/HDF5/etc.) shards.

```python
from qortex import Artifact

art = Artifact.open("converted/parquet/")
print(art.manifest.n_samples)
print(art.manifest.splits)      # {"train": ..., "val": ..., "test": ...}

X, y = art.sklearn()
ds_torch = art.torch(split="train")
```

## Type relationships

```
OpenNeuroClient
  ├─ get_dataset_rich()       →  RichDatasetInfo
  │     ├─ DatasetEngagement
  │     └─ SnapshotSummary
  │           └─ [SubjectDemographic, ...]
  ├─ get_readme()             →  str | None
  ├─ get_validation_issues()  →  list[dict]
  ├─ get_snapshots()          →  list[SnapshotRef]
  ├─ get_files()              →  (SnapshotRef, list[dict])
  └─ search_datasets_rich()   →  list[RichDatasetInfo]

Dataset (facade)
  └─ manifest()  →  Manifest
        ├─ ManifestSummary
        └─ [FileRecord, ...]

DatasetInspector.inspect()  →  DatasetProfile
  ├─ DatasetRef
  ├─ SnapshotRef
  ├─ MLReadinessScore
  └─ RichDatasetInfo (optional)

DownloadPlanner.plan()  →  DownloadPlan
  └─ [FileRecord, ...]

ConversionPipeline.run()  →  Artifact
  ├─ ArtifactManifest
  └─ split/
       └─ *.parquet (or .zarr, .h5, etc.)
```
