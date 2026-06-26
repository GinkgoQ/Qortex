# Data Model

Qortex has a layered data model. Each layer adds more specific information as you move through the pipeline.

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
- `is_essential` — True for root-level metadata files (participants.tsv, dataset_description.json, etc.)

FileRecords are read-only. Filtering happens through `manifest.filter()` or `ds.files()`.

## SelectionSpec

A `SelectionSpec` describes which files to include in a download plan. It combines:

- Entity filters: `subjects`, `sessions`, `tasks`, `modalities`, `datatypes`
- Glob patterns: `include` and `exclude` lists
- Semantic filters: `event_complete`, `label_ready`, `loadable_only`
- Size limit: `max_size_gb`

The planner resolves a SelectionSpec against a Manifest to produce a `DownloadPlan`.

## DownloadPlan

A `DownloadPlan` lists the exact files to download, with target paths and companion files included. It is produced by `DownloadPlanner.plan()` before any transfer happens. This lets you inspect or serialize the plan before committing to it.

```python
plan = ds.plan(subjects=["01", "02"], suffixes=["bold"])
print(len(plan.files))          # file count
print(plan.target_dir)          # destination path
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

## Relationships

```
Dataset
  └─ Manifest
       ├─ ManifestSummary
       └─ [FileRecord, ...]
            └─ (urls, BIDS entities, size, companions)

DownloadPlan
  └─ [FileRecord, ...]

ConversionPipeline
  └─ Artifact
       ├─ ArtifactManifest
       └─ split/
            └─ *.parquet (or .zarr, .h5, etc.)
```
