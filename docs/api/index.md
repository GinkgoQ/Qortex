# API Reference

The API reference is generated from docstrings in the source code using mkdocstrings.

[**Dataset**](dataset.md) — `qortex.Dataset` — the main entry point: inspect, download, convert, visualize

[**Visualization**](visualization.md) — `qortex.visualize` — viewers, overlays, audit

[**Artifact**](artifact.md) — `qortex.Artifact` — ML-ready artifact loading

[**Conversion**](conversion.md) — `qortex.convert` — ConversionPipeline, WindowSpec, SplitSpec

[**CLI**](cli.md) — All CLI commands with options and examples

## OpenNeuroClient

The `OpenNeuroClient` provides direct access to the OpenNeuro GraphQL API. It is the primary way to fetch rich dataset metadata without the overhead of manifest parsing.

```python
from qortex.client import OpenNeuroClient

with OpenNeuroClient() as client:
    info    = client.get_dataset_rich("ds008039")
    readme  = client.get_readme("ds008039")
    issues  = client.get_validation_issues("ds008039", "1.0.0")
    snaps   = client.get_snapshots("ds008039")
    summary = client.get_snapshot_summary("ds008039", "1.0.0")
    results = client.search_datasets_rich(modality="eeg", sort_by="downloads")
```

Key methods:

| Method | Returns | Description |
|--------|---------|-------------|
| `get_dataset_rich(id)` | `RichDatasetInfo` | All metadata for one dataset |
| `get_readme(id, tag?)` | `str \| None` | README / description text |
| `get_validation_issues(id, tag)` | `list[dict]` | BIDS validation errors and warnings |
| `get_snapshot_summary(id, tag)` | `SnapshotSummary` | Subjects, sessions, demographics |
| `get_snapshots(id)` | `list[SnapshotRef]` | All available versions |
| `get_files(id, tag?)` | `(SnapshotRef, list[dict])` | Full file tree |
| `search_datasets_rich(...)` | `list[RichDatasetInfo]` | Filtered rich search |
| `get_dataset(id)` | `DatasetRef` | Lightweight metadata |

## DatasetQuery

`DatasetQuery` is the fluent search builder for local catalog and live API searches.

```python
from qortex.catalog import DatasetQuery

results = (
    DatasetQuery()
    .modality("eeg")
    .min_subjects(30)
    .has_events()
    .containing("auditory")
    .limit(20)
    .fetch()        # local catalog
)

# Or live from API
live = DatasetQuery().modality("eeg").min_subjects(30).live()
```

See [Search & filter](../dataset/search-catalog.md) for the full filter reference.
