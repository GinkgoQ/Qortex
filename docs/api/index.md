# API Reference

Reference is auto-generated from source docstrings via mkdocstrings.

| Module | Entry point | What it covers |
|---|---|---|
| [Dataset](dataset.md) | `qortex.Dataset` | Inspect, download, convert, visualize — the main facade |
| [Visualization](visualization.md) | `qortex.visualize` | Viewers, overlays, QC figures, audit reports |
| [Artifact](artifact.md) | `qortex.Artifact` | Open and use a converted ML artifact |
| [Conversion](conversion.md) | `qortex.convert` | `ConversionPipeline`, `WindowSpec`, `SplitSpec` |
| [Datasets](datasets.md) | `qortex.datasets` | Keras-style loaders for open neuroscience datasets |
| [Neuroclassic](neuroclassic.md) | `qortex.neuroclassic` | Signal/image QC, Pearson/PLV connectivity, graph metrics, CSP, entropy/complexity, stats, splits |
| [CLI](cli.md) | `qortex <command>` | All CLI commands with options |

---

## OpenNeuroClient

Direct access to the OpenNeuro GraphQL API. Use when you need rich metadata without the `Dataset` facade.

```python
from qortex.client import OpenNeuroClient

with OpenNeuroClient() as client:
    info    = client.get_dataset_rich("ds008039")
    readme  = client.get_readme("ds008039")
    issues  = client.get_validation_issues("ds008039", "1.0.0")
    snaps   = client.get_snapshots("ds008039")
    results = client.search_datasets_rich(modality="eeg", sort_by="downloads")
```

| Method | Returns | Description |
|---|---|---|
| `get_dataset_rich(id)` | `RichDatasetInfo` | All metadata for one dataset |
| `get_readme(id, tag?)` | `str \| None` | README / description text |
| `get_validation_issues(id, tag)` | `list[dict]` | BIDS validation errors and warnings |
| `get_snapshot_summary(id, tag)` | `SnapshotSummary` | Subjects, sessions, demographics |
| `get_snapshots(id)` | `list[SnapshotRef]` | All available versions |
| `get_files(id, tag?)` | `(SnapshotRef, list[dict])` | Full file tree |
| `search_datasets_rich(...)` | `list[RichDatasetInfo]` | Filtered rich search |
| `get_dataset(id)` | `DatasetRef` | Lightweight metadata |

---

## DatasetQuery

Fluent search builder for local catalog and live API searches.

```python
from qortex.catalog import DatasetQuery

results = (
    DatasetQuery()
    .modality("eeg")
    .min_subjects(30)
    .has_events()
    .containing("auditory")
    .limit(20)
    .fetch()        # local catalog (fast, offline)
)

live = DatasetQuery().modality("eeg").min_subjects(30).live()  # OpenNeuro API
```

See [Search & filter](../dataset/search-catalog.md) for the full filter reference.
