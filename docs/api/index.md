# API Reference

The reference is generated from source docstrings with `mkdocstrings`. Use this page to choose the right public surface before opening a full module page.

## Main Surfaces

| Need | Use | Reference |
|---|---|---|
| Inspect, assess, download, visualize, or convert an OpenNeuro/BIDS dataset | `qortex.Dataset` | [Dataset](dataset.md) |
| Open converted data and bridge it to ML libraries | `qortex.Artifact` | [Artifact](artifact.md) |
| Run visual QC, overlays, viewers, or sample audits | `qortex.visualize` | [Visualization](visualization.md) |
| Convert local BIDS data into ML formats | `qortex.convert` | [Conversion](conversion.md) |
| Load tutorial datasets with typed bundles | `qortex.datasets` | [Datasets](datasets.md) |
| Compute signal QC, EEG features, graph metrics, statistics, and leakage-safe splits | `qortex.neuroclassic` | [Neuroclassic](neuroclassic.md) |
| Use Qortex from the shell | `qortex <command>` | [CLI](cli.md) |

## Decision Objects

Qortex APIs usually return reports rather than booleans. Reports carry evidence, blockers, warnings, suggested next actions, and serialization methods.

| Report family | Typical entry point | What to inspect |
|---|---|---|
| Dataset readiness | `Dataset.doctor()` | status, findings, summary, next actions |
| Training readiness | `Dataset.can_train()` | label evidence, records ready, split policy |
| Minimum subset | `Dataset.minimum()` | plan files, estimated bytes, reason |
| Local content | `Dataset.content_status()` | missing, zero-byte, pointer-like, mismatched files |
| Artifact integrity | `Artifact.leakage_check()` | subject/source overlap across splits |
| NeuroAI compatibility | `Pipeline.check()` | blockers, required transforms, uncertainty |

## OpenNeuro Client

Use `OpenNeuroClient` when you need direct GraphQL metadata without the `Dataset` facade.

```python
from qortex.client import OpenNeuroClient

with OpenNeuroClient() as client:
    info = client.get_dataset_rich("ds000001")
    readme = client.get_readme("ds000001")
    issues = client.get_validation_issues("ds000001", "1.0.0")
    snaps = client.get_snapshots("ds000001")
    results = client.search_datasets_rich(modality="eeg", sort_by="downloads")
```

| Method | Returns | Use it for |
|---|---|---|
| `get_dataset_rich(id)` | `RichDatasetInfo` | Full metadata for one dataset. |
| `get_readme(id, tag?)` | `str \| None` | Dataset description text. |
| `get_validation_issues(id, tag)` | `list[dict]` | BIDS validator issues published by OpenNeuro. |
| `get_snapshot_summary(id, tag)` | `SnapshotSummary` | Subjects, sessions, and demographics. |
| `get_snapshots(id)` | `list[SnapshotRef]` | Available dataset versions. |
| `get_files(id, tag?)` | `(SnapshotRef, list[dict])` | Remote file tree. |
| `search_datasets_rich(...)` | `list[RichDatasetInfo]` | Live catalog search. |
| `get_dataset(id)` | `DatasetRef` | Lightweight dataset metadata. |

## Dataset Search

Use `DatasetQuery` for local catalog search first, then live OpenNeuro search when you need fresh results.

```python
from qortex.catalog import DatasetQuery

results = (
    DatasetQuery()
    .modality("eeg")
    .min_subjects(30)
    .has_events()
    .containing("auditory")
    .limit(20)
    .fetch()
)

live = DatasetQuery().modality("eeg").min_subjects(30).live()
```

See [Search catalog](../dataset/search-catalog.md) for filters, facets, and ranking behavior.

## Import Rule

Top-level imports are intentionally small:

```python
from qortex import Dataset, Artifact
```

Use subpackages when the task is specialized:

```python
from qortex.datasets import eegbci
from qortex.neuroclassic import compute_epoch_feature_matrix
from qortex.neuroai import Pipeline
```
