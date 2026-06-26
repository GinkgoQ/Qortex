# Dataset

The `Dataset` class is the entry point for all Qortex operations. Create one with an OpenNeuro dataset ID and optionally a snapshot version.

```python
from qortex import Dataset

ds = Dataset("ds004130")
ds = Dataset("ds004130", snapshot="1.2.0")   # pin version
ds = Dataset("ds004130", data_dir=Path("/data/ds004130"))  # local dir
```

Creating a `Dataset` object makes no network calls. Network access begins when you call a method that needs the manifest.

## What you can do with a Dataset

[**Inspect**](inspect.md) — Fetch the manifest and read structural properties: subject count, modalities, file sizes, companion coverage.

[**Metadata**](metadata.md) — Read sidecar JSON, events.tsv, participants.tsv, and dataset_description.json without downloading imaging files.

[**Search catalog**](search-catalog.md) — Filter the local DuckDB catalog of indexed datasets by modality, task, subject count, and size.

[**Snapshots**](snapshots.md) — List available snapshot versions, compare manifests between snapshots.

[**BIDS entities**](bids-entities.md) — The entity labels parsed from file paths: subject, session, task, run, datatype, suffix.
