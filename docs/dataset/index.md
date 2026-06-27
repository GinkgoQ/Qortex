# Dataset

Qortex provides two main entry points for working with OpenNeuro datasets:

- **`Dataset`** — the high-level facade for a single dataset: inspect, download, convert, visualize.
- **`OpenNeuroClient`** — the low-level GraphQL client for rich metadata, search, README, and BIDS validation.

## Dataset — single-dataset facade

```python
from qortex import Dataset

ds = Dataset("ds004130")
ds = Dataset("ds004130", snapshot="1.2.0")            # pin version
ds = Dataset("ds004130", data_dir=Path("/data/ds004130"))  # local dir
```

Creating a `Dataset` object makes no network calls. Network access begins when you call a method that needs the manifest.

## OpenNeuroClient — rich metadata and search

```python
from qortex.client import OpenNeuroClient

with OpenNeuroClient() as client:
    info    = client.get_dataset_rich("ds008039")          # full metadata in one call
    readme  = client.get_readme("ds008039")                 # description text
    issues  = client.get_validation_issues("ds008039", "1.0.0")  # BIDS issues
    snaps   = client.get_snapshots("ds008039")              # all versions

    results = client.search_datasets_rich(
        modality="eeg", min_subjects=30, sort_by="downloads"
    )
```

## Pages in this section

[**Inspect**](inspect.md) — Rich API metadata (`RichDatasetInfo`) and manifest-level profiles (`DatasetProfile`). Covers every field visible on an OpenNeuro dataset page, including engagement metrics, demographics, funding, README, and BIDS validation.

[**Metadata**](metadata.md) — Read sidecar JSON, events.tsv, participants.tsv, and dataset_description.json without downloading imaging files.

[**Search & filter**](search-catalog.md) — `DatasetQuery` fluent builder with 10 filters, local catalog, live API search, `search_datasets_rich()` with engagement sorting.

[**Snapshots**](snapshots.md) — List available snapshot versions, compare manifests between snapshots.

[**BIDS entities**](bids-entities.md) — The entity labels parsed from file paths: subject, session, task, run, datatype, suffix.
