# Search & Filter

Qortex offers two search paths: a local DuckDB catalog (fast, offline, incremental) and live search against the OpenNeuro GraphQL API (always current, no prior refresh needed). Both expose the same fluent `DatasetQuery` interface.

## Quick reference

| Need | Method |
| --- | --- |
| Fast filtered search (offline) | `DatasetQuery().modality("eeg").fetch()` |
| Always-current search | `DatasetQuery().modality("eeg").live()` |
| Rich metadata + engagement sort | `client.search_datasets_rich(modality="eeg", sort_by="downloads")` |
| Free-text across name/description | `DatasetQuery().containing("auditory cortex").fetch()` |
| Single dataset — all metadata | `client.get_dataset_rich("ds008039")` |

## DatasetQuery — fluent builder

`DatasetQuery` chains filter methods and terminates with `.fetch()` (local catalog) or `.live()` (API).

```python
from qortex.catalog import DatasetQuery

# Local catalog — fast, works offline
results = (
    DatasetQuery()
    .modality("eeg")
    .task("rest")
    .min_subjects(30)
    .containing("auditory")
    .has_events()
    .limit(20)
    .fetch()
)

for row in results:
    print(row["dataset_id"], row["n_subjects"], row["size_gb"])
```

### Available filters

| Method | Type | Description |
| --- | --- | --- |
| `.containing(text)` | str | Free-text match on dataset name and description |
| `.modality(mod)` | str | Filter by modality: `"eeg"`, `"mri"`, `"fmri"`, `"meg"`, `"ieeg"`, `"pet"`, `"nirs"` |
| `.task(task)` | str | Filter by task name (substring match) |
| `.author(name)` | str | Filter by author name (substring match) |
| `.license(name)` | str | Filter by license: `"CC0"`, `"CC-BY"`, `"PDDL"`, etc. |
| `.min_subjects(n)` | int | Minimum number of subjects |
| `.max_size_gb(gb)` | float | Maximum total dataset size in GB |
| `.has_events(True)` | bool | Only datasets with events.tsv files |
| `.has_derivatives(True)` | bool | Only datasets with a derivatives/ directory |
| `.limit(n)` | int | Maximum results (default: 50) |
| `.offset(n)` | int | Pagination offset |

### Execution methods

```python
# Local catalog
results = query.fetch()             # list[dict]
page    = query.fetch_page()        # PageResult(results, total, offset)

# Live API (always current, no catalog refresh needed)
results = query.live()              # list[RichDatasetInfo]
results = query.live(sync_local=True)  # also updates local catalog

# Facet counts (distribution of modalities, tasks, licenses)
facets = DatasetQuery().modality("eeg").facets()
print(facets["tasks"])    # {"rest": 42, "nback": 18, ...}
```

## Refresh the local catalog

Pull the latest index from OpenNeuro once and search it offline:

```bash
qortex catalog-refresh
```

Writes to `~/.qortex/catalog.duckdb`. Incremental on subsequent runs.

## CLI search

```bash
qortex search --modality eeg
qortex search --modality eeg --task rest
qortex search --modality eeg --min-subjects 50
qortex search --modality fmri --task nback --min-subjects 30
qortex search --max-size 10           # GB
qortex search --query "resting state" # free-text on name/description
qortex search --has-events
qortex search --license CC0
```

Results:

```
ID         Name                          Subjects  Size    Modalities
ds004130   EEG resting-state alpha...    88        4.2 GB  eeg
ds003490   Resting state EEG...          64        2.1 GB  eeg
```

Add `--json` or `--csv` for machine-readable output.

## Rich search via the API

`search_datasets_rich()` searches the OpenNeuro API directly and returns fully populated `RichDatasetInfo` objects — with engagement metrics, demographics, funding, and snapshot summaries.

```python
from qortex.client import OpenNeuroClient

with OpenNeuroClient() as client:
    results = client.search_datasets_rich(
        modality="eeg",
        task="rest",
        min_subjects=30,
        min_downloads=100,
        sort_by="downloads",  # "downloads" | "views" | "stars" | "subjects" | "size" | "recent"
        limit=20,
    )

for info in results:
    snap = info.latest_snapshot_summary
    print(
        info.id,
        info.name,
        snap.n_subjects,
        info.engagement.downloads,
        info.engagement.popularity_score,
    )
```

### Sort options for `search_datasets_rich`

| `sort_by` | Orders by |
| --- | --- |
| `"downloads"` | Most downloaded first (default) |
| `"views"` | Most viewed first |
| `"stars"` | Most starred first |
| `"subjects"` | Most subjects first |
| `"size"` | Largest datasets first |
| `"recent"` | Most recently published first |

## Combining search with inspect

A typical workflow: search for candidates, then inspect the most promising ones:

```python
from qortex.catalog import DatasetQuery
from qortex.client import OpenNeuroClient

# Step 1 — find candidates (local, fast)
candidates = (
    DatasetQuery()
    .modality("eeg")
    .min_subjects(50)
    .has_events()
    .limit(5)
    .fetch()
)

# Step 2 — get full metadata for each
with OpenNeuroClient() as client:
    for row in candidates:
        info = client.get_dataset_rich(row["dataset_id"])
        readme = client.get_readme(row["dataset_id"])
        print(info.id, info.engagement.downloads, info.grant_funder)
        print(readme[:200] if readme else "(no README)")
```

## Catalog profile

```bash
qortex catalog-profile
```

Shows: total datasets indexed, last refresh timestamp, modality distribution.

## Related

- [Inspect](inspect.md) — full metadata and file-level detail for a specific dataset
- [Snapshots](snapshots.md) — list available versions of a dataset
- [Metadata](metadata.md) — read sidecar files without downloading imaging data
