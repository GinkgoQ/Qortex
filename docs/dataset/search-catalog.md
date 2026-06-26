# Search Catalog

Qortex maintains a local DuckDB catalog of dataset summaries indexed from OpenNeuro. The catalog lets you filter datasets by modality, task, subject count, and size without fetching individual manifests.

## Refresh the catalog

Before searching, pull the latest index from OpenNeuro:

```bash
qortex catalog-refresh
```

This fetches summary records for all public datasets and writes them to the local DuckDB database at `~/.qortex/catalog.duckdb`. A full refresh takes a few minutes on first run. Subsequent refreshes are incremental.

## Search from the CLI

```bash
qortex search --modality eeg
qortex search --modality eeg --task rest
qortex search --modality eeg --min-subjects 50
qortex search --modality fmri --task nback --min-subjects 30
qortex search --max-size 10  # GB
qortex search --query "resting state"  # full-text search on dataset name/description
```

Results are printed as a table:

```
ID         Name                          Subjects  Size    Modalities
ds004130   EEG resting-state alpha...    88        4.2 GB  eeg
ds003490   Resting state EEG...          64        2.1 GB  eeg
...
```

Add `--json` or `--csv` for machine-readable output.

## Search from Python

```python
from qortex.catalog import search_catalog

results = search_catalog(modality="eeg", min_subjects=50)
for row in results:
    print(row.dataset_id, row.n_subjects, row.size_gb)
```

The `search_catalog()` function returns a list of `CatalogRow` objects.

## Catalog profile

To see catalog statistics (total datasets, last refresh, modality distribution):

```bash
qortex catalog-profile
```

## Limitations

- The catalog contains only public datasets. Private or embargoed datasets do not appear.
- The catalog stores summary data — subject counts, modalities, total size. It does not store file-level information. For file-level queries, fetch the manifest with `ds.inspect()`.
- The catalog is local to your machine. It is not synchronized automatically. Run `catalog-refresh` to update.

## Related

- [Inspect](inspect.md) — fetch full manifest for a specific dataset
- [Snapshots](snapshots.md) — list available versions of a dataset
