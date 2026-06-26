# Cache

Qortex stores downloaded files in a local cache directory. Understanding how the cache works prevents duplicate downloads and helps manage disk space.

## Default cache location

```
~/.qortex/data/{dataset_id}/{snapshot}/
```

For example:

```
~/.qortex/data/ds004130/1.2.0/sub-01/eeg/sub-01_task-rest_eeg.set
```

Override the default with `data_dir`:

```python
ds.download(data_dir="/scratch/datasets/ds004130/")
```

When `data_dir` is set, files go directly into that directory (without the `~/.qortex/data/` prefix).

## Cache inspection from CLI

```bash
# Show total cache size and dataset list
qortex cache info

# List cached datasets with sizes
qortex cache list

# Show what is cached for a specific dataset
qortex cache list ds004130
```

Output example:

```
ds004130
  Snapshot: 1.2.0
  Path:     ~/.qortex/data/ds004130/1.2.0/
  Files:    1,056
  Size:     4.2 GB
  Last used: 2024-01-15
```

## Remove cache entries

```bash
# Remove all cached files for a dataset
qortex cache remove ds004130

# Remove a specific snapshot
qortex cache remove ds004130 --snapshot 1.0.0

# Clear the entire cache (all datasets)
qortex cache clear
```

From Python:

```python
from qortex.cache import clear_cache, remove_dataset_cache

clear_cache()
remove_dataset_cache("ds004130", snapshot="1.0.0")
```

## Avoiding duplicate downloads

If a file already exists at its target path and its size matches the manifest record, Qortex skips the download. Qortex does not compare checksums during the skip check — it only checks file size.

If you suspect a file is corrupted, delete it manually and re-download.

## Disk space planning

Before downloading, estimate required space:

```python
plan = ds.plan(subjects=["01", "02", "03"], tasks=["rest"])
print(f"Required: {plan.size_gb:.1f} GB")
```

## Manifests and sidecar caches

The manifest JSON and fetched sidecar files are cached at:

```
~/.qortex/manifests/{dataset_id}/{snapshot}.json
~/.qortex/sidecars/{dataset_id}/{snapshot}/{path}.json
```

These are small (< 10 MB per dataset) and are kept indefinitely. Delete them if you need to force a manifest re-fetch:

```bash
rm ~/.qortex/manifests/ds004130/1.2.0.json
```

Or force refresh in Python:

```python
manifest = ds.manifest(force_refresh=True)
```
