# Inspect

`ds.inspect()` fetches the manifest and returns a structured summary of dataset properties. It is the first method to call when evaluating a new dataset.

## Python

```python
from qortex import Dataset

ds = Dataset("ds004130")
info = ds.inspect()
print(info)
```

The returned object has:

```python
info.dataset_id        # "ds004130"
info.snapshot          # "1.2.0"
info.name              # full dataset name from dataset_description.json
info.doi               # dataset DOI
info.n_subjects        # integer
info.n_sessions        # integer (1 if no sessions)
info.modalities        # ["eeg"]
info.tasks             # ["rest"]
info.suffixes          # ["eeg", "events", "channels", "coordsystem"]
info.has_events        # True/False
info.event_columns     # ["onset", "duration", "trial_type", ...]
info.has_bval          # True/False (DWI)
info.n_files           # integer
info.total_size        # bytes
```

## CLI

```bash
qortex inspect ds004130
```

Output:

```
Dataset:   ds004130 — EEG resting-state
Snapshot:  1.2.0
DOI:       10.18112/openneuro.ds004130.v1.2.0
Subjects:  88
Sessions:  1
Modalities: eeg
Tasks:     rest
Files:     1,056
Size:      4.2 GB
Events:    yes (88 files, columns: onset duration trial_type)
```

Add `--json` for machine-readable output:

```bash
qortex inspect ds004130 --json
```

## How it works

`inspect()` calls `ds.manifest()` internally. The manifest is fetched from the OpenNeuro GraphQL API. The response includes the file tree for the latest (or specified) snapshot. Qortex parses BIDS entity labels from each file path and aggregates the counts.

The summary (`ds.info()`) is a lighter version of inspect — it returns only the ManifestSummary without the full file list.

## Inspecting a local dataset

If you have a local BIDS directory and want to build a manifest from disk rather than OpenNeuro:

```python
ds = Dataset("ds004130", data_dir="data/ds004130/")
ds.index_local()     # scan local tree, write .qortex/manifest.json
info = ds.inspect()  # reads from local manifest
```

Or from the CLI:

```bash
qortex local-index data/ds004130/ --dataset-id ds004130
qortex inspect ds004130 --local
```

## Related

- [`ds.metadata()`](metadata.md) — read sidecar files without downloading imaging data
- [`ds.doctor()`](../readiness/doctor.md) — full readiness report beyond just structure
- [`ds.files()`](metadata.md) — get a filtered list of FileRecords
