# Metadata

Qortex can read sidecar files, events.tsv, participants.tsv, and dataset_description.json without downloading the full imaging data. These files are small and fetched directly from the OpenNeuro CDN.

## Dataset-level metadata

```python
ds = Dataset("ds004130")

# Root-level files
desc = ds.metadata()         # dataset_description.json as dict
parts = ds.participants()    # participants.tsv as pandas DataFrame
```

The participants.tsv DataFrame has columns like `participant_id`, `age`, `sex`, and any other columns that dataset depositors included. Column names vary per dataset.

## File-level sidecars

```python
# JSON sidecar for a specific file
sidecar = ds.sidecar("sub-01/eeg/sub-01_task-rest_eeg.json")

# Events for a specific run
events = ds.events(subject="01", task="rest")
# returns a pandas DataFrame: onset, duration, trial_type, ...
```

`sidecar()` accepts either a path string or a FileRecord object. It fetches the JSON from CDN and returns a dict.

`events()` fetches the events.tsv for the matching file. If the subject has multiple sessions or runs, pass `session=` and `run=` to disambiguate.

## CLI

```bash
# Print dataset_description.json
qortex metadata ds004130

# Print participants.tsv
qortex metadata ds004130 --participants

# Print a specific sidecar
qortex metadata ds004130 sub-01/eeg/sub-01_task-rest_eeg.json
```

## Downloading metadata only

To download only the small metadata files (no large imaging data):

```python
ds.download_metadata(data_dir="data/ds004130/")
```

This downloads:
- `dataset_description.json`
- `participants.tsv`
- `README`, `CHANGES`
- All `*.json` sidecar files
- All `*_events.tsv` files
- All `*_channels.tsv` files
- All `*_coordsystem.json` files
- All `*.bval`, `*.bvec` files

Imaging files (`.nii.gz`, `.set`, `.fif`, `.edf`, etc.) are excluded.

CLI equivalent:

```bash
qortex download ds004130 --metadata-only --data-dir data/ds004130/
```

## What metadata tells you before downloading

A metadata-only pass reveals:

- Whether events.tsv files exist and are non-empty
- What trial types appear in events.tsv (potential label columns)
- Whether subjects are age/sex balanced
- EEG channel count and type from channels.tsv
- fMRI TR, task name, slice timing from JSON sidecars
- DWI b-values from bval files

This information is often sufficient to decide whether a dataset is worth the full download.

## Related

- [`ds.inspect()`](inspect.md) — manifest-level structural summary
- [`ds.doctor()`](../readiness/doctor.md) — includes a metadata-validity check
- [`ds.label_landscape()`](../readiness/label-readiness.md) — aggregate view of label coverage
