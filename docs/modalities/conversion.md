# Modality Conversion

Qortex converts each modality to ML-ready artifacts (Parquet, Zarr, HDF5, WebDataset, HuggingFace, TFRecord). The core conversion concepts — windows, splits, labels, provenance — are the same across modalities. What differs is the per-sample shape, windowing mode, and channel selection.

## Common conversion call

```python
from qortex import Dataset

ds = Dataset("ds004130", data_dir="data/ds004130/")
art = ds.convert(
    output_dir="artifacts/",
    format="parquet",
    window=dict(duration_s=2.0, overlap=0.5),
    split=dict(strategy="subject", val_frac=0.15, test_frac=0.15),
    label_col="trial_type",
)
```

## Per-modality output shape

| Modality | Per-sample shape | Notes |
|----------|-----------------|-------|
| Structural MRI | `(x, y, z)` | One sample per file |
| fMRI BOLD (sliding) | `(n_voxels, n_time_points)` | Flattened spatial × windowed time |
| fMRI BOLD (event-aligned) | `(n_voxels, n_time_points)` | Anchored to event onsets |
| DWI | `(x, y, z, n_directions)` | One sample per file |
| MEG | `(n_channels, n_time_points)` | Magnetometers + gradiometers |
| EEG | `(n_channels, n_time_points)` | EEG-type channels only |
| iEEG | `(n_channels, n_time_points)` | ieeg/seeg/ecog-type channels |
| fNIRS | `(n_channels, n_time_points)` | HbO + HbR interleaved |
| PET | `(x, y, z)` per frame | One sample per frame |

## Windowing modes

**Sliding window** — fixed-length overlapping windows across the full timeseries:

```python
window=dict(duration_s=30.0, overlap=0.5)
```

Best for: resting-state fMRI, MEG/EEG without discrete trials, fNIRS continuous recordings.

**Event-aligned** — windows centered on each trial onset in events.tsv:

```python
window=dict(mode="event_aligned", tmin=-0.2, tmax=0.8)
```

Best for: task-based EEG/MEG with clear trial structure, fMRI block designs.

## Label sources

| Source | Modalities | How to specify |
|--------|-----------|---------------|
| `events.tsv → trial_type` | fMRI, EEG, MEG, iEEG, fNIRS | `label_col="trial_type"` |
| `participants.tsv → column` | All (subject-level) | `label_col="age"`, `label_source="participants"` |
| No labels | Structural MRI, DWI, unsupervised | Omit `label_col` |

## MRI-specific conversion

```python
# T1w: one sample per subject, no windows
art = ds.convert(
    format="zarr",
    suffixes=["T1w"],
    split=dict(strategy="subject"),
)
# Sample shape: (x, y, z) float32

# BOLD: sliding windows, trial_type labels
art = ds.convert(
    format="parquet",
    suffixes=["bold"],
    window=dict(duration_s=20.0, overlap=0.5),
    label_col="trial_type",
)
```

## Signal-specific conversion (EEG / MEG / iEEG / fNIRS)

```python
# Event-aligned
art = ds.convert(
    format="parquet",
    datatypes=["eeg"],
    window=dict(mode="event_aligned", tmin=-0.2, tmax=0.8),
    label_col="trial_type",
)
# Sample shape: (n_eeg_channels, n_time_points)
```

## DWI conversion

```python
art = ds.convert(
    format="zarr",
    datatypes=["dwi"],
    split=dict(strategy="subject"),
)
# Sample shape: (x, y, z, n_directions)
```

bval and bvec arrays are stored as sample metadata in the artifact manifest.

## PET conversion

```python
art = ds.convert(
    format="zarr",
    datatypes=["pet"],
    split=dict(strategy="subject"),
)
# One sample per PET frame if dynamic, one sample total if static
```

## Load the artifact

All modalities produce the same artifact interface regardless of shape:

```python
from qortex import Artifact

art = Artifact.open("artifacts/")
X, y = art.sklearn(split="train")
ds_torch = art.torch(split="train")
hf_ds = art.huggingface(split="train")
```

## Related

- [Conversion pipeline](../conversion/pipeline.md)
- [Windows](../conversion/windows.md)
- [Formats](../conversion/formats.md)
- [ML bridge](../artifacts/ml-bridge.md)
