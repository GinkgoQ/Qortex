# fMRI / BOLD

fMRI BOLD (Blood Oxygen Level Dependent) is the dominant functional imaging modality on OpenNeuro. Each BOLD file is a 4D NIfTI: spatial dimensions plus time. The TR (repetition time) determines the temporal resolution. Events.tsv files define task labels aligned to acquisition time.

## BIDS files

| Suffix | Description |
|--------|-------------|
| `bold` | Primary BOLD timeseries |
| `sbref` | Single-band reference volume (EPI geometry) |
| `events` | `events.tsv` — onset, duration, trial_type |
| `physio` | Physiological recordings (respiration, cardiac) |

Companions:
- `*_bold.json` — TR, task name, slice timing, field strength
- `*_events.tsv` — required for supervised learning
- `*_desc-confounds_timeseries.tsv` — motion and nuisance regressors (from fMRIPrep)

## Inspect before download

```python
from qortex import Dataset

ds = Dataset("ds004130")

# Check BOLD file count and event coverage
report = ds.doctor()
print(report.to_text())
# Events: yes — 88 files, columns: onset duration trial_type
# Size: 4.2 GB

# Full event coverage check
print(ds.label_landscape(target_col="trial_type"))
```

## Inspect after download

```python
info = ds.nifti_info("sub-01/func/sub-01_task-rest_bold.nii.gz")

info["shape"]                    # [64, 64, 32, 240]
info["n_volumes"]                # 240
info["tr_s"]                     # 2.0  (seconds)
info["voxel_size_mm"]            # [3.0, 3.0, 3.5]
info["total_acquisition_time_s"] # 480.0
info["task_name"]                # "rest"
info["slice_timing_available"]   # True
info["magnetic_field_strength"]  # 3.0
```

## Visualize (QC)

```python
from qortex.visualize import fmri_summary

fig = fmri_summary(
    "data/ds004130/sub-01/func/sub-01_task-rest_bold.nii.gz",
    events="data/ds004130/sub-01/func/sub-01_task-rest_events.tsv",
)
fig.show()
```

The 6-panel QC figure includes: mean EPI, standard deviation map, tSNR, global signal timeseries, framewise displacement, and a trial-type overlay on the global signal.

CLI:

```bash
qortex fmri-qc sub-01/func/sub-01_task-rest_bold.nii.gz \
    --events sub-01/func/sub-01_task-rest_events.tsv \
    --output sub01_bold_qc.html
```

## Events and labels

Events.tsv for BOLD defines the label for each trial window:

```
onset   duration  trial_type
0.0     20.0      rest
20.0    20.0      task
40.0    20.0      rest
```

Qortex reads events automatically when converting event-aligned windows:

```python
art = ds.convert(
    data_dir="data/ds004130/",
    output_dir="artifacts/",
    format="parquet",
    window=dict(mode="event_aligned", tmin=-2.0, tmax=18.0),
    label_col="trial_type",
)
```

## Conversion — sliding vs event-aligned windows

| Mode | Description |
|------|-------------|
| `sliding` | Fixed-length windows stepped across the full timeseries |
| `event_aligned` | Windows anchored to event onsets in events.tsv |

```python
# Sliding window — 30 s, 50% overlap
art = ds.convert(
    format="parquet",
    window=dict(duration_s=30.0, overlap=0.5),
    label_col="trial_type",
)

# Event-aligned — 2 s before to 18 s after onset
art = ds.convert(
    format="parquet",
    window=dict(mode="event_aligned", tmin=-2.0, tmax=18.0),
    label_col="trial_type",
)
```

Each sample in the artifact is a `(n_voxels, n_time_points)` array or a flattened feature vector depending on the output format.

## Readiness considerations

- **Events.tsv is required** for supervised classification. Use `ds.can_train()` to verify.
- **TR must be in JSON sidecar.** If the sidecar is missing, TR falls back to the NIfTI pixdim[4] which is unreliable.
- **Confound files.** fMRIPrep confound TSVs are in `derivatives/`. They are excluded by default. Set `include_derivatives=True` to include them.
- **Motion.** High motion subjects are not automatically flagged. Use framewise displacement from the confound file.

## Related

- [Readiness recipes](../../readiness/recipes.md) — `fmri-classification` recipe
- [fMRI QC](../../visualization/fmri-qc.md) — full QC reference
- [Windows](../../conversion/windows.md) — sliding vs event-aligned window details
