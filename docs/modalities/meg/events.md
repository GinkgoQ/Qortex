# MEG Events

MEG events define the timing of stimuli and responses within a recording. In BIDS, events are stored in `events.tsv`. Some older MEG datasets store events only in the stimulus trigger channel inside the raw file.

## BIDS events.tsv

```
onset   duration  trial_type     stim_file
0.300   0.0       face/famous    stimuli/f001.bmp
0.700   0.0       face/scrambled stimuli/s001.bmp
1.100   0.0       face/unfamiliar stimuli/u001.bmp
```

- `onset` — time relative to recording start (seconds)
- `duration` — event duration (0 for instantaneous stimuli)
- `trial_type` — label for ML; may use slash notation for hierarchical conditions

## Check event coverage

```python
from qortex import Dataset

ds = Dataset("ds000117")
report = ds.doctor()
# Events: yes — 102 files, columns: onset duration trial_type stim_file

# Per-subject detail
print(ds.label_landscape(target_col="trial_type"))
```

## Read events for a specific file

```python
events = ds.events(subject="01", task="facerecognition")
# Polars or pandas DataFrame: onset, duration, trial_type, ...
print(events["trial_type"].value_counts())
```

## Stim-channel events (no events.tsv)

Some legacy datasets store events only in the trigger channel. MNE-Python can extract them:

```python
import mne
raw = mne.io.read_raw_fif("sub-01/meg/sub-01_task-rest_meg.fif", preload=False)
events = mne.find_events(raw, stim_channel="STI014")
# events: array of [sample, prev_value, event_id]
```

Qortex's readiness checks flag datasets that have a stim channel but no `events.tsv` — these require manual event extraction before supervised conversion.

## Event-aligned windows for ML

```python
art = ds.convert(
    data_dir="data/ds000117/",
    output_dir="artifacts/",
    format="parquet",
    window=dict(mode="event_aligned", tmin=-0.2, tmax=0.8),
    label_col="trial_type",
    datatypes=["meg"],
)
```

Each sample is a `(n_channels, n_time_points)` array. The time axis spans `tmin` to `tmax` relative to each event onset.

## Readiness considerations

- `trial_type` with slash notation (`face/famous`) — Qortex treats the full string as the label class. Flatten if needed: `"famous"` instead of `"face/famous"`.
- Response columns (`response`, `response_time`) can serve as regression targets.
- Recordings with stim-channel-only events need manual preprocessing outside Qortex before supervised conversion.

## Related

- [MEG files](files.md)
- [Label readiness](../../readiness/label-readiness.md)
- [Event-aligned windows](../../conversion/windows.md)
