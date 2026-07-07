# Windows

A window is a fixed-length segment of a time series. The conversion pipeline extracts windows from each recording and writes them as rows in the output artifact.

## WindowSpec

```python
from qortex.convert import WindowSpec

window = WindowSpec(
    duration_s=30.0,      # window length in seconds
    overlap=0.5,           # fraction of overlap between adjacent windows (0–1)
    tmin=None,             # start offset from trial onset (for event-aligned)
    event_aligned=False,   # if True, use event onsets instead of sliding windows
)
```

Pass as a dict to `ds.convert()`:

```python
art = ds.convert(
    window=dict(duration_s=30.0, overlap=0.5),
    ...
)
```

## Sliding windows

The default mode. Starting from time 0, windows are extracted at regular intervals:

- Window 1: `[0, duration_s)`
- Window 2: `[duration_s * (1 - overlap), duration_s * (1 - overlap) + duration_s)`
- ...

Windows that extend past the end of the recording are dropped.

With `duration_s=30` and `overlap=0.5`, a 10-minute recording yields:
```
(600 - 30) / (30 * 0.5) + 1 = 39 windows
```

## Event-aligned windows

When `event_aligned=True`, windows are extracted relative to event onsets in the events.tsv:

```python
window = WindowSpec(
    duration_s=1.0,
    tmin=-0.2,            # 200 ms before onset
    event_aligned=True,
)
```

Each row in events.tsv with `trial_type` equal to the target label produces one window. The window runs from `onset + tmin` to `onset + tmin + duration_s`.

Labels come from the `trial_type` (or `label_col`) value for that event.

## Choosing a window strategy

Use **sliding windows** when:
- The recording is a single condition throughout (resting-state)
- You want maximum sample count
- You do not have event markers

Use **event-aligned windows** when:
- The experiment has discrete trials with onset times
- You want each sample to correspond to a single trial
- You need the label to be known at window time (not inferred post-hoc)

## Window size and feature dimensions

After windowing, each sample has shape `(n_channels, n_timepoints)`, where:

```
n_timepoints = duration_s × sampling_rate
```

For EEG at 256 Hz with 30 s windows: `n_timepoints = 7680`.

For fMRI with TR = 2.0 s and 30 s windows: `n_timepoints = 15` (in volume units, not seconds).

The feature dimension in the Parquet output is flattened: `n_channels × n_timepoints`.

## Handling recordings shorter than the window

If a recording is shorter than `duration_s`, the file is skipped with a warning. No partial windows are produced.

Set `min_duration_s` to skip files below a threshold:

```python
window = WindowSpec(duration_s=30.0, min_duration_s=60.0)
# Only process recordings of at least 60 seconds
```




## Related

- [Pipeline](pipeline.md) — full ConversionPipeline configuration
- [Splits](splits.md) — how subjects are assigned to train/val/test after windowing
