# EEG Events

EEG events define the onset and label of each trial in a recording. Qortex reads `events.tsv` files and uses them to construct supervised training samples via event-aligned windowing.

## BIDS events.tsv

```
onset    duration  trial_type  response_time
0.000    1.5       rest        n/a
1.500    1.5       task        0.612
3.000    1.5       rest        n/a
4.500    1.5       task        0.498
```

Required columns: `onset`, `duration`.
Recommended column: `trial_type`.

## Check event availability

```python
from qortex import Dataset

ds = Dataset("ds000001", snapshot="1.0.0")
report = ds.doctor()

training = ds.can_train(target="trial_type")
print(training.to_text())

print(ds.label_landscape(label_column="trial_type", max_events_files=4).summary())
```

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/ds000001-events-timeline.png" alt="Real event timeline for ds000001 subject 01 run 01.">
  <figcaption>Although this figure comes from an fMRI task, the same BIDS `events.tsv` contract is used for EEG and MEG event-aligned windows: onset, duration, label, and optional response columns.</figcaption>
</figure>

## Read events for a specific subject

```python
events = ds.events(subject="01", task="balloonanalogrisktask", run="01")
print(events["trial_type"].value_counts())
print(events.shape)                    # (158, 8)
```

For multiple sessions or runs:

```python
events = ds.events(subject="01", task="rest", session="01", run="01")
```

## Event-aligned window conversion

```python
art = ds.convert(
    data_dir="data/ds004130/",
    output_dir="artifacts/",
    format="parquet",
    window=dict(mode="event_aligned", tmin=-0.2, tmax=0.8),
    label_col="trial_type",
)
# Each sample: (n_channels, n_time_points) — 200 ms pre- to 800 ms post-onset
```

## Stimulus channel triggers (no events.tsv)

Some datasets have trigger codes only in a `STI` or `Status` channel. Qortex's readiness check warns when `events.tsv` is absent but a stimulus channel is detected:

```python
report = ds.doctor()
# WARNING [NO_EVENTS_TSV]: sub-03 has STI channel but no events.tsv
# Suggestion: extract events manually with mne.find_events() and save as events.tsv
```

## Hierarchical trial types

BIDS allows slash-delimited trial types: `condition/stimulus`. Qortex treats the full string as the label class.

```python
# events: ["go/compatible", "go/incompatible", "nogo"]
# → 3 classes in the classifier
```

Collapse to top-level:

```python
import polars as pl
events = ds.events(subject="01", task="flanker")
events = events.with_columns(
    pl.col("trial_type").str.split("/").list.get(0).alias("trial_type")
)
# → ["go", "go", "nogo"]
```








<!-- qortex-evidence:start -->

## Evidence

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/ds000001-events-timeline.png" alt="Timeline of ds000001 events and trial-type counts for subject 01 run 01.">
  <figcaption>Real `events.tsv` timeline for ds000001 sub-01 run-01.</figcaption>
</figure>

```python
events = ds.events(subject='01', task='balloonanalogrisktask', run='01')
print(events.shape)
```

Result artifact: [ds000001-example-results.json](/Qortex/assets/results/ds000001-example-results.json)

<!-- qortex-evidence:end -->

## Related

- [EEG files](files.md)
- [Label readiness](../../readiness/label-readiness.md)
- [Windows — event-aligned](../../conversion/windows.md)
