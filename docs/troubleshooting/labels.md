# Label Troubleshooting

## No events.tsv found

```
NO_EVENTS: No events.tsv files found in manifest
```

The dataset does not have events files. This is expected for datasets without task-based paradigms (resting-state EEG, structural MRI).

If you expected events files:
1. Check the manifest: `qortex inspect ds004130 --json | jq '.suffixes'`
2. Events files may be named differently (e.g., `*_stim.tsv` instead of `*_events.tsv`)
3. The BIDS spec requires suffix `events` — if the dataset uses a different name, Qortex will not recognize them

## events.tsv exists but label_col is empty

```
EMPTY_LABEL: trial_type column has no non-null values
```

Check the events file directly:

```python
events = ds.events(subject="01")
print(events.head())
print(events["trial_type"].value_counts())
```

The `trial_type` column may be all NaN for some datasets where conditions are encoded differently (e.g., in a numeric `stim_id` column).

Use `label_landscape()` with a different column:

```python
landscape = ds.label_landscape(target_col="stim_id")
```

## Events file has only one class

```
ONE_CLASS: trial_type has only 1 unique value: 'rest'
```

Resting-state datasets often have a single trial type. For classification, you need multiple classes. Options:

1. Use a participant-level label from `participants.tsv` instead (e.g., group, age_group, sex)
2. Use self-supervised learning that does not require event-based labels
3. Compute summary statistics per window and regress against a continuous behavioral measure

## Label column has mixed types

If `trial_type` contains both strings and NaN, pandas may infer a mixed dtype. Qortex handles this by converting NaN to an explicit "unknown" class. Check:

```python
events = ds.events(subject="01")
print(events["trial_type"].dtype)  # should be object (string)
events = events.dropna(subset=["trial_type"])
```

## Events file not aligned with recording

If onset times in events.tsv exceed the recording duration, some events will be skipped during event-aligned windowing. Qortex logs these as warnings:

```
WARNING: 3 events for sub-01 extend past recording end (600.0 s)
```

This can happen if the events file was created for a longer version of the recording, or if TR and onset times use different units.

## Related

- [Label readiness](../readiness/label-readiness.md) — per-subject label coverage
- [Can train](../readiness/can-train.md) — binary label check
- [EDA](../conversion/eda.md) — signal statistics and label analysis
