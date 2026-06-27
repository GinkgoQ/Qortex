# Modality Readiness

Each modality has different companion files, quality indicators, and ML prerequisites. This page summarizes what Qortex checks per modality before recommending a dataset for training.

## Overview

| Modality | Required companions | Key readiness checks |
|----------|--------------------|--------------------|
| Structural MRI (T1w/T2w) | JSON sidecar | voxel size, orientation, shape |
| fMRI BOLD | JSON sidecar, events.tsv | TR present, events exist, label column usable |
| DWI | bval, bvec, JSON sidecar | bval/bvec present, ≥ 30 directions, multi-shell |
| MEG | channels.tsv, events.tsv | channel count, SSS status, events present |
| EEG | channels.tsv, events.tsv | channel count, sampling rate, events present |
| iEEG | channels.tsv, events.tsv, electrodes.tsv | electrode positions present, events present |
| fNIRS | channels.tsv, events.tsv | HbO/HbR channels detected, events present |
| PET | JSON sidecar (with tracer info) | TracerName, FrameTimesStart, body weight |

## Run readiness checks

The `doctor()` check applies modality-specific rules automatically:

```python
from qortex import Dataset

ds = Dataset("ds004130")
report = ds.doctor()
print(report.to_text())
```

Use a built-in recipe for stricter modality-specific thresholds:

```python
report = ds.doctor(recipe="eeg-classification")
report = ds.doctor(recipe="fmri-classification")
report = ds.doctor(recipe="dwi")
```

## Structural MRI

**No events required.** Labels come from `participants.tsv` (age, sex, group, diagnosis).

```python
ok = ds.can_train(
    target_col="group",          # from participants.tsv
    label_source="participants",
    min_subjects=30,
)
```

Readiness flags:
- `MISSING_T1W` — no T1w file for some subjects
- `SHAPE_MISMATCH` — T1w shapes differ across subjects (preprocessing may be needed)

## fMRI BOLD

```python
ok = ds.can_train(target_col="trial_type")
```

Readiness flags:
- `NO_EVENTS` — no events.tsv files
- `MISSING_TR` — TR not in JSON sidecar
- `NO_COMPANIONS` — events.tsv missing for some subjects that have BOLD

## DWI

```python
report = ds.doctor(recipe="dwi")
```

Readiness flags:
- `NO_BVAL` — bval file missing
- `NO_BVEC` — bvec file missing
- `FEW_DIRECTIONS` — fewer than 30 gradient directions (warning)
- `SINGLE_SHELL` — only one non-zero b-value (warning for multi-shell models)

## EEG / MEG

```python
ok = ds.can_train(target_col="trial_type")
report = ds.doctor(recipe="eeg-classification")
```

Readiness flags:
- `NO_CHANNELS_TSV` — channels.tsv missing
- `NO_EVENTS` — events.tsv missing or empty
- `LOW_SFREQ` — sampling frequency below threshold (warning)

## iEEG

Same checks as EEG, plus:

- `NO_ELECTRODES` — electrodes.tsv missing (spatial analysis not possible)
- `NO_COORDSYSTEM` — coordsystem.json missing

## fNIRS

Same event checks as EEG, plus:

- `NO_HBO_CHANNELS` — no HbO channels detected after loading
- `NO_OPTODES` — optodes.tsv missing

## PET

- `MISSING_TRACER` — TracerName or TracerRadionuclide absent in sidecar
- `MISSING_FRAME_TIMES` — FrameTimesStart not in sidecar (static PET assumed)
- `NO_SUV_FIELDS` — BodyWeight or InjectedRadioactivity absent (SUV not possible)

## Related

- [Doctor](../readiness/doctor.md) — full readiness report
- [Recipes](../readiness/recipes.md) — modality-specific recipe presets
- [Modality selection](selection.md) — choosing a modality
