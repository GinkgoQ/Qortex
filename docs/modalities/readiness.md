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

Use `minimum()` and `can_train()` for goal-specific decisions:

```python
fmri_training = ds.can_train(target="trial_type", modality="fmri")
first_batch = ds.minimum(goal="first-batch", modality="fmri", target="trial_type")
print(fmri_training.to_text())
print(first_batch.to_text())
```

## Structural MRI

**No events required.** Labels come from `participants.tsv` (age, sex, group, diagnosis).

```python
report = ds.can_train(target="group", modality="mri")
print(report.to_text())
```

Readiness flags:
- `MISSING_T1W` — no T1w file for some subjects
- `SHAPE_MISMATCH` — T1w shapes differ across subjects (preprocessing may be needed)

## fMRI BOLD

```python
report = ds.can_train(target="trial_type", modality="fmri")
print(report.to_text())
```

Readiness flags:
- `NO_EVENTS` — no events.tsv files
- `MISSING_TR` — TR not in JSON sidecar
- `NO_COMPANIONS` — events.tsv missing for some subjects that have BOLD

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/ds000001-events-timeline.png" alt="Event timeline for ds000001 subject 01 run 01 with cash, control_pumps, explode, and pumps events over a 600 second BOLD run.">
  <figcaption>Real events from `ds000001/sub-01_task-balloonanalogrisktask_run-01_events.tsv`. fMRI readiness depends on this timing table as much as on the BOLD image.</figcaption>
</figure>

## DWI

```python
report = ds.doctor()
print(report.to_text())
```

Readiness flags:
- `NO_BVAL` — bval file missing
- `NO_BVEC` — bvec file missing
- `FEW_DIRECTIONS` — fewer than 30 gradient directions (warning)
- `SINGLE_SHELL` — only one non-zero b-value (warning for multi-shell models)

## EEG / MEG

```python
training = ds.can_train(target="trial_type", modality="eeg")
print(training.to_text())
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
