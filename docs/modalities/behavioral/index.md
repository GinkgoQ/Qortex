# Behavioral and Events

Behavioral data in BIDS includes events.tsv files, response logs, and behavioral task records. For Qortex, behavioral data is not just supplementary — it is often what makes a dataset trainable. Events.tsv defines the trial-type labels that supervised ML depends on.

## What counts as behavioral data

- `events.tsv` — trial onsets, durations, trial types, response times (paired with every signal recording)
- `beh/` datatype — standalone behavioral tables without an imaging companion
- `participants.tsv` — subject-level covariates: age, sex, diagnosis, group

## Why behavioral data matters for ML

A BOLD or EEG file without `events.tsv` cannot be used for supervised classification. Qortex's readiness checks verify event coverage before you download any large imaging files.

## Behavioral pages

[**Events TSV**](events-tsv.md) — Structure of events.tsv files, required and optional columns, BIDS rules.

[**Labels and trial types**](labels-and-trial-types.md) — How Qortex detects label columns, evaluates class distributions, and checks supervised-learning readiness.

## Check behavioral readiness

```python
from qortex import Dataset

ds = Dataset("ds000001", snapshot="1.0.0")

training = ds.can_train(target="trial_type")
print(training.to_text())

landscape = ds.label_landscape(label_column="trial_type", max_events_files=4)
print(landscape.summary())
```

CLI:

```bash
qortex can-train ds000001 --snapshot 1.0.0 --target trial_type
qortex eda ds004130 --label trial_type
```

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/ds000001-events-timeline.png" alt="Timeline of real ds000001 events across the first BOLD run.">
  <figcaption>The docs build generated this plot from a real Qortex `events()` call. The same table drives label readiness and event-aligned windows.</figcaption>
</figure>

## Access behavioral data

```python
# participants.tsv
participants = ds.participants()
print(participants.head())

# events.tsv for a subject
events = ds.events(subject="01", task="rest")
print(events.head())

# standalone behavioral file
beh_files = ds.files(datatypes=["beh"])
```
