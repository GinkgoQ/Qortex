# Labels and Trial Types

Qortex extracts labels from `trial_type` in `events.tsv` and numeric columns from both `events.tsv` and `participants.tsv`. Understanding how label detection works helps you choose the right target column and verify readiness before committing to a full download.

## Auto-detection

When `label_column=None`, `label_landscape()` auto-selects the best label column:

1. Read all `events.tsv` files across subjects (fetched from CDN — no large download needed)
2. Score each column by number of distinct non-null values, excluding known non-label columns: `onset`, `duration`, `stim_file`, `HED`, `response_time`, `trial_type_id`
3. Select the column with the most distinct values that is not purely numeric

```python
from qortex import Dataset

ds = Dataset("ds000001", snapshot="1.0.0")
landscape = ds.label_landscape(max_events_files=4)
print(landscape.summary())
```

Specify explicitly to avoid surprises:

```python
training = ds.can_train(target="trial_type")
print(training.to_text())
```

## Label landscape

`ds.label_landscape()` shows per-class counts and missing subjects:

```python
landscape = ds.label_landscape(label_column="trial_type", max_events_files=4)
print(landscape.summary())
```

Output:

```
Label Landscape — ds000001
Events files: 4/4 fetched
Label column: trial_type
Classes: 4  Total events: 568
Coverage: 100.0% of signal keys have events
Imbalance: 7.87x (severe)
Cross-subject consistency: 100.0%

Class distribution:
  pumps_demean                       299  ███████████████
  control_pumps_demean               187  █████████
  cash_demean                         44  ██
  explode_demean                      38  ██
```

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/ds000001-can-train.png" alt="Training readiness chart for ds000001.">
  <figcaption>Real `can_train(target="trial_type")` output for `ds000001`. The report separates label evidence from local file availability so you know what must be downloaded next.</figcaption>
</figure>

## Regression labels

Numeric columns (response time, accuracy, behavioral score) can be used for regression:

```python
# From participants.tsv
parts = ds.participants()
print(parts.select(["participant_id", "age", "diagnosis"]))
```

For regression-style work, inspect the candidate numeric column directly before conversion. `can_train()` currently reports supervised classification readiness; subject-level numeric outcomes need a conversion recipe that preserves the target and a model objective chosen outside Qortex.

## Hierarchical trial types

BIDS permits slash notation: `condition/stimulus`. Qortex treats the full string as the label by default.

```python
landscape = ds.label_landscape(label_column="trial_type")
# Classes: ["go/compatible", "go/incompatible", "nogo"] → 3 classes

# Collapse to top level:
import polars as pl
events = ds.events(subject="01", task="flanker")
events = events.with_columns(
    pl.col("trial_type").str.split("/").list.get(0)
)
# Classes: ["go", "go", "nogo"] → 2 classes
```

## Class imbalance

Severe imbalance makes train/val/test split difficult without stratification. Check per-class counts:

```python
landscape = ds.label_landscape(label_column="trial_type")
print(landscape.summary())
```

Qortex does not automatically balance classes during conversion. Handle imbalance in your training loop (class weights, oversampling, or undersampling).

## Can-train requirements

`can_train()` checks the evidence Qortex can prove from the manifest and available local files:

- events exist and can be paired with signal files
- the requested target appears in the behavioral tables
- enough recordings can plausibly support a first supervised run
- a subject-level split is feasible enough to avoid leakage

Use `label_landscape()` for class-balance detail, then use `minimum(goal="first-batch")` when the report asks for local confirmation.








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

- [Events TSV](events-tsv.md)
- [Can train](../../readiness/can-train.md)
- [Label readiness](../../readiness/label-readiness.md)
- [EDA](../../conversion/eda.md)
