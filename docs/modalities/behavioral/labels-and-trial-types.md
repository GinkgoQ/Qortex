# Labels and Trial Types

Qortex extracts labels from `trial_type` in `events.tsv` and numeric columns from both `events.tsv` and `participants.tsv`. Understanding how label detection works helps you choose the right target column and verify readiness before committing to a full download.

## Auto-detection

When `target_col=None`, Qortex auto-selects the best label column:

1. Read all `events.tsv` files across subjects (fetched from CDN — no large download needed)
2. Score each column by number of distinct non-null values, excluding known non-label columns: `onset`, `duration`, `stim_file`, `HED`, `response_time`, `trial_type_id`
3. Select the column with the most distinct values that is not purely numeric

```python
from qortex import Dataset

ds = Dataset("ds004130")
ok = ds.can_train()  # auto-detects label column
print(ds.label_landscape())
# Label column: trial_type (auto-detected)
```

Specify explicitly to avoid surprises:

```python
ok = ds.can_train(target_col="trial_type")
```

## Label landscape

`ds.label_landscape()` shows per-class counts and missing subjects:

```python
landscape = ds.label_landscape(target_col="trial_type")
print(landscape)
```

Output:

```
Label column: trial_type
Total subjects: 88   With labels: 86   Missing labels: 2

Class distribution:
  rest        2,112 samples   (88 subjects)
  task          528 samples   (86 subjects)  ← 2 subjects missing

Subjects missing 'task':
  sub-23  sub-47
```

## Regression labels

Numeric columns (response time, accuracy, behavioral score) can be used for regression:

```python
# From events.tsv
ok = ds.can_train(target_col="response_time", task_type="regression")

# From participants.tsv
parts = ds.participants()
print(parts.select(["participant_id", "age", "diagnosis"]))
```

For regression from `participants.tsv`, labels are subject-level, not trial-level. One label per subject propagates to all windows from that subject.

## Hierarchical trial types

BIDS permits slash notation: `condition/stimulus`. Qortex treats the full string as the label by default.

```python
landscape = ds.label_landscape(target_col="trial_type")
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
landscape = ds.label_landscape(target_col="trial_type")
# If one class has < 30 samples total: warning in doctor()
```

Qortex does not automatically balance classes during conversion. Handle imbalance in your training loop (class weights, oversampling, or undersampling).

## Can-train requirements

`can_train()` requires all of the following:

| Check | Default threshold |
|-------|------------------|
| `min_classes` | 2 |
| `min_per_class` | 10 samples total |
| `min_subjects` | 10 subjects with the label |
| `require_events` | True |

Adjust thresholds for your use case:

```python
ok = ds.can_train(
    target_col="trial_type",
    min_classes=3,
    min_per_class=50,
    min_subjects=30,
)
```

## Related

- [Events TSV](events-tsv.md)
- [Can train](../../readiness/can-train.md)
- [Label readiness](../../readiness/label-readiness.md)
- [EDA](../../conversion/eda.md)
