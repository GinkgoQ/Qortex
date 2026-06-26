# Can Train

`ds.can_train()` returns True or False: does this dataset have enough labeled samples to train a classifier with the given requirements?

It is a binary gate. Use it before committing to a full download.

## Python

```python
from qortex import Dataset

ds = Dataset("ds004130")

ok = ds.can_train(
    target_col="trial_type",
    min_classes=2,
    min_per_class=10,
    min_subjects=20,
)
print(ok)  # True or False
```

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `target_col` | `None` | Events.tsv column to use as labels. If None, auto-detects. |
| `min_classes` | `2` | Minimum number of distinct label classes required. |
| `min_per_class` | `10` | Minimum total samples per class across all subjects. |
| `min_subjects` | `10` | Minimum number of subjects with the target label. |
| `require_events` | `True` | Fail if no events.tsv files exist. |

## CLI

```bash
qortex can-train ds004130 --label trial_type --min-classes 2 --min-per-class 10
```

Output:

```
can_train: True
  Label column: trial_type
  Classes: rest (240), eyes-open (236), task (88)
  Subjects with labels: 88/88
```

If False:

```
can_train: False
  Reason: Only 1 class found in trial_type. Need at least 2.
  Suggestion: Check label_landscape() for per-subject class distribution.
```

## Auto-detecting the label column

If `target_col=None`, Qortex scans all events.tsv columns and selects the one with the most non-null distinct values. It skips columns named `onset`, `duration`, `stim_file`, `HED`, and `response_time`.

Auto-detection is useful when you are exploring a new dataset and do not know the events structure yet. However, for production use, specify the column explicitly to avoid surprises when the data changes.

## What can_train does NOT check

- Whether the subjects are balanced across classes
- Whether enough subjects exist for a meaningful train/val/test split
- Whether the label column has the right encoding for your task
- Whether the imaging data is free of motion artifacts

Use `ds.doctor()` for the full report that covers split feasibility and data quality warnings.

## Related

- [Doctor](doctor.md) — comprehensive report including label checks
- [Label readiness](label-readiness.md) — per-subject label coverage detail
- [First batch](first-batch.md) — end-to-end pipeline test
