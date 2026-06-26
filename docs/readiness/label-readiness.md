# Label Readiness

Label readiness tells you whether a dataset has enough labeled samples to train a classifier. It goes beyond the binary `can_train()` check by showing per-subject and per-class detail.

## Python

```python
from qortex import Dataset

ds = Dataset("ds004130")
landscape = ds.label_landscape()
print(landscape)
```

Output:

```
Label column: trial_type
Total subjects: 88   With labels: 88   Missing labels: 0

Class distribution:
  rest        2,112 samples  (88 subjects)
  eyes-open   2,064 samples  (86 subjects)  ← 2 subjects missing
  task           88 samples  (88 subjects)

Subjects missing 'eyes-open':
  sub-23  sub-47
```

## What label_landscape shows

- Which events.tsv column was selected (or the column you specified)
- Per-class sample counts summed across all subjects
- Which subjects are missing each class
- Whether label coverage is balanced

## Specify the label column

```python
landscape = ds.label_landscape(target_col="response_type")
```

If `target_col` is None, Qortex auto-selects the column with the most distinct non-null values (excluding `onset`, `duration`, `stim_file`, `HED`, `response_time`).

## CLI

```bash
qortex eda ds004130 --label trial_type
```

The `eda` command runs a more comprehensive exploratory analysis that includes label_landscape plus signal statistics.

## Working from local events.tsv files

If you have already downloaded the metadata:

```python
ds = Dataset("ds004130", data_dir="data/ds004130/")
landscape = ds.label_landscape()  # reads from local events.tsv files
```

Without a local directory, `label_landscape()` fetches events.tsv files from CDN. This is fast because events files are small.

## Interpreting class imbalance

A class with fewer samples per subject is not automatically a problem. If `task` appears only once per 10-minute run while `rest` appears every 20 seconds, the imbalance is by design. Review the events.tsv structure before deciding to exclude subjects or classes.

## Related

- [Can train](can-train.md) — binary gate based on label requirements
- [Doctor](doctor.md) — includes label landscape in the full report
- [EDA](../conversion/eda.md) — full exploratory data analysis including signal statistics
