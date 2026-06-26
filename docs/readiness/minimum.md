# Minimum

`ds.minimum()` computes the smallest real download that enables a specific goal. It returns a `DownloadPlan` with exactly the files needed — no padding, no extras beyond required companions.

## Python

```python
from qortex import Dataset

ds = Dataset("ds004130")

plan = ds.minimum(goal="first-batch")
print(f"{len(plan.files)} files, {plan.size_gb:.2f} GB")
# 12 files, 0.4 GB
```

## Goals

| Goal | What it means |
|------|--------------|
| `first-batch` | Enough subjects to run one complete pipeline pass (inspect → download → convert → train step). Usually 3–5 subjects. |
| `label-check` | Only the events.tsv and sidecar JSON files. No imaging data. |
| `validation` | A representative subset for BIDS validation — one subject per modality/task combination. |
| `metadata` | All JSON sidecars, TSV, bval/bvec, root-level files. No imaging data. |

## CLI

```bash
qortex minimum ds004130 --goal first-batch
qortex minimum ds004130 --goal label-check
qortex minimum ds004130 --goal metadata
```

Output:

```
Goal: first-batch
Subjects: sub-01, sub-02, sub-03
Files: 12
Size: 0.4 GB
```

Add `--download` to execute the plan immediately:

```bash
qortex minimum ds004130 --goal first-batch --download --data-dir data/ds004130/
```

## How first-batch is computed

`first-batch` selects subjects that maximize label class coverage within the minimum count:

1. Fetch the manifest and events.tsv for all subjects
2. For each subject, count label occurrences per class
3. Select the minimum number of subjects such that every class appears at least once
4. Add one extra subject as a buffer for val/test split
5. Include all companions (JSON, events, bval/bvec) for the selected subjects

If the dataset has no label column (no events.tsv), `first-batch` falls back to selecting 3 subjects arbitrarily.

## Using the plan

```python
plan = ds.minimum(goal="first-batch")

# Inspect before committing
for f in plan.files:
    print(f.path, f.size)

# Download
ds.download_paths(plan.files, data_dir="data/ds004130/")
```

## After first-batch succeeds

A successful first-batch run confirms:

- The download engine works for this dataset
- The conversion pipeline can extract windows and labels
- The ML bridge (PyTorch/sklearn) can load the artifact

If first-batch fails, the error is almost always a data issue (missing events, malformed sidecar, LFS pointers) rather than a library bug.

## Related

- [Plan](../download/plan.md) — for custom file selections
- [First batch](first-batch.md) — detailed description of the first-batch diagnostic
