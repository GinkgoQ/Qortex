# Quickstart

This page walks through the minimal path from dataset ID to a Parquet artifact ready for training.

The example uses `ds004130`, an EEG resting-state dataset from OpenNeuro with 88 subjects.

## 1. Inspect the dataset

```python
from qortex import Dataset

ds = Dataset("ds004130")
report = ds.doctor()
print(report.to_text())
```

Expected output:

```
Dataset: ds004130 (snapshot: 1.2.0)
State:   manifest_only
Subjects: 88    Sessions: 1    Modalities: eeg
Events:   yes — 88 files found
Labels:   trial_type has 3 classes (rest, eyes-open, task)
Size:     ~4.2 GB
Next action: download
  Use ds.minimum(goal="first-batch") to get the smallest subset.
```

The manifest fetch takes a few seconds. No data is transferred.

## 2. Check label readiness

```python
ok = ds.can_train(target_col="trial_type", min_classes=2, min_per_class=10)
print(ok)  # True
```

If this returns False, call `ds.label_landscape()` to see which subjects are missing labels or have too few samples.

## 3. Plan the minimum download

```python
plan = ds.minimum(goal="first-batch")
print(f"{len(plan.files)} files, {plan.size_gb:.1f} GB")
# 12 files, 0.4 GB
```

`minimum()` picks the smallest real set of subjects whose data can complete one full pipeline pass. It includes the primary data files AND their sidecar companions automatically.

## 4. Download

```python
ds.download_paths(plan.files, data_dir="data/ds004130/")
```

Progress is logged per file. If interrupted, re-running resumes incomplete files.

## 5. Convert to Parquet

```python
art = ds.convert(
    data_dir="data/ds004130/",
    output_dir="artifacts/ds004130_parquet/",
    format="parquet",
    window=dict(duration_s=30.0, overlap=0.5),
    split=dict(strategy="subject", val_frac=0.15, test_frac=0.15),
    label_col="trial_type",
)
print(art.manifest.n_samples)
```

## 6. Load for training

```python
from qortex import Artifact

art = Artifact.open("artifacts/ds004130_parquet/")
X_train, y_train = art.sklearn(split="train")
X_val,   y_val   = art.sklearn(split="val")
```

Or with PyTorch:

```python
train_ds = art.torch(split="train")
val_ds   = art.torch(split="val")
```

## CLI equivalent

All of the above can be run from the command line:

```bash
qortex doctor ds004130
qortex can-train ds004130 --label trial_type
qortex download ds004130 --min-goal first-batch --data-dir data/ds004130/
qortex convert ds004130 \
    --data-dir data/ds004130/ \
    --output artifacts/ds004130_parquet/ \
    --format parquet \
    --window 30 \
    --overlap 0.5 \
    --label trial_type
```

## Next steps

- [Selective download](../download/selective-download.md) — filter by subject, session, task, run
- [Visual audit](first-visual-audit.md) — inspect file coverage before downloading
- [Conversion formats](../conversion/formats.md) — Zarr, HDF5, WebDataset, HuggingFace
