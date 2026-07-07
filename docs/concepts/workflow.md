# Workflow

A full Qortex workflow moves from catalog search to a trained model in six stages. Each stage has a decision gate before proceeding to the next.

## Stage 1: Find candidate datasets

Search the local catalog (backed by DuckDB) or the live OpenNeuro API for datasets that match your modality and task requirements.

```bash
qortex search --modality eeg --task rest --min-subjects 50
```

Via Python — local catalog (fast, offline):

```python
from qortex.catalog import DatasetQuery

results = (
    DatasetQuery()
    .modality("eeg")
    .task("rest")
    .min_subjects(50)
    .has_events()
    .fetch()
)
```

Via Python — live API with engagement sorting:

```python
from qortex.client import OpenNeuroClient

with OpenNeuroClient() as client:
    results = client.search_datasets_rich(
        modality="eeg", task="rest", min_subjects=50, sort_by="downloads"
    )
    for info in results:
        print(info.id, info.name, info.engagement.downloads)
```

If you already know the dataset ID, skip the search:

```python
from qortex import Dataset
ds = Dataset("ds004130")
```

At this point, no network call has been made. The `Dataset` object holds only the ID.

## Stage 2: Inspect the manifest

Fetch the remote file tree and check structural properties.

```python
report = ds.doctor()
print(report.to_text())
```

`doctor()` reads the manifest and a sample of sidecar files. It reports:

- Subject count and session count
- Modalities present
- Whether events.tsv files exist and are non-empty
- Whether bval/bvec companions exist for DWI
- Total uncompressed size estimate

**Decision gate:** If `report.state` is `not_usable`, stop here. Do not download.

## Stage 3: Verify label readiness

Before downloading bulk imaging data, confirm that labels are present and have enough coverage.

```python
training = ds.can_train(target="trial_type")
print(training.to_text())

landscape = ds.label_landscape(label_column="trial_type", max_events_files=4)
print(landscape.summary())
```

`can_train()` returns a decision report, not a loose boolean. The report names the state, the evidence Qortex found, the missing evidence, a split recommendation, and the next command when the dataset needs a smaller metadata or first-batch download.

On OpenNeuro `ds000001`, the label scan over four real `events.tsv` files reports four classes and 568 events. It also flags a severe class imbalance: `pumps_demean` has 299 events, while `explode_demean` has 38. That does not make the dataset unusable, but it changes the training plan: use class weighting or resampling, and report per-class metrics instead of accuracy alone.

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/ds000001-can-train.png" alt="Training readiness chart for ds000001 showing uncertain status, 16 subjects, 80 recordings, zero locally confirmed label-ready recordings, and 0.05 GB required for the first batch plan.">
  <figcaption>Real `Dataset("ds000001").can_train(target="trial_type")` output visualized from the generated docs run. Qortex is cautious until local label evidence is confirmed.</figcaption>
</figure>

**Decision gate:** proceed only when the report is `possible`, or when the next action is small and explicit enough to verify before any large download.

## Stage 4: Download the minimum

`minimum()` returns the smallest download that enables the next step. `first_batch()` returns the subset needed to test the pipeline end-to-end.

```python
plan = ds.minimum(goal="first-batch")
ds.download_paths(plan.files)
```

The plan includes all companion files (events.tsv, JSON sidecars, bval/bvec) automatically.

For a full selective download:

```python
ds.download(subjects=["01", "02", "03"], tasks=["rest"], suffixes=["bold"])
```

**Decision gate:** After download, run `ds.validate()` to confirm the local BIDS structure is complete.

## Stage 5: Convert to ML format

Once data is on disk, convert to a format suitable for training.

```python
art = ds.convert(
    output_dir="converted/",
    format="parquet",
    window=dict(duration_s=30.0, overlap=0.5),
    split=dict(strategy="subject", val_frac=0.15, test_frac=0.15),
    label_col="trial_type",
)
```

The `ConversionPipeline` writes one artifact directory with `train/`, `val/`, and `test/` subdirectories and an `artifact_manifest.json`.

**Decision gate:** Run `ds.leakage_check(art)` to confirm no subject appears in two splits.

## Stage 6: Load for training

```python
from qortex import Artifact

art = Artifact.open("converted/")

# PyTorch
train_ds = art.torch(split="train")
val_ds   = art.torch(split="val")

# scikit-learn
X_train, y_train = art.sklearn(split="train")
X_val,   y_val   = art.sklearn(split="val")

# HuggingFace Dataset
hf_ds = art.huggingface(split="train")
```

## Skipping stages

Not all projects need all stages. A team that already has a downloaded BIDS dataset on disk can start at Stage 5 by calling `ds.index_local(data_dir=...)` to build the local manifest without fetching from OpenNeuro.

A team that already has a converted artifact can start at Stage 6.

## Parallel workflows

Multiple datasets can be inspected in parallel before committing to any download:

```python
import asyncio
from qortex import Dataset

ids = ["ds000001", "ds000002", "ds004130"]
datasets = [Dataset(d) for d in ids]

# doctor() calls are fast (manifest + sidecar reads only)
reports = [ds.doctor() for ds in datasets]
usable = [r for r in reports if r.state != "not_usable"]
```
