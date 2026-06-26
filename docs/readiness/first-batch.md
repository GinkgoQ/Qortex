# First Batch

`ds.first_batch()` runs a complete end-to-end pipeline test on the minimum subset of subjects: download → convert → extract one batch → return it.

It is the equivalent of a smoke test. If `first_batch()` succeeds, the full pipeline will work.

## Python

```python
from qortex import Dataset

ds = Dataset("ds004130")
batch = ds.first_batch(
    target_col="trial_type",
    data_dir="data/ds004130/",
    format="parquet",
)
print(batch.X.shape)    # (n_samples, n_features)
print(batch.y[:5])      # first 5 labels
print(batch.subject_ids[:5])
```

The returned object has:

- `X` — numpy array, shape `(n_samples, n_features)`
- `y` — numpy array of label strings or integers
- `subject_ids` — list of subject IDs, one per row in X
- `feature_names` — list of feature names (channel × time, or channel × frequency, etc.)
- `artifact_dir` — path to the temporary artifact written during the test

## What it does internally

1. Calls `ds.minimum(goal="first-batch")` to select 3–5 subjects
2. Downloads those subjects' files to `data_dir`
3. Runs `ConversionPipeline` with default window settings
4. Loads the first batch from the resulting Parquet artifact
5. Returns the batch

If any step fails, a descriptive exception is raised with the stage that failed.

## CLI

```bash
qortex first-batch ds004130 \
    --label trial_type \
    --data-dir data/ds004130/ \
    --format parquet
```

Output on success:

```
first_batch: ok
  Subjects downloaded: 3 (sub-01, sub-02, sub-03)
  Samples:   96
  Features:  6,400 (64 channels × 100 time points)
  Classes:   rest (32), eyes-open (32), task (32)
  Batch shape: (96, 6400)
```

## Debugging first_batch failures

Common failures and what they mean:

**`EventsNotFound`** — events.tsv missing for one of the selected subjects. Usually means the manifest is not consistent — some subjects have events and some do not.

**`EmptyWindowError`** — the window duration is longer than the shortest trial. Set a shorter `window_s`.

**`LFSPointerError`** — the downloaded file is a Git LFS pointer, not real data. The dataset uses DataLad or Git LFS and was not properly fetched. Try `git-annex get` on the raw dataset.

**`NoLabelError`** — the target column is not in events.tsv. Check the actual column names with `ds.events(subject="01")`.

## Related

- [Minimum](minimum.md) — understand the subject selection logic
- [Can train](can-train.md) — binary label check before running first_batch
- [Conversion pipeline](../conversion/pipeline.md) — what happens during the convert step
