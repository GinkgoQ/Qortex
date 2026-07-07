# Artifact Visualization

Once a dataset has been converted to a Qortex artifact (Parquet, Zarr, HDF5, etc.), you can inspect individual samples and compare splits without loading the full dataset.

## Visualize a sample

```python
from qortex import Artifact

art = Artifact.open("artifacts/ds004130_parquet/")
art.visualize_sample(split="train", index=0)
```

This opens a figure showing the first sample from the training split:

- For EEG: a butterfly plot of the window
- For fMRI: the extracted volume window (mean across time if 4D)
- For tabular: a bar chart of feature values

Pass a subject ID instead of an integer index:

```python
art.visualize_sample(split="train", subject="01", label="rest")
```

## Visual audit of an artifact

```python
audit = art.visual_audit()
audit.show()
```

The artifact visual audit shows:

- Sample counts per split
- Class distribution per split
- Feature range summary (min, max, mean per feature)
- Any suspicious samples (all-zero windows, NaN features)

## Compare splits

```python
art.compare_splits()
```

Produces a side-by-side comparison of the train, val, and test splits:

- Class distribution histogram (should be similar across splits)
- Subject count per split
- Mean signal amplitude per split (should be similar)
- Feature correlation matrix (should match between train and val)

Any major differences between splits suggest a data issue (e.g., non-stationary signal, inadvertent stratification).

## CLI

```bash
qortex artifact-visualize artifacts/ds004130_parquet/ --split train --index 0
qortex artifact-visualize artifacts/ds004130_parquet/ --compare-splits
```

## Inspecting the artifact manifest

```python
print(art.manifest.n_samples)
print(art.manifest.splits)        # {"train": 1200, "val": 260, "test": 280}
print(art.manifest.feature_names) # list of feature names
print(art.manifest.label_classes) # ["rest", "eyes-open", "task"]
print(art.manifest.source_dataset) # "ds004130@1.2.0"
print(art.manifest.created_at)    # ISO timestamp
```




## Related

- [Artifact overview](../artifacts/index.md) — the Artifact class and its splits
- [ML bridge](../artifacts/ml-bridge.md) — loading artifacts for training
- [Compare splits](../artifacts/compare-splits.md) — split quality checks
