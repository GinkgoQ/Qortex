# ML Bridge

The ML bridge loads a Qortex artifact into the data structures expected by major ML frameworks.

## scikit-learn

```python
from qortex import Artifact

art = Artifact.open("artifacts/ds004130/")
X_train, y_train = art.sklearn(split="train")
X_val,   y_val   = art.sklearn(split="val")
X_test,  y_test  = art.sklearn(split="test")

# X shape: (n_samples, n_features)
# y shape: (n_samples,) — string labels

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
```

Label encoding: by default, labels are returned as strings. Pass `encode_labels=True` for integer encoding:

```python
X_train, y_train = art.sklearn(split="train", encode_labels=True)
# y_train is np.int64
```

## PyTorch

```python
from qortex import Artifact

art = Artifact.open("artifacts/ds004130/")
train_ds = art.torch(split="train")
val_ds   = art.torch(split="val")

from torch.utils.data import DataLoader
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

for X, y in train_loader:
    print(X.shape, y.shape)
    break
```

Requires: `pip install "qortex[torch]"`

## PyTorch Lightning

```python
from qortex import Artifact

art = Artifact.open("artifacts/ds004130/")
dm = art.lightning_datamodule(batch_size=32)

import lightning as L
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, dm)
```

Requires: `pip install "qortex[lightning]"`

## HuggingFace Datasets

```python
from qortex import Artifact
from datasets import load_from_disk

art = Artifact.open("artifacts/ds004130/")
hf_ds = art.huggingface()

# Or load directly with datasets library (for HuggingFace format artifacts)
hf_ds = load_from_disk("artifacts/ds004130/train/")
```

Requires: `pip install "qortex[hf]"`

## Feature names

```python
print(art.manifest.feature_names[:10])
# ['Fp1_0', 'Fp1_1', ..., 'Fp1_7679', 'Fp2_0', ...]
```

Feature names are `{channel}_{timepoint}` for EEG or `voxel_{x}_{y}_{z}` for fMRI.

## Metadata columns

Every artifact includes metadata columns alongside features:

- `subject_id` — BIDS subject label
- `label` — string label (from `label_col`)
- `onset_s` — window start time in seconds
- `run_id` — BIDS run label (if applicable)

Access metadata separately:

```python
import pandas as pd
df = pd.read_parquet("artifacts/ds004130/train/shard-000000.parquet")
print(df.columns.tolist()[:5])    # feature columns
print(df[["subject_id", "label", "onset_s"]].head())
```

## Memory usage

For large artifacts, use chunked loading:

```python
# Zarr — only loads chunks as needed
import zarr
z = zarr.open("artifacts/ds004130_zarr/train/data.zarr")
X_chunk = z["X"][0:32]  # loads only 32 samples
```

For Parquet, use pyarrow with a filter:

```python
import pyarrow.parquet as pq
ds = pq.ParquetDataset("artifacts/ds004130/train/")
table = ds.read(filters=[("label", "=", "rest")])
```








<!-- qortex-evidence:start -->

## Evidence

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/conversion-split-evidence.png" alt="Subject-safe split chart showing train, validation, and test allocation counts.">
  <figcaption>`ds000001` split plan derived from 16 subjects and 80 candidate BOLD recordings.</figcaption>
</figure>

```bash
qortex convert data/ds000001 artifacts/ds000001 --format parquet --split subject
```

Result artifact: [neuroai-fixture-summary.json](/Qortex/assets/results/neuroai-fixture-summary.json)

<!-- qortex-evidence:end -->
