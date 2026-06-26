# Output Formats

Qortex supports six output formats for converted artifacts.

## Parquet

Default format. Each split is written as a collection of Parquet files (shards). Each row is one window, columns are flattened features plus metadata (`subject_id`, `label`, `onset_s`, `run_id`).

```bash
pip install qortex  # no extras needed for Parquet
```

```python
art = ds.convert(format="parquet", ...)

# Load with pandas
import pandas as pd
df = pd.read_parquet("artifacts/ds004130/train/")
X = df.drop(columns=["subject_id", "label", "onset_s"]).values
y = df["label"].values

# Load with Qortex helper
X, y = art.sklearn(split="train")
```

**Best for:** tabular ML, scikit-learn, pandas workflows, < 10 GB artifacts.

**Limitation:** Parquet stores data as flat rows. For 4D fMRI data, each row contains a flattened volume. For large volumes (e.g., MNI-space BOLD at 91×109×91), this produces very wide rows that may be slow to read.

## Zarr

Chunked array format. Each split is a Zarr store with shape `(n_samples, ...)`. Supports random access by sample index without reading the full array.

```bash
pip install "qortex[zarr]"
```

```python
art = ds.convert(format="zarr", ...)

import zarr
z = zarr.open("artifacts/ds004130/train/data.zarr")
X = z["X"][0:32]  # first 32 samples
y = z["y"][0:32]
```

**Best for:** large 4D data, random access, Dask pipelines, PyTorch DataPipes.

## HDF5

Similar to Zarr but stored in a single `.h5` file per split. Compatible with h5py and PyTables.

```bash
pip install "qortex[hdf5]"
```

```python
import h5py
with h5py.File("artifacts/ds004130/train/data.h5", "r") as f:
    X = f["X"][:32]
    y = f["y"][:32]
```

**Best for:** legacy pipelines that already use HDF5, MATLAB interop.

**Limitation:** A single `.h5` file can become a bottleneck under parallel reads.

## WebDataset

Sharded tar archives, each containing one sample as a pair of files: `{key}.npy` (features) and `{key}.cls` (label). Compatible with the WebDataset library.

```bash
pip install "qortex[torch]"  # webdataset is bundled with the torch extra
```

```python
import webdataset as wds

ds_wds = wds.WebDataset("artifacts/ds004130/train/shard-{000000..000099}.tar")
for sample in ds_wds:
    X = sample["npy"]
    y = sample["cls"]
```

**Best for:** large-scale distributed training, cloud storage (S3, GCS).

## HuggingFace Dataset

Written as a HuggingFace `datasets.Dataset` on disk. Can be pushed directly to the HuggingFace Hub.

```bash
pip install "qortex[hf]"
```

```python
from datasets import load_from_disk

hf_ds = load_from_disk("artifacts/ds004130/train/")
X = hf_ds["X"]
y = hf_ds["label"]
```

Push to Hub:

```python
hf_ds.push_to_hub("my-org/ds004130-eeg-parquet")
```

**Best for:** sharing processed neuroimaging datasets on the HuggingFace Hub.

## TFRecord

Serialized TensorFlow Example protos.

```bash
pip install "qortex[tf]"
```

```python
import tensorflow as tf

raw_ds = tf.data.TFRecordDataset("artifacts/ds004130/train/*.tfrecord")
```

**Best for:** TensorFlow training pipelines.

**Limitation:** TFRecord is not human-readable and cannot be inspected without TensorFlow. Prefer Parquet for data inspection tasks.

## Format comparison

| Format | Random access | Human-readable | Cloud-friendly | Extra needed |
|--------|--------------|----------------|---------------|--------------|
| Parquet | Partial (by row group) | Yes (via pandas) | Yes | None |
| Zarr | Yes | No | Yes | `zarr` |
| HDF5 | Yes | No | Limited | `hdf5` |
| WebDataset | Sequential | No | Yes | `torch` |
| HuggingFace | Yes | Yes (via Hub) | Yes | `hf` |
| TFRecord | Sequential | No | Yes | `tf` |
