# Conversion API

The conversion layer reads a local BIDS dataset, applies windowing and splits, and writes an ML-format artifact.

```python
from qortex import Dataset
from qortex.convert import ConversionPipeline, WindowSpec, SplitSpec

ds = Dataset("ds004130", data_dir="data/")
art = ds.convert(
    output_dir="artifacts/ds004130/",
    format="parquet",
    window=WindowSpec(duration_s=4.0, overlap=0.5, event_aligned=True),
    split=SplitSpec(train=0.7, val=0.15, test=0.15, strategy="subject"),
    label_col="trial_type",
)
```

---

## ConversionPipeline

Orchestrates the full ETL: load → window → split → write format.
Instantiated directly when you need lower-level control, or via `ds.convert()`.

::: qortex.convert.ConversionPipeline
    options:
      show_source: false
      members:
        - run

---

## WindowSpec

Controls how raw signals are cut into fixed-length or event-aligned windows.

::: qortex.convert.WindowSpec
    options:
      show_source: false

**Fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `duration_s` | `float` | required | Window length in seconds |
| `overlap` | `float` | `0.0` | Fraction of window that overlaps with the previous (sliding only) |
| `tmin` | `float` | `0.0` | Seconds before event onset (event-aligned; negative = baseline period) |
| `event_aligned` | `bool` | `False` | If `True`, cut one window per event; if `False`, use fixed-stride sliding |

---

## SplitSpec

Controls how subjects are assigned to train / val / test splits.

::: qortex.convert.SplitSpec
    options:
      show_source: false

**Fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `train` | `float` | `0.7` | Fraction of subjects in the training split |
| `val` | `float` | `0.15` | Fraction of subjects in the validation split |
| `test` | `float` | `0.15` | Fraction of subjects in the test split |
| `seed` | `int` | `42` | Random seed for reproducibility |
| `stratify_by_label` | `bool` | `True` | Balance class distribution across splits |
| `strategy` | `str` | `"subject"` | `"subject"` — no subject spans two splits; `"random"` — sample-level random; `"stratified"` — stratified by label |

Fractions must sum to 1.0; `__post_init__` raises `ValueError` otherwise.

---

## Windowing functions

::: qortex.convert.fixed_windows
    options:
      show_source: false

::: qortex.convert.event_aligned_windows
    options:
      show_source: false

---

## Split functions

::: qortex.convert.apply_split
    options:
      show_source: false

---

## Provenance

::: qortex.convert.build_provenance
    options:
      show_source: false

::: qortex.convert.save_provenance
    options:
      show_source: false

::: qortex.convert.load_provenance
    options:
      show_source: false

---

## FormatWriter protocol

Every output-format writer implements this protocol.

::: qortex.convert.FormatWriter
    options:
      show_source: false
      members:
        - write
        - estimate_size

---

## Output formats

| Format | Extra required | Notes |
|---|---|---|
| `"parquet"` | *(none)* | Default. Columnar, fast random access. |
| `"zarr"` | `qortex[zarr]` | Chunked array store, good for large volumes. |
| `"hdf5"` | `qortex[hdf5]` | Single-file archive. |
| `"webdataset"` | *(none)* | Tar shards, optimised for streaming. |
| `"huggingface"` | `qortex[hf]` | HuggingFace `datasets` format. |
| `"tfrecord"` | `qortex[tf]` | TensorFlow record format. |
