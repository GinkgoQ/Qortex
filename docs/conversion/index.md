# Conversion

`ConversionPipeline` reads a local BIDS dataset and writes ML-format artifacts. The output is a directory with one subdirectory per split and an `artifact_manifest.json` describing the provenance, sample counts, feature names, and label classes.

## Why a separate conversion step

Raw neuroimaging data is stored in formats optimized for storage and archive, not for training loops. NIfTI and EEGLAB files are read one volume at a time, have no random-access guarantees, and carry metadata in separate sidecar files. Parquet and Zarr expose array slices efficiently, support chunk caching, and pack all metadata inline.

Conversion is a one-time cost. After conversion, loading a batch for training takes milliseconds instead of seconds.

## Conversion pages

[**Pipeline**](pipeline.md) — The full ConversionPipeline API: how formats, windows, splits, and label extraction are composed.

[**Formats**](formats.md) — Parquet, Zarr, HDF5, WebDataset, HuggingFace Dataset, TFRecord — requirements and trade-offs.

[**Windows**](windows.md) — Sliding and event-aligned windows: WindowSpec parameters and how windowing interacts with labels.

[**Splits**](splits.md) — Subject-level splits, stratification, and reproducible random seeds.

[**Provenance**](provenance.md) — What is recorded in artifact_manifest.json and how to trace an artifact back to its source.

[**EDA**](eda.md) — Exploratory data analysis before conversion: signal statistics, label landscape, class imbalance.

## Quickstart

```python
from qortex import Dataset

ds = Dataset("ds004130", data_dir="data/ds004130/")

art = ds.convert(
    output_dir="artifacts/ds004130/",
    format="parquet",
    window=dict(duration_s=30.0, overlap=0.5),
    split=dict(strategy="subject", val_frac=0.15, test_frac=0.15),
    label_col="trial_type",
)
print(art.manifest.n_samples)  # total samples across all splits
```

---

**Next →** [Artifacts](../artifacts/index.md) — open, inspect, and connect your artifact to a training framework.








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
