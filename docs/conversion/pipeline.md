# Conversion Pipeline

`ConversionPipeline` orchestrates format conversion from BIDS to ML-ready artifacts. It chains windowing, label extraction, and splitting into one pass.

## Python

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
```

Or using the pipeline directly:

```python
from qortex.convert import ConversionPipeline, WindowSpec, SplitSpec

pipeline = ConversionPipeline(
    data_dir="data/ds004130/",
    output_dir="artifacts/ds004130/",
    format="parquet",
    window=WindowSpec(duration_s=30.0, overlap=0.5),
    split=SplitSpec(strategy="subject", val_frac=0.15, test_frac=0.15),
    label_col="trial_type",
)
art = pipeline.run()
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | `Path` | required | Local BIDS directory |
| `output_dir` | `Path` | required | Where to write the artifact |
| `format` | `str` | `"parquet"` | Output format |
| `window` | `WindowSpec` or dict | `{"duration_s": 30.0}` | Window parameters |
| `split` | `SplitSpec` or dict | `{"strategy": "subject"}` | Split parameters |
| `label_col` | `str` or None | None | events.tsv column to use as labels |
| `subjects` | `list[str]` or None | None | Subjects to include (all if None) |
| `tasks` | `list[str]` or None | None | Tasks to include (all if None) |
| `suffixes` | `list[str]` or None | None | BIDS suffixes to include |
| `seed` | `int` | `42` | Random seed for split assignment |
| `n_workers` | `int` | `4` | Parallel workers for file reading |
| `overwrite` | `bool` | `False` | Overwrite existing artifact |

## Processing order

1. Scan `data_dir` and build a local manifest
2. Filter files by `subjects`, `tasks`, `suffixes`
3. Assign subjects to splits (see [Splits](splits.md))
4. For each file:
   a. Read the time series (NIfTI / EEG / etc.)
   b. Apply windowing (see [Windows](windows.md))
   c. Extract labels from events.tsv if `label_col` is set
   d. Write windows to the split's output shard
5. Write `artifact_manifest.json`

## Filtering during conversion

You can exclude specific subjects or tasks during conversion:

```python
art = ds.convert(
    output_dir="artifacts/",
    subjects=[f"{i:02d}" for i in range(1, 61)],  # first 60 subjects
    tasks=["rest"],
    label_col="trial_type",
)
```

This is more efficient than post-hoc filtering because unneeded files are never read.

## CLI

```bash
qortex convert data/ds004130 artifacts/ds004130 \
    --format parquet \
    --window 30 \
    --overlap 0.5 \
    --split subject
```

## After conversion

```python
from qortex import Artifact

art = Artifact.open("artifacts/ds004130/")
print(art.manifest.n_samples)
print(art.manifest.splits)
```

See [Formats](formats.md) for format-specific loading.








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

## Related

- [Formats](formats.md) — Parquet, Zarr, HDF5, WebDataset, HuggingFace, TFRecord
- [Windows](windows.md) — WindowSpec: duration, overlap, event alignment
- [Splits](splits.md) — SplitSpec: subject strategy, stratification, seeds
