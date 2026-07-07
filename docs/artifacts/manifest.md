# Artifact Manifest

`artifact_manifest.json` describes the full artifact: where the data came from, how it was converted, and what it contains.

## Reading the manifest

```python
from qortex import Artifact

art = Artifact.open("artifacts/ds004130/")
mf = art.manifest

print(mf.source_dataset)     # "ds004130"
print(mf.source_snapshot)    # "1.2.0"
print(mf.format)             # "parquet"
print(mf.n_samples)          # total samples (all splits)
print(mf.label_classes)      # ["rest", "eyes-open", "task"]
print(mf.feature_names[:5])  # first 5 feature column names
print(mf.splits)             # {"train": {...}, "val": {...}, "test": {...}}
```

## Split-level summary

```python
for split_name, split_info in mf.splits.items():
    print(f"{split_name}: {split_info['n_samples']} samples, {split_info['n_subjects']} subjects")
```

## Subject assignment

```python
print(mf.train_subjects)  # list of subject ID strings
print(mf.val_subjects)
print(mf.test_subjects)
```

## Conversion parameters

```python
print(mf.window.duration_s)     # 30.0
print(mf.window.overlap)        # 0.5
print(mf.split.strategy)        # "subject"
print(mf.split.seed)            # 42
print(mf.split.val_frac)        # 0.15
```

## Timestamps and versions

```python
print(mf.created_at)          # datetime
print(mf.qortex_version)      # "0.3.1"
```

## JSON structure

See [Provenance](../conversion/provenance.md) for the full JSON schema.








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
