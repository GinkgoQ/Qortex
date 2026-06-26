# Artifacts

An artifact is the output of a conversion run. It lives on local disk and contains one subdirectory per split plus an `artifact_manifest.json`.

```
artifacts/ds004130/
  artifact_manifest.json
  train/
    shard-000000.parquet
    shard-000001.parquet
    ...
  val/
    shard-000000.parquet
  test/
    shard-000000.parquet
```

## Opening an artifact

```python
from qortex import Artifact

art = Artifact.open("artifacts/ds004130/")
```

This loads the manifest but not the data. Data is only read when you call a load method.

## Artifact pages

[**Manifest**](manifest.md) — Reading the artifact_manifest.json: sample counts, feature names, label classes, provenance.

[**Inspect**](inspect.md) — Inspect individual samples and the artifact structure.

[**Visualize samples**](visualize-samples.md) — Plot individual windows from the artifact.

[**Compare splits**](compare-splits.md) — Check that train/val/test splits have similar distributions.

[**ML bridge**](ml-bridge.md) — Load artifacts for PyTorch, scikit-learn, HuggingFace, and Lightning.
