# Artifact API

An `Artifact` is the output of a `ConversionPipeline` run — a directory containing split subdirectories and an `artifact_manifest.json`.

```python
from qortex import Artifact

art = Artifact.open("artifacts/ds004130/")
art.summary()
```

::: qortex.Artifact
    options:
      show_source: false
      members:
        - open
        - summary
        - sklearn
        - torch
        - compare_splits
        - check_leakage
        - validate_contract
        - visualize_sample
        - visual_audit
