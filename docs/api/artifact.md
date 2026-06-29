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
        - validate_schema
        - visualize_sample
        - visual_audit

## Schema validation

`validate_schema()` checks label quality across all splits **after conversion** — before you start a training run. It is the artifact-level complement to the pre-download readiness checks.

```python
result = art.validate_schema(
    label_col="trial_type",      # column to inspect; auto-detected when None
    min_samples_per_class=10,    # error on train, warning on val/test
    max_null_fraction=0.05,      # fraction of missing labels allowed per split
)

if not result["ok"]:
    for err in result["errors"]:
        print("ERROR:", err)
for warn in result["warnings"]:
    print("WARN:", warn)
```

The returned `dict` has the following keys:

| Key | Type | Description |
|---|---|---|
| `ok` | `bool` | `True` when no errors were found |
| `errors` | `list[str]` | Blocking problems (null fraction exceeded, missing label col, train class too small) |
| `warnings` | `list[str]` | Non-blocking issues (val/test class under threshold) |
| `label_col` | `str \| None` | Column that was inspected |
| `stats` | `dict` | Per-split class counts and null fraction |

Labels present in `train` but absent from `val` or `test` (or vice-versa) are reported as warnings so you can catch silent class mismatches before training.
