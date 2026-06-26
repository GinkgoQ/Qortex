# Provenance

Every Qortex artifact contains an `artifact_manifest.json` that records the full provenance of the conversion. This file is the artifact's identity — it lets you trace any model prediction back to the exact dataset version, conversion parameters, and split assignment that produced the training data.

## artifact_manifest.json structure

```json
{
  "qortex_version": "0.3.1",
  "created_at": "2024-01-15T14:23:00Z",
  "source_dataset": "ds004130",
  "source_snapshot": "1.2.0",
  "source_doi": "10.18112/openneuro.ds004130.v1.2.0",
  "format": "parquet",
  "label_col": "trial_type",
  "label_classes": ["rest", "eyes-open", "task"],
  "feature_names": ["Fp1_0", "Fp1_1", ..., "O2_7679"],
  "n_samples": 1740,
  "splits": {
    "train": {"n_samples": 1200, "n_subjects": 61},
    "val":   {"n_samples": 270,  "n_subjects": 14},
    "test":  {"n_samples": 270,  "n_subjects": 13}
  },
  "train_subjects": ["01", "02", ...],
  "val_subjects":   ["63", "64", ...],
  "test_subjects":  ["77", "78", ...],
  "window": {
    "duration_s": 30.0,
    "overlap": 0.5,
    "event_aligned": false
  },
  "split": {
    "strategy": "subject",
    "val_frac": 0.15,
    "test_frac": 0.15,
    "stratify_by_label": true,
    "seed": 42
  },
  "subjects_included": ["01", "02", ..., "88"],
  "tasks_included": ["rest"],
  "suffixes_included": ["eeg"]
}
```

## Reading provenance from Python

```python
from qortex import Artifact

art = Artifact.open("artifacts/ds004130/")
mf = art.manifest

print(mf.source_dataset)    # "ds004130"
print(mf.source_snapshot)   # "1.2.0"
print(mf.created_at)        # datetime object
print(mf.qortex_version)    # "0.3.1"
print(mf.window.duration_s) # 30.0
print(mf.split.seed)        # 42
```

## Provenance from individual VisualResult

Each rendered figure also carries a provenance dict:

```python
from qortex.visualize import fmri_summary

result = fmri_summary("sub-01_task-rest_bold.nii.gz")
prov = result.to_provenance_dict()
print(prov["qortex_version"])
print(prov["file"])
print(prov["render_params"])
```

## Using provenance in model cards

When publishing a model trained on Qortex artifacts, include the artifact manifest in the model card:

```python
import json

with open("artifacts/ds004130/artifact_manifest.json") as f:
    prov = json.load(f)

print(f"Trained on {prov['source_dataset']} @ {prov['source_snapshot']}")
print(f"DOI: {prov['source_doi']}")
print(f"Split seed: {prov['split']['seed']}")
```

## Limitations

- Provenance covers the Qortex conversion step only. If the source data was preprocessed before indexing (e.g., with fMRIPrep), that preprocessing is not recorded in `artifact_manifest.json`.
- Subject IDs in the manifest are the raw BIDS subject labels. They do not include PHI — OpenNeuro datasets use coded participant IDs.
