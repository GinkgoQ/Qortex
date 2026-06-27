# PET

Positron emission tomography (PET) measures radiotracer concentration in tissue over time. PET datasets on OpenNeuro are quantitative: files contain radioactivity values, and the JSON sidecar carries all the physics metadata needed to interpret them (tracer name, injected activity, body weight, frame timing).

## Install

```bash
pip install "qortex[mri]"    # nibabel is sufficient for PET NIfTI loading
```

## BIDS structure

```
sub-01/
  pet/
    sub-01_trc-18FFDG_pet.nii.gz    # radiotracer image
    sub-01_trc-18FFDG_pet.json      # required sidecar
    sub-01_trc-18FFDG_blood.tsv     # optional blood data
    sub-01_trc-18FFDG_blood.json
```

The `trc-` entity identifies the tracer. PET files always use the `pet` datatype and `pet` suffix.

## Search for PET datasets

```python
from qortex.catalog import DatasetQuery
from qortex.client import OpenNeuroClient

results = DatasetQuery().modality("pet").fetch()

with OpenNeuroClient() as client:
    results = client.search_datasets_rich(modality="PET", sort_by="downloads")
```

## PET pages

[**PET metadata**](metadata.md) — Tracer name, injected activity, frame times, SUV normalization, BIDS sidecar fields.

[**PET visualization**](visualization.md) — Volume viewing, colormaps, overlay on structural MRI.
