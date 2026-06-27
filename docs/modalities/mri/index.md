# MRI

MRI is the largest modality family on OpenNeuro. Qortex treats four MRI workflows as related but distinct: structural T1w/T2w, functional BOLD, diffusion DWI, and fieldmaps. Each has its own loader, BIDS suffix set, companion requirements, and conversion path.

## Install

```bash
pip install "qortex[mri]"
```

## BIDS structure

```
sub-01/
  anat/
    sub-01_T1w.nii.gz
    sub-01_T1w.json
    sub-01_T2w.nii.gz
  func/
    sub-01_task-rest_bold.nii.gz
    sub-01_task-rest_bold.json
    sub-01_task-rest_events.tsv
  dwi/
    sub-01_dwi.nii.gz
    sub-01_dwi.bval
    sub-01_dwi.bvec
    sub-01_dwi.json
  fmap/
    sub-01_magnitude1.nii.gz
    sub-01_phasediff.nii.gz
    sub-01_phasediff.json
```

## Searching for MRI datasets

```python
from qortex.catalog import DatasetQuery
from qortex.client import OpenNeuroClient

# Local catalog
results = DatasetQuery().modality("mri").min_subjects(30).fetch()

# Live — sorted by downloads
with OpenNeuroClient() as client:
    results = client.search_datasets_rich(modality="MRI", sort_by="downloads", limit=20)
```

CLI:

```bash
qortex search --modality mri --min-subjects 30
qortex search --modality mri --task rest --has-events
```

## MRI pages

[**Structural MRI**](structural-mri.md) — T1w, T2w, FLAIR: inspection, loading, visualization.

[**fMRI / BOLD**](fmri-bold.md) — BOLD timeseries: TR, events, QC panels, windowed conversion.

[**DWI / Diffusion**](dwi-diffusion.md) — Gradient table, b-values, shells, b0/high-b previews.

[**Fieldmaps**](fieldmaps.md) — Magnitude, phasediff, EPI-fieldmap: intended-for relationships.
