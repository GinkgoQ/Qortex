# fNIRS

Functional near-infrared spectroscopy (fNIRS) measures haemodynamic responses by detecting changes in oxygenated (HbO) and deoxygenated (HbR) haemoglobin concentrations via light absorption. fNIRS is non-invasive and increasingly available in OpenNeuro BIDS datasets.

## Install

```bash
pip install "qortex[eeg]"    # MNE-Python reads SNIRF and NIRX files
```

## BIDS structure

```
sub-01/
  nirs/
    sub-01_task-rest_nirs.snirf
    sub-01_task-rest_nirs.json
    sub-01_task-rest_channels.tsv
    sub-01_task-rest_events.tsv
    sub-01_optodes.tsv
    sub-01_coordsystem.json
```

## fNIRS pages

[**fNIRS files**](files.md) — SNIRF, NIRX formats, HbO/HbR channels, optode metadata, loading.

## Search for fNIRS datasets

```python
from qortex.catalog import DatasetQuery
from qortex.client import OpenNeuroClient

results = DatasetQuery().modality("nirs").fetch()

with OpenNeuroClient() as client:
    results = client.search_datasets_rich(modality="NIRS", sort_by="downloads")
```
