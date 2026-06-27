# MEG

Magnetoencephalography (MEG) measures tiny magnetic fields produced by neuronal currents using arrays of SQUID sensors. MEG datasets on OpenNeuro are large raw signal files paired with channel metadata, head-position files, and event tables.

## Install

```bash
pip install "qortex[eeg]"    # MNE-Python is used for all MEG and EEG reading
```

## BIDS structure

```
sub-01/
  meg/
    sub-01_task-rest_meg.fif
    sub-01_task-rest_meg.json
    sub-01_task-rest_channels.tsv
    sub-01_task-rest_events.tsv
    sub-01_coordsystem.json
    sub-01_headshape.pos
```

## Supported formats

| Format | Extension | System |
|--------|-----------|--------|
| MNE / Elekta | `.fif` | Neuromag (now MEGIN) TRIUX, VectorView |
| CTF | `.ds` (directory) | CTF MEG |
| KIT / Yokogawa | `.sqd`, `.con` | KIT system |
| BTi / 4D | `.4d` | 4D Neuroimaging |

## Search for MEG datasets

```python
from qortex.catalog import DatasetQuery
from qortex.client import OpenNeuroClient

results = DatasetQuery().modality("meg").fetch()

with OpenNeuroClient() as client:
    results = client.search_datasets_rich(modality="MEG", sort_by="downloads", limit=10)
```

## MEG pages

[**MEG files**](files.md) — File formats, channel types, SSS detection, loading.

[**MEG events**](events.md) — Events.tsv, stim channel events, trial types, label readiness.
