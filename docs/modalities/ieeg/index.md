# iEEG

Intracranial EEG (iEEG) records electrical activity directly from the brain surface or depth using implanted electrodes. It encompasses stereo-EEG (sEEG) depth probes, electrocorticography (ECoG) grids and strips, and local field potentials (LFP) from single-unit probes. iEEG datasets require both signal files and electrode localization metadata.

## Install

```bash
pip install "qortex[eeg]"
```

## BIDS structure

```
sub-01/
  ieeg/
    sub-01_task-memory_ieeg.edf
    sub-01_task-memory_ieeg.json
    sub-01_task-memory_channels.tsv
    sub-01_task-memory_events.tsv
    sub-01_electrodes.tsv          # electrode locations
    sub-01_coordsystem.json        # coordinate system definition
```

## iEEG pages

[**iEEG files**](files.md) — Signal formats, channel metadata, loading.

[**Electrodes and coordinates**](electrodes-and-coordinates.md) — Electrode tables, coordinate systems, anatomical references.

## Search for iEEG datasets

```python
from qortex.catalog import DatasetQuery
from qortex.client import OpenNeuroClient

results = DatasetQuery().modality("ieeg").fetch()

with OpenNeuroClient() as client:
    results = client.search_datasets_rich(modality="iEEG", sort_by="downloads")
```
