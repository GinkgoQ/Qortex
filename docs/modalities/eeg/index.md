# EEG

Electroencephalography (EEG) records scalp electrical potentials from electrode arrays. EEG is the most common electrophysiology modality on OpenNeuro. The EEG signal is high-dimensional in time (sampling rates 250–2048 Hz), requires companion channel and event files, and is the primary target for Qortex's event-aligned window conversion.

## Install

```bash
pip install "qortex[eeg]"    # MNE-Python is used for all EEG reading
```

## BIDS structure

```
sub-01/
  eeg/
    sub-01_task-rest_eeg.set          # EEG data file
    sub-01_task-rest_eeg.fdt          # data companion (EEGLAB)
    sub-01_task-rest_eeg.json         # required sidecar
    sub-01_task-rest_channels.tsv     # channel metadata
    sub-01_task-rest_events.tsv       # trial labels
    sub-01_task-rest_electrodes.tsv   # electrode positions (optional)
    sub-01_coordsystem.json           # coordinate system (optional)
```

## EEG pages

[**EEG files**](files.md) — Formats, channel types, sampling metadata, loading.

[**EEG events**](events.md) — Events.tsv, trial types, label extraction.

[**EEG visualization**](visualization.md) — Butterfly plots, PSD, spectrograms, topomaps, epoched previews.

## Search for EEG datasets

```python
from qortex.catalog import DatasetQuery
from qortex.client import OpenNeuroClient

# Local catalog
results = DatasetQuery().modality("eeg").min_subjects(20).has_events().fetch()

# Live — sorted by downloads
with OpenNeuroClient() as client:
    results = client.search_datasets_rich(modality="EEG", sort_by="downloads", limit=20)
```

CLI:

```bash
qortex search --modality eeg --min-subjects 20 --has-events
```
