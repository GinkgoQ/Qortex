# iEEG Files

iEEG signal files are structurally similar to EEG but recorded from implanted electrodes at much higher spatial specificity. Qortex reads iEEG files using the same MNE-Python infrastructure as EEG.

## Supported formats

| Extension | Format |
|-----------|--------|
| `.edf` | European Data Format — most common for clinical iEEG |
| `.bdf` | BioSemi |
| `.vhdr` + `.vmrk` + `.eeg` | BrainVision |
| `.set` | EEGLAB |
| `.fif` | MNE native |
| `.nwb` | Neurodata Without Borders |

## Check before download

```python
from qortex import Dataset

ds = Dataset("ds003688")
ieeg_files = ds.files(datatypes=["ieeg"])
for f in ieeg_files:
    print(f.subject, f.extension, f.path)
```

## Inspect after download

```python
info = ds.nifti_info("sub-01/ieeg/sub-01_task-memory_ieeg.edf")

info["sfreq"]                # 1000.0  (Hz)
info["n_channels"]           # 128
info["channel_type_counts"]  # {"seeg": 124, "eog": 2, "misc": 2}
info["n_bad_channels"]       # 3
info["powerline_freq"]       # 60.0
```

## Load into Python

```python
from qortex.parse.ieeg import IEEGLoader
from pathlib import Path

loader = IEEGLoader()
record = loader.load(file_record, local_path=Path("sub-01/ieeg/sub-01_task-memory_ieeg.edf"))

record.sfreq          # 1000.0
record.n_channels     # 124  (ieeg-type channels only)
record.duration       # 600.0  (seconds)
record.channel_names  # ["LA1", "LA2", "LB1", ...]
record.channel_types  # ["seeg", "seeg", "seeg", ...]

data = loader.to_numpy(record)   # (n_ieeg_channels, n_times) float64 array
```

## Channel types

BIDS distinguishes several iEEG channel types in `channels.tsv`:

| Type | Description |
|------|-------------|
| `SEEG` | Stereo-EEG depth electrode contact |
| `ECOG` | Electrocorticography grid or strip |
| `LFP` | Local field potential |
| `DBS` | Deep brain stimulation recording |

These are exposed as `channel_type_counts` in the inspect output.

## Related

- [Electrodes and coordinates](electrodes-and-coordinates.md)
- [EEG visualization](../eeg/visualization.md) — butterfly, PSD, epoched views work for iEEG too
