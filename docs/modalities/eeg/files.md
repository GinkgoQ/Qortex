# EEG Files

EEG data files on OpenNeuro come in several vendor-specific formats. Qortex uses MNE-Python to read all of them through a single `EEGLoader` interface.

## Supported formats

| Extension | Format | System |
|-----------|--------|--------|
| `.set` + `.fdt` | EEGLAB | MATLAB-based toolbox (most common on OpenNeuro) |
| `.edf` | European Data Format | Clinical / portable |
| `.bdf` | BioSemi Data Format | BioSemi ActiveTwo |
| `.fif` | MNE native / Elekta | Converted from EEG systems |
| `.vhdr` + `.vmrk` + `.eeg` | BrainVision | Brain Products |
| `.cnt` | Neuroscan | Older clinical |
| `.mff` | EGI MFF | Electric Geodesics |
| `.gdf` | General Data Format | Open-source |

## Check before download

```python
from qortex import Dataset

ds = Dataset("ds004130")
eeg_files = ds.files(datatypes=["eeg"])
for f in eeg_files:
    print(f.subject, f.extension, f.size)
```

## Inspect after download

```python
info = ds.nifti_info("sub-01/eeg/sub-01_task-rest_eeg.set")

info["sfreq"]                # 256.0  (Hz)
info["n_channels"]           # 64
info["channel_type_counts"]  # {"eeg": 61, "eog": 2, "misc": 1}
info["n_bad_channels"]       # 0
info["powerline_freq"]       # 50.0  (Hz)
info["eeg_reference"]        # "CZ"
info["software_filters"]     # {"Highpass": "0.016 Hz", "Lowpass": "100 Hz"}
```

## Load into Python

```python
from qortex.parse.eeg import EEGLoader
from pathlib import Path

loader = EEGLoader()
record = loader.load(file_record, local_path=Path("sub-01/eeg/sub-01_task-rest_eeg.set"))

record.sfreq          # 256.0
record.n_channels     # 61  (EEG-type channels only — EOG excluded by default)
record.duration       # 480.0  (seconds)
record.channel_names  # ["Fp1", "Fp2", "F3", ...]
record.channel_types  # ["eeg", "eeg", "eeg", ...]
record.metadata["eeg_reference"]   # "CZ"
record.metadata["bad_channels"]    # []

data = loader.to_numpy(record)   # (n_eeg_channels, n_times) float64 array
```

Qortex selects EEG-type channels and excludes EOG, ECG, EMG, and stimulus channels by default. This is the data used for ML conversion.

## MNE-BIDS integration

Qortex reads EEG via MNE-BIDS when a BIDS root can be inferred, which correctly applies sidecar inheritance for channel metadata. If MNE-BIDS is unavailable, it falls back to direct MNE reading.

## Lazy loading

```python
record = loader.lazy_load(file_record, local_path=...)
# MNE Raw object — data stays on disk until to_numpy() or slicing
```




## Related

- [EEG events](events.md)
- [EEG visualization](visualization.md)
- [Channels metadata](../../dataset/metadata.md)
