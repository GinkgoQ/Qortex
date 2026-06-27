# MEG Files

MEG raw files are large (often 1–10 GB per run) and format-specific. Qortex uses MNE-Python to read all common MEG formats and selects gradiometer and magnetometer channels automatically.

## Inspect before download

The manifest tells you file sizes and paths before downloading. For a typical 6-minute resting MEG run in `.fif` format, expect 500 MB–2 GB.

```python
from qortex import Dataset

ds = Dataset("ds000117")
meg_files = ds.files(datatypes=["meg"])
for f in meg_files:
    print(f.subject, f.path, f"{f.size / 1e9:.1f} GB" if f.size else "size unknown")
```

## Inspect after download

```python
info = ds.nifti_info("sub-01/meg/sub-01_task-facerecognition_meg.fif")
# Note: nifti_info works on MEG files too via the modality-specific inspector

info["sfreq"]                 # 1100.0 Hz
info["n_channels"]            # 306
info["channel_type_counts"]   # {"grad": 204, "mag": 102, "stim": 1}
info["meg_channel_count"]     # 306
info["sss_applied"]           # True  — Maxwell Spatial Suppression detected
info["n_bad_channels"]        # 2
info["powerline_freq"]        # 50  (Europe) or 60 (North America)
```

## Load into Python

```python
from qortex.parse.meg import MEGLoader
from pathlib import Path

loader = MEGLoader()
record = loader.load(file_record, local_path=Path("sub-01/meg/sub-01_task-rest_meg.fif"))

record.sfreq          # 1100.0
record.n_channels     # 306 (after excluding bad channels and ref channels)
record.duration       # 360.0  (seconds)
record.channel_names  # ["MEG0111", "MEG0112", ...]
record.channel_types  # ["grad", "grad", "mag", ...]

record.metadata["sss_applied"]    # True
record.metadata["bad_channels"]   # ["MEG0141"]
record.metadata["powerline_freq"] # 50.0

data = loader.to_numpy(record)    # (n_channels, n_times) float64 array
```

Qortex selects magnetometers and gradiometers (`meg=True`) and excludes bad channels and reference channels automatically. Pass a `SignalRecord` directly to the `TimeSeriesViewer` for visualization.

## Maxwell Spatial Suppression (SSS/MaxFilter)

SSS removes environmental noise from Elekta/MEGIN systems. When SSS has been applied, the signal has already been projected and the effective rank is reduced (typically to 64 or 80). Qortex detects SSS from the processing history in the `.fif` file header and reports it via `sss_applied`.

For ML: after SSS, the number of meaningful components is the SSS basis rank, not the raw channel count. Expect to apply PCA or ICA before windowed conversion.

## Lazy loading

Large MEG files should be loaded lazily to avoid memory exhaustion:

```python
record = loader.lazy_load(file_record, local_path=...)
# raw MNE object backed by disk — data read on first access
```

## Related

- [MEG events](events.md)
- [EEG viewer](../../visualization/eeg-viewer.md) — visualization also works for MEG
- [DWI QC](../../visualization/dwi-qc.md)
