# fNIRS Files

fNIRS data in BIDS uses the `.snirf` format (Shared Near Infrared spectroscopy Format) as the primary container. Legacy datasets may use NIRX format. Qortex separates HbO and HbR channels automatically.

## Supported formats

| Extension | Format |
|-----------|--------|
| `.snirf` | SNIRF (BIDS required) — HDF5-based |
| `.nirs` | Homer2 legacy format (NIRX directory) |

## Inspect before download

```python
from qortex import Dataset

ds = Dataset("ds004148")
nirs_files = ds.files(datatypes=["nirs"])
for f in nirs_files:
    print(f.subject, f.extension, f.path)
```

## Inspect after download

```python
info = ds.nifti_info("sub-01/nirs/sub-01_task-rest_nirs.snirf")

info["sfreq"]                  # 7.8125  (Hz — low sampling rate typical for fNIRS)
info["n_channels"]             # 52  (HbO + HbR channels)
info["channel_type_counts"]    # {"hbo": 26, "hbr": 26}
info["duration_s"]             # 600.0
info["short_channel_count"]    # 4  (short-separation channels)
info["manufacturer"]           # "NIRx"
```

## Load into Python

```python
from qortex.parse.fnirs import FNIRSLoader
from pathlib import Path

loader = FNIRSLoader()
record = loader.load(file_record, local_path=Path("sub-01/nirs/sub-01_task-rest_nirs.snirf"))

record.sfreq         # 7.8125
record.n_channels    # 52
record.duration      # 600.0
record.channel_names # ["S1_D1 hbo", "S1_D1 hbr", "S1_D2 hbo", ...]
record.channel_types # ["hbo", "hbr", "hbo", ...]

record.metadata["n_hbo_channels"]  # 26
record.metadata["n_hbr_channels"]  # 26
record.metadata["short_channel_count"]  # 4

data = loader.to_numpy(record)   # (n_channels, n_times) — HbO and HbR interleaved
```

## HbO and HbR channel access

```python
hbo_idx = record.metadata["hbo_picks"]   # indices into channel list
hbr_idx = record.metadata["hbr_picks"]

data_hbo = data[hbo_idx, :]   # (n_source_detector_pairs, n_times)
data_hbr = data[hbr_idx, :]
```

## Optode metadata

```python
import polars as pl
optodes = pl.read_csv(
    "data/ds004148/sub-01/nirs/sub-01_optodes.tsv",
    separator="\t",
)
# name   type    x       y       z
# S1     source  -35.2   75.1    22.4
# D1     detector -30.8  80.3    19.2
```

## Conversion

fNIRS converts as a low-sampling-rate signal. Event-aligned windows are common:

```python
art = ds.convert(
    data_dir="data/ds004148/",
    output_dir="artifacts/",
    format="parquet",
    datatypes=["nirs"],
    window=dict(mode="event_aligned", tmin=-2.0, tmax=20.0),
    label_col="trial_type",
)
```

Each sample is a `(n_channels, n_time_points)` array. Low sampling rate means small arrays even for long windows.

## Related

- [fNIRS overview](index.md)
- [EEG events](../eeg/events.md) — events work the same for fNIRS
