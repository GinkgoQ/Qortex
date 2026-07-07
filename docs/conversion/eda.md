# Exploratory Data Analysis

`ds.eda()` runs signal statistics and label analysis on downloaded data before committing to a full conversion. It answers whether the data looks clean and whether the labels are usable.

## Python

```python
from qortex import Dataset

ds = Dataset("ds004130", data_dir="data/ds004130/")
report = ds.eda(label_col="trial_type")
report.show()
```

## CLI

```bash
qortex eda ds004130 --output reports/ds004130_eda.html
```

## What EDA reports

### Label landscape

Per-class sample counts across all subjects. Which subjects are missing which classes. See [Label readiness](../readiness/label-readiness.md).

### Signal statistics (per subject, per file)

For each data file in the dataset:

- Sampling rate (EEG) / TR (fMRI)
- Duration in seconds
- Number of channels / voxels
- Mean signal amplitude
- Signal range (min, max, 1st/99th percentile)
- Channels with flatline signal (all-zero or constant) — likely bad channels
- Files shorter than the requested window duration (would be skipped at conversion)

### Global class distribution

Stacked bar chart showing sample counts per class per subject.

### Recommended window parameters

Based on the minimum recording duration and event durations, EDA suggests:

```
Recommended window: 30.0 s (max within shortest recording)
Recommended overlap: 0.5 (standard for sliding windows)
Estimated samples per split:
  Train (60 subjects): ~1,200
  Val   (14 subjects): ~  270
  Test  (13 subjects): ~  270
```

## Example output

```
EDA: ds004130
Data directory: data/ds004130/

Label column: trial_type
Classes: rest (240), eyes-open (236), task (88) [IMBALANCED — task has 3× fewer samples]

Signal summary (88 subjects):
  Sampling rate:  256 Hz (all subjects)
  Duration:       10 min 0 s (min), 10 min 0 s (max)
  Channels:       64 (all subjects)
  Bad channels:   1 subject has flatline on Fp2

Files too short for 30 s window: 0
Missing events.tsv: 2 subjects (sub-23, sub-47)
```

## Limitations

- EDA requires a local data directory. It does not work against the remote manifest alone.
- Signal statistics are computed from small reads (first N seconds) to avoid loading full recordings.
- Bad channel detection uses a simple flatline heuristic (std < 0.01 μV). Proper bad channel detection requires MNE-Python's interpolation pipeline.
