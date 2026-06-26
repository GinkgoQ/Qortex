# EEG Viewer

`TimeSeriesViewer` renders EEG, MEG, and iEEG recordings from local files. It uses MNE-Python for reading and requires the `eeg` extra.

## Setup

```bash
pip install "qortex[eeg]"
```

## Python

```python
from qortex.visualize.timeseries import TimeSeriesViewer

viewer = TimeSeriesViewer("sub-01_task-rest_eeg.set")
```

MNE reads the channel layout and electrode positions from the `.set` file and its companion `*_channels.tsv` / `*_coordsystem.json`. Pass these explicitly if they are not auto-detected:

```python
viewer = TimeSeriesViewer(
    "sub-01_task-rest_eeg.set",
    channels_path="sub-01_task-rest_channels.tsv",
    coordsystem_path="sub-01_task-rest_coordsystem.json",
)
```

## Butterfly plot

All channels overlaid on the same time axis, color-coded by channel type:

```python
fig = viewer.butterfly(t_start=0, t_stop=10)  # 0–10 seconds
fig.show()
```

The butterfly plot is good for detecting channels with extreme artifacts — they appear as lines far outside the cluster of others.

## Power spectral density

```python
fig = viewer.psd(fmin=1, fmax=80, method="welch")
fig.show()
```

Uses Welch's method (scipy.signal). The plot shows PSD in dB/Hz for each channel, with the mean across channels highlighted. Look for the 1/f drop-off with alpha (8–13 Hz) and beta (13–30 Hz) peaks.

## Spectrogram

```python
fig = viewer.spectrogram(channel="Oz", fmin=1, fmax=80)
fig.show()
```

Uses STFT via scipy. Time is on the x-axis, frequency on the y-axis, power in decibels as color. One channel at a time.

## Topomap

```python
fig = viewer.topomap(freq_band=(8, 12))  # alpha band power
fig.show()
```

Renders a 2D topographic map of power in a frequency band. Electrode positions are taken from the coordsystem JSON or the MNE standard montage. Interpolation uses IDW (inverse distance weighting) with unit-circle normalization.

If no electrode positions are available, the topomap falls back to a summary-only output listing channels by power rank.

## Epoched (ERP)

For event-related potential visualization:

```python
fig = viewer.epoched(
    events_path="sub-01_task-rest_events.tsv",
    event_id="stimulus",    # trial_type value
    tmin=-0.2,              # seconds before onset
    tmax=0.8,               # seconds after onset
)
fig.show()
```

Shows the mean epoch per channel with a shaded standard error band. The time-zero line is the event onset.

## Dashboard

`dashboard()` launches a local Panel server with all views combined in one interactive interface:

```python
viewer.dashboard()
# opens http://localhost:5006
```

Requires the `dashboard` extra: `pip install "qortex[dashboard]"`.

## CLI

```bash
qortex visualize ds004130 \
    --subject 01 \
    --suffix eeg \
    --data-dir data/ds004130/
```

## Supported file formats

MNE reads all standard EEG formats:

- `.set` (EEGLAB)
- `.fif` (MNE-Python, Elekta/Neuromag)
- `.edf` (European Data Format)
- `.cnt` (Neuroscan)
- `.vhdr` / `.vmrk` / `.eeg` (BrainVision)

BrainVision sets need all three files (`*.vhdr`, `*.vmrk`, `*.eeg`) in the same directory.

## Limitations

- `topomap()` with IDW interpolation requires at least 3 channels with known 3D positions.
- Very long recordings (> 1 hour) are read in segments. `butterfly()` shows only the first `max_duration` seconds (default: 30).
- MEG is supported in terms of file reading, but `topomap()` for MEG sensor layouts uses the MNE default layout and may not match custom MEG configurations.
