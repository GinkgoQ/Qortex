# EEG Visualization

Qortex visualizes EEG signals using MNE-Python for rendering and Plotly for interactive HTML output.

## Install

```bash
pip install "qortex[eeg,visual]"
```

## Butterfly plot

Overlay all EEG channels on a single time axis. Reveals flat or noisy channels, common artifacts, and overall signal quality.

```python
from qortex.visualize import TimeSeriesViewer

viewer = TimeSeriesViewer("data/ds004130/sub-01/eeg/sub-01_task-rest_eeg.set")
fig = viewer.butterfly()
fig.show()
```

CLI:

```bash
qortex visualize data/ds004130/sub-01/eeg/sub-01_task-rest_eeg.set --mode interactive
```

## Power spectral density

PSD reveals frequency-band characteristics and powerline noise:

```python
fig = viewer.psd(fmin=1, fmax=80)
fig.show()
```

Expected features in clean EEG: alpha peak (8–12 Hz), 1/f slope, sharp 50/60 Hz notch if filtered.

## Spectrogram

Time–frequency representation:

```python
fig = viewer.spectrogram(
    channel="Oz",
    fmin=1,
    fmax=50,
    baseline=(-0.2, 0.0),    # baseline period in seconds
)
fig.show()
```

## Topomap

Scalp distribution map at a specific time or frequency:

```python
fig = viewer.topomap(time=0.1)           # μV at 100 ms
fig = viewer.topomap(freq_band=(8, 12))  # alpha band power
fig.show()
```

Topomap requires electrode positions from `electrodes.tsv` or a standard 10-20 layout. Qortex falls back to a standard layout if no positions are available.

## Epoched preview

Event-aligned epochs overlaid per trial type:

```python
fig = viewer.epoched(
    events="data/ds004130/sub-01/eeg/sub-01_task-rest_events.tsv",
    event_id={"rest": 1, "task": 2},
    tmin=-0.2,
    tmax=0.8,
)
fig.show()
```

## EEG dashboard

All panels combined in one local HTML report:

```python
viewer.dashboard("sub01_eeg_qc.html")
```

Or from CLI:

```bash
qortex visualize data/ds004130/sub-01/eeg/sub-01_task-rest_eeg.set \
    --mode interactive --output eeg_qc.html
```




## Related

- [EEG files](files.md)
- [EEG events](events.md)
- [EEG viewer reference](../../visualization/eeg-viewer.md)
