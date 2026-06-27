# Sources

Every source adapter implements three methods: `probe()` returns a `SourceProfile` from headers only; `stream()` yields windowed data as `QortexTimeSeries` or `QortexVolume`; `replay(speed)` simulates real-time playback.

Source type is set by `source.type` in the pipeline YAML. The adapter is chosen by `make_source_adapter()`.

## Local file (EDF/BDF/FIF)

```yaml
source:
  type: local_file          # or: edf, bdf, fif, mff, cnt
  path: data/sub-01.edf
  modality: eeg             # eeg | meg | ieeg | fnirs
```

Uses MNE to read any format MNE supports. `probe()` reads the file info without loading raw data. Yields `QortexTimeSeries` windows with shape `[channels, samples]`.

Auto-detection: if `type` is omitted, the extension triggers automatic routing — `.edf` → `LocalFileAdapter`, `.nwb` → `NWBAdapter`, `.xdf` → `XDFAdapter`, image extensions → `ImageVideoAdapter`, and so on.

## BIDS dataset

```yaml
source:
  type: bids
  path: data/ds004130
  modality: eeg
  subject: "01"             # optional — single subject or list
  session: null             # optional
  task: rest                # optional
```

Uses `mne_bids.BIDSPath` to locate and load recordings. `probe()` reads the BIDS sidecar without loading signal data.

## DICOM folder

```yaml
source:
  type: dicom
  path: dicom/study/
```

Groups `.dcm` files by `SeriesInstanceUID`. Sorts slices by `InstanceNumber` or `ImagePositionPatient` z-coordinate. Applies `RescaleSlope` and `RescaleIntercept` to convert to Hounsfield units. Builds a 4×4 affine from `ImageOrientationPatient`, `ImagePositionPatient`, `PixelSpacing`, and `SliceThickness`.

Returns `QortexVolume` with `axes="zyx"`, `units="HU"`, `coordinate_frame="patient_lps"`.

Patient names and identifiers are never written to logs.

Auto-detection: a directory without `dataset_description.json` but containing `.dcm` files is routed to `DICOMFolderAdapter`.

## DICOMweb

```yaml
source:
  type: dicomweb
  path: https://dicomweb.server/wado/rs
  extra:
    study_uid: "1.2.3.4"
    series_uid: "1.2.3.4.5"
    auth:
      type: bearer
      token: "${DICOM_TOKEN}"   # or: type: basic, username: ..., password: ...
```

Uses QIDO-RS to fetch instance metadata and WADO-RS to retrieve pixel data. Authentication is bearer token or basic auth, passed via `spec.extra["auth"]`.

URL pattern: `{base}/studies/{study_uid}/series/{series_uid}/instances`

## NWB

```yaml
source:
  type: nwb
  path: data/sub-01.nwb
```

Opens the NWB file with `pynwb.NWBHDF5IO`. Finds `ElectricalSeries` objects in the acquisition group. NWB stores data as `[time, channels]` — the adapter transposes to `[channels, time]` before returning. Channel names are extracted from electrode labels when present. Sampling rate comes from `nominal_srate` or is inferred from timestamp differences.

## XDF

```yaml
source:
  type: xdf
  path: recording.xdf
  query:                    # optional — select one stream
    type: EEG               # or: name: my_stream
```

Uses `pyxdf.load_xdf()`. Without a query, the first EEG stream is used. XDF stores `[time, channels]` — transposed to `[channels, time]`. `replay(speed)` sleeps between windows to simulate real-time timing.

## LSL stream

```yaml
source:
  type: lsl
  query:
    type: EEG               # LSL stream type
    name: null              # or match by name
  extra:
    wait_time_s: 5.0        # how long to wait for a stream to appear
```

Calls `pylsl.resolve_streams(wait_time=5.0)`. Uses the ring buffer for windowed streaming. Channel names are extracted from the LSL stream's XML descriptor. `pull_chunk(timeout=win_dur/4, max_samples=512)` is polled in a loop.

For real-time use, the ring buffer handles the producer-consumer gap between LSL's chunk delivery and the pipeline's window size.

## BrainFlow board

```yaml
source:
  type: brainflow
  extra:
    board_id: 0             # BoardIds enum integer
    serial_port: /dev/ttyUSB0
    mac_address: null
    ip_address: null
    ip_port: null
```

`probe()` calls `BoardShim.get_eeg_channels()`, `get_sampling_rate()`, and `get_eeg_names()` without opening a board session. Session opens on first `stream()` call. Uses ring buffer for windowed streaming. `board.get_board_data(win_samples)` is polled every `win_dur / 4` seconds.

## Image and video

```yaml
source:
  type: image               # or: video
  path: frames/             # directory of images, or single video file
```

Images (`.png`, `.jpg`, `.tif`, `.bmp`, `.webp`) are loaded with PIL/Pillow. Videos (`.mp4`, `.avi`, `.mov`, `.mkv`) use OpenCV. `stream()` yields batches of `window_spec.duration_s * fps` frames as `QortexVolume(axes="nhwc")`.

## Source profile

`probe()` returns a `SourceProfile` regardless of adapter:

```python
profile.source_id           # str — path or stream name
profile.modality            # "eeg" | "meg" | "mri" | "dicom" | ...
profile.n_channels          # int | None
profile.sampling_rate_hz    # float | None
profile.duration_s          # float | None — None for live streams
profile.channel_names       # list[str] | None
profile.shape               # tuple[int, ...] | None — for volumes
profile.dtype               # str — "float32" etc.
profile.coordinate_frame    # str | None — "patient_lps" for DICOM
profile.evidence            # dict[str, EvidenceStatus]
```

All fields carry an `EvidenceStatus`: `confirmed` (read from file), `inferred` (derived from other fields), `missing`, or `unknown`. The `CompatibilityEngine` uses these statuses to decide whether a mismatch is a hard blocker or an uncertainty.

## Internal data types

Adapters yield one of these types into the pipeline:

| Type | Shape convention | Use case |
|---|---|---|
| `QortexTimeSeries` | `[channels, samples]` | EEG, MEG, iEEG, LFP, fNIRS |
| `QortexVolume` | `[z, y, x]` or `[n, h, w, c]` | MRI, CT, DICOM, image batches |
| `QortexImageSeries` | `[n, h, w, c]` with timestamps | Ordered 2D images |
| `QortexVideo` | `[n, h, w, c]` with fps | Video frames |
| `QortexEmbeddingTable` | `[n, d]` | Embedding vectors |
| `QortexStream` | descriptor only | Live stream (not data) |
