# Modalities

Qortex works with OpenNeuro and BIDS datasets across neuroimaging, electrophysiology, and behavioral data. This section explains what each modality usually contains, how Qortex detects it, which files matter, what can be inspected before download, and what is currently supported for visualization, readiness checks, and conversion.

---

## What this section covers

Each modality page should answer the same practical questions:

1. What files usually appear in BIDS/OpenNeuro datasets?
2. How does Qortex detect this modality?
3. Which companion files matter?
4. What can be inspected before download?
5. What requires local files?
6. What can be visualized?
7. What can be converted into ML artifacts?
8. What are the current limitations?

The goal is not to duplicate the BIDS specification. The goal is to explain what Qortex can do with each modality in a real OpenNeuro workflow.

---

## Modality map

| Modality   | Common BIDS datatype          | Typical files                                            | Qortex role                               |
| ---------- | ----------------------------- | -------------------------------------------------------- | ----------------------------------------- |
| MRI        | `anat`, `func`, `dwi`, `fmap` | `.nii`, `.nii.gz`, `.json`, `.bval`, `.bvec`             | inspect, download, visualize, QC, convert |
| PET        | `pet`                         | `.nii`, `.nii.gz`, `.json`, tracer metadata              | inspect, visualize, overlay, convert      |
| MEG        | `meg`                         | `.fif`, `.ds`, `.tsv`, `.json`, events                   | inspect, signal preview, events, convert  |
| EEG        | `eeg`                         | `.edf`, `.bdf`, `.vhdr`, `.vmrk`, `.eeg`, `.set`, `.fdt` | inspect, signal preview, events, convert  |
| iEEG       | `ieeg`                        | `.edf`, `.vhdr`, `.set`, electrodes, coordsystem         | inspect, contacts, events, signal preview |
| fNIRS      | `nirs`                        | `.snirf`, `.tsv`, `.json`                                | inspect, metadata, future visualization   |
| Behavioral | `beh`, events files           | `.tsv`, `.csv`, `.json`                                  | labels, readiness, conversion             |

---

## MRI

MRI is the broadest modality family in OpenNeuro. Qortex treats structural MRI, fMRI/BOLD, DWI, and fieldmaps as related but distinct workflows.

<div class="tq-card-grid">
  <div class="tq-card">
    <h3><a href="mri/">MRI overview</a></h3>
    <p>How Qortex detects MRI files, reads NIfTI metadata, handles BIDS entities, and prepares MRI files for visual QC.</p>
  </div>
  <div class="tq-card">
    <h3><a href="mri/structural-mri/">Structural MRI</a></h3>
    <p>T1w, T2w, FLAIR, masks, segmentations, anatomical references, and orthogonal visualization.</p>
  </div>
  <div class="tq-card">
    <h3><a href="mri/fmri-bold/">fMRI / BOLD</a></h3>
    <p>4D BOLD files, events, TR, mean EPI, temporal standard deviation, tSNR, and global signal QC.</p>
  </div>
  <div class="tq-card">
    <h3><a href="mri/dwi-diffusion/">DWI / Diffusion</a></h3>
    <p>DWI NIfTI files, b-values, b-vectors, b0 previews, high-b previews, shell summaries, and gradient plots.</p>
  </div>
  <div class="tq-card">
    <h3><a href="mri/fieldmaps/">Fieldmaps</a></h3>
    <p>Magnitude, phase, phasediff, fieldmap metadata, intended-for relationships, and distortion-correction readiness.</p>
  </div>
</div>

### MRI detection

Qortex primarily detects MRI data from:

- BIDS datatype folders: `anat`, `func`, `dwi`, `fmap`
- BIDS suffixes: `T1w`, `T2w`, `FLAIR`, `bold`, `dwi`, `fieldmap`, `phasediff`, `magnitude`
- File extensions: `.nii`, `.nii.gz`, `.mgz`, `.mgh`
- JSON sidecars and filename entities

### MRI visualization

Qortex can render:

- orthogonal anatomical views
- slice-lightbox views
- fMRI QC summaries
- DWI QC summaries
- mask overlays
- statistical overlays
- PET-on-MRI overlays
- segmentation comparisons

---

## PET

PET datasets usually contain tracer-specific image data and metadata. Qortex treats PET as a quantitative imaging modality where metadata and units matter.

<div class="tq-card-grid">
  <div class="tq-card">
    <h3><a href="pet/">PET overview</a></h3>
    <p>How PET files are detected and how tracer metadata is handled.</p>
  </div>
  <div class="tq-card">
    <h3><a href="pet/metadata/">PET metadata</a></h3>
    <p>Tracer names, units, injection metadata, acquisition timing, and BIDS sidecars.</p>
  </div>
  <div class="tq-card">
    <h3><a href="pet/visualization/">PET visualization</a></h3>
    <p>PET volume viewing, PET colormaps, and PET overlay on anatomical images.</p>
  </div>
</div>

### PET detection

Qortex detects PET data from:

- BIDS datatype folder: `pet`
- suffix: `pet`
- tracer entities such as `trc-`
- NIfTI image files and PET JSON sidecars

### PET visualization

PET images can be shown as standalone volumes or overlaid on anatomical MRI when matching geometry is available.

---

## MEG

MEG datasets are signal-based. They usually depend on event files, channel metadata, head-position information, and acquisition-specific formats.

<div class="tq-card-grid">
  <div class="tq-card">
    <h3><a href="meg/">MEG overview</a></h3>
    <p>MEG file detection, event handling, and signal inspection.</p>
  </div>
  <div class="tq-card">
    <h3><a href="meg/files/">MEG files</a></h3>
    <p>Common MEG raw formats, sidecars, channel files, and BIDS metadata.</p>
  </div>
  <div class="tq-card">
    <h3><a href="meg/events/">MEG events</a></h3>
    <p>Events TSV files, trial labels, stimulus timing, and readiness for supervised learning.</p>
  </div>
</div>

### MEG detection

Qortex detects MEG data from:

- BIDS datatype folder: `meg`
- MEG raw file formats such as `.fif`
- MEG metadata sidecars
- `events.tsv`, `channels.tsv`, and related companion files

### MEG visualization

MEG visualization should focus on:

- short raw signal previews
- channel-level summaries
- event-aligned previews
- PSD and spectrogram views
- readiness for conversion into windowed ML samples

---

## EEG

EEG is common in OpenNeuro and often appears in several raw formats. Qortex should treat EEG as a signal modality where events, channels, sampling frequency, and labels are central.

<div class="tq-card-grid">
  <div class="tq-card">
    <h3><a href="eeg/">EEG overview</a></h3>
    <p>How Qortex detects EEG files and prepares them for inspection and conversion.</p>
  </div>
  <div class="tq-card">
    <h3><a href="eeg/files/">EEG files</a></h3>
    <p>EDF, BDF, BrainVision, EEGLAB, sidecars, channel metadata, and sampling metadata.</p>
  </div>
  <div class="tq-card">
    <h3><a href="eeg/events/">EEG events</a></h3>
    <p>Events TSV files, labels, trial timing, and event-aligned windows.</p>
  </div>
  <div class="tq-card">
    <h3><a href="eeg/visualization/">EEG visualization</a></h3>
    <p>Butterfly plots, PSD, spectrograms, epoched previews, and signal QC.</p>
  </div>
</div>

### EEG detection

Qortex detects EEG data from:

- BIDS datatype folder: `eeg`
- file formats such as `.edf`, `.bdf`, `.vhdr`, `.vmrk`, `.eeg`, `.set`, `.fdt`
- `channels.tsv`
- `events.tsv`
- EEG JSON sidecars

### EEG visualization

Qortex signal visualization should answer:

- Does the file load?
- How many channels are present?
- What is the sampling frequency?
- Are there obvious flat or noisy channels?
- Are events available?
- Can event windows be created for ML?

---

## iEEG

iEEG datasets are signal-based but often need electrode and coordinate metadata to be useful. Qortex should treat iEEG as both a signal and localization modality.

<div class="tq-card-grid">
  <div class="tq-card">
    <h3><a href="ieeg/">iEEG overview</a></h3>
    <p>Intracranial EEG detection, raw signals, electrodes, and coordinate metadata.</p>
  </div>
  <div class="tq-card">
    <h3><a href="ieeg/files/">iEEG files</a></h3>
    <p>Common iEEG signal formats, BIDS sidecars, events, and channel metadata.</p>
  </div>
  <div class="tq-card">
    <h3><a href="ieeg/electrodes-and-coordinates/">Electrodes and coordinates</a></h3>
    <p>Electrode tables, coordinate systems, anatomical references, and localization readiness.</p>
  </div>
</div>

### iEEG detection

Qortex detects iEEG data from:

- BIDS datatype folder: `ieeg`
- raw electrophysiology files
- `channels.tsv`
- `electrodes.tsv`
- `coordsystem.json`
- `events.tsv`

### iEEG visualization

For early support, iEEG visualization should focus on signal previews and metadata completeness. Later support can include electrode localization over anatomical images.

---

## fNIRS

fNIRS support is useful for BIDS completeness, but it should be presented honestly as an evolving modality in Qortex unless the loader and visualization paths are fully tested.

<div class="tq-card-grid">
  <div class="tq-card">
    <h3><a href="fnirs/">fNIRS overview</a></h3>
    <p>Functional near-infrared spectroscopy files, metadata, optodes, and readiness checks.</p>
  </div>
  <div class="tq-card">
    <h3><a href="fnirs/files/">fNIRS files</a></h3>
    <p>SNIRF files, channel metadata, optode metadata, and events.</p>
  </div>
</div>

### fNIRS detection

Qortex should detect fNIRS data from:

- BIDS datatype folder: `nirs`
- `.snirf` files
- `channels.tsv`
- `optodes.tsv`
- `events.tsv`
- fNIRS JSON sidecars

### Current status

fNIRS should be documented as metadata/readiness-first unless the full visualization and conversion path has test coverage.

---

## Behavioral and events

Behavioral and event files are central to Qortex because they often define labels for supervised learning.

<div class="tq-card-grid">
  <div class="tq-card">
    <h3><a href="behavioral/">Behavioral overview</a></h3>
    <p>Behavioral tables, task metadata, and non-imaging records.</p>
  </div>
  <div class="tq-card">
    <h3><a href="behavioral/events-tsv/">Events TSV</a></h3>
    <p>Onsets, durations, trial types, responses, conditions, and task labels.</p>
  </div>
  <div class="tq-card">
    <h3><a href="behavioral/labels-and-trial-types/">Labels and trial types</a></h3>
    <p>How Qortex detects candidate labels and evaluates supervised-learning readiness.</p>
  </div>
</div>

### Behavioral detection

Qortex detects behavioral and event data from:

- `events.tsv`
- `events.json`
- `beh` datatype folders
- `.tsv` and `.csv` tables
- `participants.tsv`
- task sidecars

### Why this matters

For many ML workflows, the imaging or signal file is not enough. The dataset becomes trainable only when Qortex can connect recordings to usable labels, split policy, and provenance.

---

## Cross-modality workflows

Some datasets contain multiple modalities. Qortex should not treat modalities as isolated silos. It should connect files through BIDS entities, companions, and logical recordings.

### Common cross-modality questions

- Which subjects have both T1w and BOLD?
- Which BOLD runs have matching events?
- Which DWI files have both bval and bvec companions?
- Which PET scans have anatomical references?
- Which EEG files have channels and events?
- Which iEEG files have electrodes and coordinate systems?
- Which files can be visualized before conversion?
- Which recordings can become ML samples?

### Relevant Qortex tools

```python
from qortex import Dataset

ds = Dataset("ds000001")

manifest = ds.manifest()
plan = ds.plan(subjects=["01"], modalities=["fmri"])
report = ds.visual_audit("qc/", suffixes=["T1w", "bold", "dwi"])
```
