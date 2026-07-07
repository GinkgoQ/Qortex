# Modalities

Neurodata workflows fail in different places depending on the modality. EEG fails around channels, sampling, events, and labels. MRI fails around geometry, companions, and metadata. PET fails when units or tracer metadata are ambiguous. This section shows what Qortex checks for each data type and where to go next.

## Modality Map

| Modality | BIDS datatype | Primary risk | Qortex support |
|---|---|---|---|
| [EEG](eeg/index.md) | `eeg` | missing events, channel metadata, sampling mismatch, noisy channels | inspect, MNE-BIDS reads, windows, feature matrices, QC, visualization, conversion |
| [MEG](meg/index.md) | `meg` | event alignment, channel metadata, acquisition-specific formats | inspect, events, signal previews, conversion paths |
| [iEEG](ieeg/index.md) | `ieeg` | electrode coordinates, channel labels, localization metadata | inspect, channel/electrode tables, coordinate readiness |
| [fNIRS](fnirs/index.md) | `nirs` | source/detector metadata, channel geometry, file format support | inspect, MNE-backed loading, metadata checks |
| [Structural MRI](mri/structural-mri.md) | `anat` | orientation, spacing, missing masks, unreadable volumes | NIfTI streaming, ortho views, masks, overlays, QC |
| [fMRI / BOLD](mri/fmri-bold.md) | `func` | TR, event coverage, confounds, motion, 4D size | header reads, events, fMRI QC, visual audit |
| [DWI](mri/dwi-diffusion.md) | `dwi` | missing or mismatched `bval`/`bvec`, absent b0, shell errors | companion checks, shell summary, DWI QC |
| [Fieldmaps](mri/fieldmaps.md) | `fmap` | missing `IntendedFor`, phase metadata, distortion-correction readiness | entity and sidecar inspection |
| [PET](pet/index.md) | `pet` | tracer identity, units, timing, anatomical alignment | metadata checks, PET viewing, overlays |
| [Behavioral](behavioral/index.md) | `beh`, `events.tsv` | label columns, class balance, timing fields | readiness, label landscape, conversion |

## What Qortex Reads Before Download

| Evidence | Why it matters |
|---|---|
| BIDS entities | Subject, session, task, acquisition, run, datatype, suffix, extension. |
| Sidecars | Sampling frequency, TR, units, task metadata, image metadata, intended-for links. |
| Events | Label candidates, trial timing, class balance, subject coverage. |
| Participants | Demographics and grouping columns for split strategy and cohort checks. |
| File sizes | Minimum-download planning and large-file warnings. |
| NIfTI headers | Shape, dtype, affine, voxel spacing, TR, orientation, endian handling. |

## What Requires Local Files

| Check | Needs local data because |
|---|---|
| Signal QC | Flat channels, spectral summaries, entropy, and autocorrelation need samples. |
| Image QC | Overlays, masks, fMRI/DWI panels, and thumbnails need image bytes. |
| Label confirmation | Candidate labels from remote events should be checked against local content. |
| Conversion | Artifacts require local source files and companions. |
| NeuroAI inference | Source/model compatibility can be planned from profiles, but prediction needs data. |

## Choose A Page

<div class="tq-card-grid tq-card-grid-3">
  <div class="tq-card">
    <h3><a href="eeg/">EEG / PSG</a></h3>
    <p>EDF, BDF, BrainVision, EEGLAB, events, channels, windows, bandpower, PLV, CSP, and MNE-BIDS integration.</p>
  </div>
  <div class="tq-card">
    <h3><a href="mri/">MRI family</a></h3>
    <p>Structural MRI, fMRI, DWI, fieldmaps, NIfTI headers, sidecars, overlays, and modality-specific QC panels.</p>
  </div>
  <div class="tq-card">
    <h3><a href="pet/">PET</a></h3>
    <p>Tracer-aware metadata, units, timing, PET volumes, anatomical overlays, and conversion considerations.</p>
  </div>
  <div class="tq-card">
    <h3><a href="meg/">MEG</a></h3>
    <p>MEG file structures, events, channels, signal inspection, and conversion readiness.</p>
  </div>
  <div class="tq-card">
    <h3><a href="ieeg/">iEEG</a></h3>
    <p>Intracranial signals, electrodes, coordinate systems, and localization metadata.</p>
  </div>
  <div class="tq-card">
    <h3><a href="behavioral/">Behavioral</a></h3>
    <p>Events tables, labels, trial types, timing, class balance, and supervised-learning readiness.</p>
  </div>
</div>

## Practical Rule

Start with the file family that carries the model input, then inspect its companions. For example, a DWI file without matching `bval` and `bvec` is not a usable diffusion sample; an EEG file without events may still support unsupervised QC but not event-aligned classification; a PET image without tracer and unit metadata is hard to compare across subjects.
