# fMRI QC

`fmri_summary()` produces a multi-panel QC figure for a single BOLD NIfTI file. It is designed to catch obvious acquisition failures and motion artifacts before investing time in preprocessing or training.

## Basic usage

```python
from qortex.visualize import fmri_summary

result = fmri_summary("sub-01_task-rest_bold.nii.gz")
result.show()
```

Or through VolumeViewer:

```python
from qortex.visualize.volume import VolumeViewer

viewer = VolumeViewer("sub-01_task-rest_bold.nii.gz", modality="fmri")
result = viewer.fmri_summary()
result.show()
```

## Six-panel layout

The summary produces six panels arranged in a 3×2 grid:

| Position | Panel | What it shows |
|----------|-------|---------------|
| Row 1, Left | Mean EPI | Time-averaged volume — center axial slice |
| Row 1, Right | Middle frame | One frame at timepoint N/2 — shows TR-level contrast |
| Row 2, Left | Temporal std | Voxel-wise standard deviation across time |
| Row 2, Right | tSNR | Temporal signal-to-noise ratio (mean / std), per voxel |
| Row 3, Left | Global signal | Mean signal across all brain voxels, one value per TR |
| Row 3, Right | Framewise intensity | Mean signal per frame — useful for detecting volume dropouts |

### How tSNR and std are computed

tSNR and std are computed using Welford's online algorithm. This processes the 4D volume one 3D frame at a time, keeping only one frame in memory at a time. Peak memory = 3 × one 3D frame (running mean, running M2, current frame).

For a 91×109×91 MNI-space BOLD at float32: one frame is ~3.6 MB, so peak memory during tSNR computation is ~11 MB regardless of the number of timepoints.

## With events overlay

Pass `events_path` to overlay trial onset markers on the global signal and framewise intensity plots:

```python
result = fmri_summary(
    "sub-01_task-rest_bold.nii.gz",
    events_path="sub-01_task-rest_events.tsv",
)
result.show()
```

Onset times are converted to TR units and drawn as vertical lines color-coded by `trial_type`.

## With confounds (motion QC)

Pass `confounds_path` to add a third row of panels showing head motion metrics:

```python
result = fmri_summary(
    "sub-01_task-rest_bold.nii.gz",
    events_path="sub-01_task-rest_events.tsv",
    confounds_path="sub-01_task-rest_desc-confounds_timeseries.tsv",
)
result.show()
```

The confounds row shows three panels:

- **Framewise displacement** — head motion in mm per TR (from `framewise_displacement` column)
- **DVARS** — signal change magnitude per TR (from `dvars` column)
- **Std DVARS** — standardized DVARS (from `std_dvars` column)

A horizontal reference line at FD = 0.5 mm marks the common motion censoring threshold.

The confounds file must be an fMRIPrep-format TSV with a header row. Qortex reads only the three columns above and ignores all others.

## CLI

```bash
qortex fmri-qc sub-01_task-rest_bold.nii.gz
qortex fmri-qc sub-01_task-rest_bold.nii.gz \
    --events sub-01_task-rest_events.tsv \
    --confounds sub-01_task-rest_desc-confounds_timeseries.tsv \
    --output figures/sub-01_fmri_qc.html
```

## Individual panel methods

If you need individual panels rather than the combined figure:

```python
viewer = VolumeViewer("sub-01_task-rest_bold.nii.gz", modality="fmri")

mean_fig   = viewer.mean_epi_figure()
std_fig    = viewer.std_epi_figure()
tsnr_fig   = viewer.tsnr_figure()
gs_fig     = viewer.global_signal_timeseries()
frame_fig  = viewer.framewise_preview()
```

Each returns a matplotlib Figure.

## Interpreting the panels

**Mean EPI.** Look for signal dropout in frontal or temporal regions (dark voids where there should be brain tissue). Common near air–tissue interfaces.

**Temporal std.** High variance at brain edges indicates motion. High variance in deep structures may indicate pulsatility artifacts.

**tSNR.** Typical cortical tSNR for a 3T resting-state acquisition is 50–200. Values below 30 in cortex suggest poor shimming, strong motion, or thermal noise issues.

**Global signal.** A sudden step change in global signal (large jump at one TR) indicates a within-scanner event (e.g., the subject sneezed). Slow drift suggests scanner instability or physiological noise.

**Framewise displacement.** Volumes with FD > 0.5 mm are typically censored before analysis. If more than 20% of volumes exceed this threshold, consider excluding the subject.








<!-- qortex-evidence:start -->

## Evidence

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/ds000001-bold-axial.png" alt="Axial BOLD slice from OpenNeuro ds000001 subject 01 run 01.">
  <figcaption>Real BOLD axial slice streamed with `Dataset.stream_slice()` without downloading the full NIfTI file.</figcaption>
</figure>

```python
sl = ds.stream_slice(subject='01', modality='bold', run='01', time_index=0, axis=2)
```

Result artifact: [ds000001-example-results.json](/Qortex/assets/results/ds000001-example-results.json)

<!-- qortex-evidence:end -->

## Related

- [DWI QC](dwi-qc.md) — equivalent panels for diffusion data
- [Overlays](overlays.md) — add brain mask or ROI overlays to the mean EPI
