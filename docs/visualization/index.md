# Visualization

Qortex includes visualization tools for inspecting neuroimaging data before and after download. All viewers are designed around lazy loading: only the slices needed for the current view are read from disk.

## Design principles

**One center slice per NIfTI, not the whole volume.** NIfTI files can be several hundred MB. Loading one at full resolution for a thumbnail is wasteful. Qortex reads only the center slice along each axis using nibabel's ArrayProxy.

**Plotly for interactive HTML, matplotlib fallback for static figures.** The `visual` extra adds Plotly. Without it, static PNG output is available via matplotlib.

**No Pillow required for basic rendering.** Colormaps (gray, hot, plasma, RdBu_r) are implemented as pure numpy LUTs.

## Viewers

[**Visual pipeline**](visual-pipeline.md) — How VisualAsset, VisualPlan, and VisualResult compose a rendering pass.

[**Local viewer**](local-viewer.md) — `visualize()` on a local BIDS directory. Renders one figure per file type.

[**Visualize OpenNeuro**](visualize-openneuro.md) — `visualize-openneuro` CLI command. Fetches center slices from CDN without full download.

[**Visual audit**](visual-audit.md) — `VisualAuditReport` coverage matrix, warnings, and action items.

[**fMRI QC**](fmri-qc.md) — `fmri_summary()` 6-panel QC figure with optional confound overlays.

[**DWI QC**](dwi-qc.md) — `dwi_summary()` 4-panel QC and gradient sphere.

[**Overlays**](overlays.md) — Mask, labelmap, stat map, PET, and contour overlays on anatomical background.

[**Segmentation comparison**](segmentation-comparison.md) — `compare_masks()` for side-by-side label comparison.

[**DICOM browser**](dicom-browser.md) — `browse_dicom()` for series-level DICOM inspection.

[**EEG viewer**](eeg-viewer.md) — `TimeSeriesViewer` butterfly, PSD, spectrogram, topomap, and ERP.

[**Artifact visualization**](artifact-visualization.md) — `Artifact.visualize_sample()` for ML-format artifact inspection.

[**Colormaps and windows**](colormaps-and-windows.md) — CT window presets and MRI window auto-detection.

## Requirements

```bash
pip install "qortex[visual-all]"
```

For MRI/fMRI/DWI viewers only:

```bash
pip install "qortex[mri,dwi]"
```

For EEG viewers only:

```bash
pip install "qortex[eeg]"
```
