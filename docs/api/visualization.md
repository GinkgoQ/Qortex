# Visualization API

All visualization functions use lazy loading — only the slices or segments needed for the current view are read from disk.

```bash
pip install "qortex[visual-all]"   # Plotly + nibabel + MNE
pip install "qortex[mri]"          # nibabel only (MRI/fMRI/DWI)
pip install "qortex[eeg]"          # MNE only (EEG/MEG)
```

---

## Top-level functions

::: qortex.visualize
    options:
      show_source: false
      members:
        - inspect
        - visualize
        - browse_dicom
        - overlay_mask
        - overlay_labelmap
        - overlay_stat
        - overlay_pet
        - overlay_contour
        - overlay_edges
        - compare_masks
        - dwi_summary
        - fmri_summary
        - surface_summary
        - volume
        - timeseries
        - inspect_surface
        - find_hemisphere_pair
        - list_dicom_series
        - load_dicom_series

---

## Core result types

Every rendering call flows through these types: `inspect()` returns a `VisualAsset`, `asset.plan()` returns a `VisualPlan`, and `asset.render()` returns a `VisualResult`.

::: qortex.visualize.VisualWarning
    options:
      show_source: false

::: qortex.visualize.VisualAsset
    options:
      show_source: false
      members:
        - is_4d
        - is_dicom_series
        - size_voxels
        - estimated_memory_mb
        - has_errors
        - voxel_size_str
        - shape_str
        - warn
        - summary
        - plan
        - render
        - to_dict

::: qortex.visualize.VisualPlan
    options:
      show_source: false
      members:
        - estimated_memory_mb
        - describe

::: qortex.visualize.VisualResult
    options:
      show_source: false
      members:
        - show
        - to_html
        - to_png
        - to_provenance_dict

---

## VolumeViewer

Renders orthogonal slices, lightboxes, and QC figures for NIfTI volumes.

::: qortex.visualize.volume.VolumeViewer
    options:
      show_source: false
      members:
        - ortho
        - lightbox
        - timeseries_at
        - interactive_html
        - to_html
        - show
        - overlay
        - fmri_summary
        - mean_epi_figure
        - mean_volume
        - std_epi_figure
        - tsnr_figure
        - global_signal_timeseries
        - framewise_preview
        - n_volumes
        - shape
        - voxel_sizes
        - tr

---

## DWIViewer

Renders b0 images, high-b images, bval histograms, and gradient sphere plots.

::: qortex.visualize.dwi.DWIViewer
    options:
      show_source: false
      members:
        - b0
        - high_b
        - bval_histogram
        - gradient_sphere
        - dwi_summary
        - contact_sheet
        - b0_indices
        - high_b_indices
        - shells
        - n_volumes
        - warnings

---

## TimeSeriesViewer

Renders EEG, MEG, and iEEG signal plots.

::: qortex.visualize.timeseries.TimeSeriesViewer
    options:
      show_source: false
      members:
        - butterfly
        - psd
        - spectrogram
        - topomap
        - epoched
        - dashboard
        - to_html
        - show
        - n_channels
        - n_samples
        - sfreq
        - duration_s
        - channel_names

---

## VisualAuditReport

Coverage matrix and QC summary for a local BIDS directory.

::: qortex.visualize._audit.VisualAuditReport
    options:
      show_source: false
      members:
        - coverage_matrix
        - warning_summary
        - summary
        - per_suffix_counts
        - per_subject_counts
        - per_datatype_counts
        - n_expected
        - n_local_present
        - n_missing_local
        - failed_files
        - missing_expected_files
        - action_items
        - to_html
        - to_json
        - to_markdown
        - visual_manifest_json
        - show

---

## DICOM types

::: qortex.visualize.dicom.DicomSeries
    options:
      show_source: false
      members:
        - shape
        - spacing_str
        - to_dict

::: qortex.visualize.dicom.DicomSeriesBrowser
    options:
      show_source: false
      members:
        - scan
        - to_html

---

## Surface types

::: qortex.visualize.surface.SurfaceArrayInfo
    options:
      show_source: false

::: qortex.visualize.surface.SurfaceInfo
    options:
      show_source: false
