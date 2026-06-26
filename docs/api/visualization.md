# Visualization API

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

## VolumeViewer

::: qortex.visualize.volume.VolumeViewer
    options:
      show_source: false
      members:
        - ortho
        - lightbox
        - timeseries_at
        - interactive_html
        - fmri_summary
        - mean_epi_figure
        - std_epi_figure
        - tsnr_figure
        - global_signal_timeseries
        - framewise_preview

## DWIViewer

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

## TimeSeriesViewer

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

## VisualAuditReport

::: qortex.visualize._audit.VisualAuditReport
    options:
      show_source: false
      members:
        - coverage_matrix
        - warning_summary
        - per_suffix_counts
        - to_html
        - to_json
        - to_markdown
        - visual_manifest_json
        - action_items
        - missing_expected_files
        - show
