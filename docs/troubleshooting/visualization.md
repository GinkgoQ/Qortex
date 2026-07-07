# Visualization Troubleshooting

## ImportError when calling visualize functions

```
ImportError: nibabel is required for NIfTI visualization.
Install with: pip install "qortex[mri]"
```

Install the appropriate extra:

```bash
pip install "qortex[mri]"       # NIfTI, fMRI, MRI
pip install "qortex[dwi]"       # DWI with bval/bvec
pip install "qortex[eeg]"       # EEG/MEG time series
pip install "qortex[visual]"    # interactive Plotly output
pip install "qortex[visual-all]" # everything
```

## Blank figures

If `result.show()` produces a blank figure in Jupyter:

1. Make sure `%matplotlib inline` or `%matplotlib widget` is set
2. For Plotly figures, check that the Plotly renderer is configured:

```python
import plotly.io as pio
pio.renderers.default = "notebook"  # or "browser"
```

## MemoryError during fMRI visualization

The `fmri_summary()` function uses Welford's algorithm to compute tSNR in a single pass with constant memory. However, loading a full 4D volume (e.g., for `lightbox()`) can require several GB.

Check how much memory the volume needs:

```python
viewer = VolumeViewer("sub-01_task-rest_bold.nii.gz", modality="fmri")
print(viewer.shape)           # (91, 109, 91, 200)
print(viewer.nbytes_3d)       # bytes for one 3D frame
print(viewer.nbytes_total)    # bytes for the full 4D volume
```

Use `fmri_summary()` instead of `lightbox()` for large files — it never loads the full volume.

## Figure shows wrong slice

The default center slice is `shape[axis] // 2`. For an asymmetric brain or a dataset in native space rather than MNI, the center may be in an unexpected location.

Pass an explicit slice index:

```python
viewer.ortho(z_idx=80)  # axial slice 80
```

## NIfTI header error

```
InvalidNIfTIHeaderError: Could not read NIfTI header from ...
```

The file may be:
- A Git LFS pointer (check with `content-status`)
- Corrupted (check file size vs manifest)
- In ANALYZE format (`.img`/`.hdr` pair) not supported




## Related

- [Local viewer](../visualization/local-viewer.md) — visualize local files
- [fMRI QC](../visualization/fmri-qc.md) — fMRI-specific QC panels
