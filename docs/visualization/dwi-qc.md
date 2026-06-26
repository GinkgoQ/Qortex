# DWI QC

`dwi_summary()` produces a four-panel QC figure for a DWI acquisition. It shows b=0 volume, one high-b volume, b-value histogram, and gradient direction sphere.

## Basic usage

```python
from qortex.visualize import dwi_summary

result = dwi_summary(
    "sub-01_dwi.nii.gz",
    bval_path="sub-01_dwi.bval",
    bvec_path="sub-01_dwi.bvec",
)
result.show()
```

Or through DWIViewer directly:

```python
from qortex.visualize.dwi import DWIViewer

viewer = DWIViewer(
    "sub-01_dwi.nii.gz",
    bval_path="sub-01_dwi.bval",
    bvec_path="sub-01_dwi.bvec",
)
result = viewer.dwi_summary()
result.show()
```

## Four-panel layout

| Position | Panel | What it shows |
|----------|-------|---------------|
| Row 1, Left | b=0 volume | Mean of all b=0 (or low-b) volumes — center axial slice |
| Row 1, Right | High-b volume | One volume from the highest b-value shell |
| Row 2, Left | b-value histogram | Distribution of b-values, colored by shell |
| Row 2, Right | Gradient sphere | 3D sphere with gradient directions plotted as points |

## Individual panels

```python
viewer = DWIViewer("sub-01_dwi.nii.gz", bval_path="...", bvec_path="...")

b0_fig      = viewer.b0()            # b=0 center slice
high_b_fig  = viewer.high_b()        # highest b-value slice
hist_fig    = viewer.bval_histogram() # b-value distribution
sphere_fig  = viewer.gradient_sphere() # gradient directions on sphere
```

## Contact sheet

`contact_sheet()` produces a grid of axial slices, one per gradient direction, arranged in acquisition order:

```python
sheet_fig = viewer.contact_sheet(n_cols=10)
sheet_fig.show()
```

This is useful for identifying dropped volumes (all-zero or corrupted slices) or gradient encoding errors.

## CLI

```bash
qortex dwi-qc sub-01_dwi.nii.gz \
    --bval sub-01_dwi.bval \
    --bvec sub-01_dwi.bvec \
    --output figures/sub-01_dwi_qc.html
```

## Interpreting the panels

**b=0 volume.** Should show brain anatomy with T2-weighted contrast. Dark regions may indicate susceptibility artifacts (common near ear canals and prefrontal sinuses). Signal dropout here will affect all diffusion metrics derived from that voxel.

**High-b volume.** At high b-values (≥ 1000 s/mm²), signal drops dramatically. What you see is mostly noise plus diffusion-weighted contrast. The brain should still be roughly visible. An all-noise image suggests the high-b gradient was not applied correctly.

**b-value histogram.** For single-shell data, you expect two peaks: one near b=0 and one at the shell value. For multi-shell data (HCP style), you expect three or more peaks. An unexpected peak at an intermediate value may indicate gradient non-linearity or a protocol error.

**Gradient sphere.** Gradient directions should be approximately uniformly distributed on the hemisphere. Clustering in one region means poor angular coverage, which degrades reconstruction accuracy for fiber orientation distributions.

## Limitations

- `gradient_sphere()` uses matplotlib's 3D projection, not Plotly. The interactive rotation is limited.
- DWIViewer requires the `dwi` extra: `pip install "qortex[dwi]"`.
- Dipy is used internally for bval/bvec parsing and shell detection. If dipy is not installed, bval/bvec are read with numpy and shell detection falls back to a simple threshold.
