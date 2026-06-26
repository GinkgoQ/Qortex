# Visual Pipeline

Qortex visualization is organized around three data classes: `VisualAsset`, `VisualPlan`, and `VisualResult`. Understanding their relationships explains how rendering is dispatched.

## VisualAsset

A `VisualAsset` represents one file to be visualized. It combines a file path with an intent classification that tells the renderer what kind of figure to produce.

```python
from qortex.visualize._asset import VisualAsset, INTENT_BOLD

asset = VisualAsset(
    path=Path("sub-01/func/sub-01_task-rest_bold.nii.gz"),
    intent=INTENT_BOLD,
    subject="01",
    task="rest",
    metadata={"TR": 2.0},
)
```

### Intent constants

| Constant | Meaning |
|----------|---------|
| `INTENT_ANATOMICAL` | Structural MRI (T1w, T2w) |
| `INTENT_BOLD` | fMRI BOLD series |
| `INTENT_DWI` | Diffusion-weighted imaging |
| `INTENT_PET` | PET tracer image |
| `INTENT_CT` | CT volume |
| `INTENT_FIELDMAP` | B0 or magnitude fieldmap |
| `INTENT_MASK` | Binary mask |
| `INTENT_LABELMAP` | Integer-valued segmentation label map |
| `INTENT_STAT_MAP` | Statistical map (z-score, t-stat) |
| `INTENT_SURFACE` | Surface geometry (GIFTI/CIFTI) |
| `INTENT_SERIES_BROWSER` | DICOM series |
| `INTENT_RAW_SIGNAL` | EEG/MEG time series |
| `INTENT_UNKNOWN` | Unrecognized file type |

Intent is assigned automatically by `qortex.visualize.inspect()` based on suffix, extension, and sidecar JSON.

## VisualPlan

A `VisualPlan` is a list of `VisualAsset` objects grouped by intent. The plan is produced by the `inspect()` function in `qortex.visualize`:

```python
from qortex.visualize import inspect

plan = inspect("data/ds004130/sub-01/")
for asset in plan.assets:
    print(asset.path, asset.intent)
```

The plan does not render anything — it only decides what to render and in what order.

## VisualResult

A `VisualResult` wraps one rendered output. It has:

```python
result.asset         # the VisualAsset that produced this result
result.figure        # matplotlib Figure or plotly Figure
result.intent        # same as asset.intent
result.warnings      # list of strings
result.show()        # display inline (Jupyter) or open browser
result.to_html(path) # save as interactive HTML (plotly)
result.to_png(path)  # save as PNG (matplotlib)
result.to_provenance_dict()  # dict with asset, params, qortex version
```

## High-level entry points

Most users do not interact with these classes directly. The high-level functions in `qortex.visualize` handle the full pipeline:

```python
from qortex.visualize import visualize, fmri_summary, dwi_summary

# Visualize all files in a directory
results = visualize("data/ds004130/sub-01/")

# Visualize a specific BOLD file
result = fmri_summary("sub-01_task-rest_bold.nii.gz")

# Visualize DWI
result = dwi_summary("sub-01_dwi.nii.gz", bval_path="sub-01_dwi.bval", bvec_path="sub-01_dwi.bvec")
```

## Adding overlays

Overlays are attached to an existing `VisualResult` through the overlay functions:

```python
from qortex.visualize import overlay_mask

result = visualize("sub-01_T1w.nii.gz")
result = overlay_mask(result, mask_path="sub-01_brain_mask.nii.gz", alpha=0.4, color="red")
result.show()
```
