# Overlays

Qortex can overlay masks, labelmaps, statistical maps, PET tracers, contours, and edge maps on anatomical or functional backgrounds.

## Available overlay functions

```python
from qortex.visualize import (
    overlay_mask,
    overlay_labelmap,
    overlay_stat,
    overlay_pet,
    overlay_contour,
    overlay_edges,
)
```

All overlay functions take an existing `VisualResult` as the first argument and return a new `VisualResult` with the overlay applied.

## Mask overlay

```python
from qortex.visualize import visualize, overlay_mask

bg = visualize("sub-01_T1w.nii.gz")
result = overlay_mask(
    bg,
    mask_path="sub-01_brain_mask.nii.gz",
    color="red",
    alpha=0.4,
)
result.show()
```

Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `color` | `"red"` | Color for mask voxels (any matplotlib color) |
| `alpha` | `0.4` | Transparency (0 = invisible, 1 = opaque) |

## Labelmap overlay

```python
result = overlay_labelmap(
    bg,
    labelmap_path="sub-01_dseg.nii.gz",
    lut=None,   # auto-assign colors
    alpha=0.6,
)
result.show()
```

Each integer label gets a distinct color. By default, colors are assigned from a fixed palette that matches the FreeSurfer colormap for common parcellation indices. Pass a custom `lut` dict (`{integer_label: color}`) to override.

## Statistical map overlay

```python
result = overlay_stat(
    bg,
    stat_path="contrast_zstat.nii.gz",
    threshold=2.3,          # hide values below this absolute value
    colormap="RdBu_r",      # diverging colormap
    vmax=None,              # auto from 99th percentile
)
result.show()
```

Values below `threshold` are rendered transparent. The colormap center (white) is at zero for diverging maps.

## PET overlay

```python
result = overlay_pet(
    bg,
    pet_path="sub-01_trc-FDG_pet.nii.gz",
    colormap="hot",
    alpha=0.7,
)
result.show()
```

The `hot` colormap (black → red → yellow → white) is standard for PET visualization.

## Contour overlay

```python
result = overlay_contour(
    bg,
    mask_path="sub-01_brain_mask.nii.gz",
    color="cyan",
    linewidth=1,
)
result.show()
```

A contour draws only the boundary of the mask rather than filling it, which keeps the background anatomy visible.

## Edge overlay

```python
result = overlay_edges(
    bg,
    edge_path="sub-01_T2w.nii.gz",
    color="yellow",
    alpha=0.8,
)
result.show()
```

`overlay_edges` computes a Sobel edge map from the input volume and overlays it. Useful for checking T1/T2 registration alignment.

## Chaining overlays

Multiple overlays can be chained:

```python
result = visualize("sub-01_T1w.nii.gz")
result = overlay_mask(result, "sub-01_brain_mask.nii.gz", color="blue", alpha=0.2)
result = overlay_contour(result, "sub-01_wm_mask.nii.gz", color="white")
result = overlay_stat(result, "contrast_z.nii.gz", threshold=3.0)
result.show()
```

## Segmentation comparison

For side-by-side comparison of two labelmaps (e.g., automatic vs. manual segmentation), use `compare_masks()`:

```python
from qortex.visualize import compare_masks

result = compare_masks(
    "sub-01_T1w.nii.gz",
    mask_a="auto_seg.nii.gz",
    mask_b="manual_seg.nii.gz",
    labels={"a": "FreeSurfer", "b": "Manual"},
)
result.show()
```

See [Segmentation comparison](segmentation-comparison.md) for details.

## Limitations

- All overlay volumes must be in the same voxel space as the background image. Qortex does not resample.
- Overlay rendering uses matplotlib. For interactive HTML output, overlays are rasterized and embedded in the Plotly figure as a background image.
