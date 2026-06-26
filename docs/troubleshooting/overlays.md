# Overlay Troubleshooting

## Affine mismatch error

```
AffineMismatchError: Background and overlay have different affines.
Background: [[2.0, 0, 0, -90], ...], Overlay: [[1.0, 0, 0, -90], ...]
```

The overlay is in a different voxel space from the background. Qortex does not resample volumes automatically.

Fix options:

1. **Resample before overlaying** using nilearn or ANTs:

```python
from nilearn.image import resample_to_img
import nibabel as nib

bg = nib.load("sub-01_T1w.nii.gz")
overlay = nib.load("sub-01_brain_mask.nii.gz")
overlay_resampled = resample_to_img(overlay, bg, interpolation="nearest")
overlay_resampled.to_filename("sub-01_brain_mask_resampled.nii.gz")
```

2. **Check if the volumes are the same space** but with floating-point rounding differences:

```python
import numpy as np
import nibabel as nib

bg = nib.load("sub-01_T1w.nii.gz")
ov = nib.load("sub-01_brain_mask.nii.gz")
print(np.allclose(bg.affine, ov.affine, atol=1e-3))  # True if nearly identical
```

If they are within floating-point tolerance, pass `check_affine=False`:

```python
from qortex.visualize import overlay_mask
result = overlay_mask(bg_result, "mask.nii.gz", check_affine=False)
```

## Overlay appears misaligned visually

If the overlay appears in the wrong position despite no affine error, the files may have different orientations (RAS vs. LAS). Qortex renders center slices in voxel coordinates, not world coordinates.

Check the orientation codes:

```python
import nibabel as nib
print(nib.aff2axcodes(nib.load("sub-01_T1w.nii.gz").affine))
print(nib.aff2axcodes(nib.load("mask.nii.gz").affine))
```

If they differ (e.g., `RAS` vs. `LAS`), reorient before overlaying.

## Mask shows no visible overlay

The mask may have all-zero values. Check:

```python
import nibabel as nib
import numpy as np
mask = nib.load("mask.nii.gz")
print(np.unique(mask.get_fdata()))  # should include non-zero values
print(np.sum(mask.get_fdata() > 0))  # number of non-zero voxels
```

Also check that the mask's center slice passes through non-zero voxels.

## Stat map threshold too high

If `overlay_stat` shows nothing, the threshold may be above all values:

```python
import nibabel as nib
import numpy as np
stat = nib.load("zstat.nii.gz")
print(np.max(np.abs(stat.get_fdata())))  # maximum absolute value
```

Lower the threshold:

```python
from qortex.visualize import overlay_stat
result = overlay_stat(bg, "zstat.nii.gz", threshold=1.5)  # lower threshold
```
