# Segmentation Comparison

`compare_masks()` renders a side-by-side or overlay comparison of two binary or integer-label masks on the same anatomical background. Use it to check agreement between manual and automated segmentations.

## Python

```python
from qortex.visualize import compare_masks

result = compare_masks(
    background="sub-01_T1w.nii.gz",
    mask_a="auto_seg.nii.gz",
    mask_b="manual_seg.nii.gz",
    labels={"a": "FreeSurfer", "b": "Manual"},
    mode="side-by-side",   # or "overlay"
)
result.show()
```

## Modes

**`side-by-side`** (default): Two sets of ortho panels are placed next to each other. The left panel shows mask A on the background; the right shows mask B.

**`overlay`**: Both masks are shown on the same background. Voxels where masks agree are green, where only A is present are blue, and where only B is present are red.

## CLI

```bash
qortex compare-masks sub-01_T1w.nii.gz \
    --mask-a auto_seg.nii.gz \
    --mask-b manual_seg.nii.gz \
    --labels FreeSurfer Manual \
    --mode overlay \
    --output seg_comparison.html
```

## Dice coefficient

The comparison result includes a Dice similarity coefficient:

```python
print(result.dice)           # float
print(result.jaccard)        # float
print(result.n_agree)        # voxel count where masks agree
print(result.n_only_a)       # voxels in A only
print(result.n_only_b)       # voxels in B only
```

For labelmaps with multiple integer labels, Dice is computed per label and returned as a dict:

```python
print(result.dice_per_label)  # {1: 0.92, 2: 0.87, 3: 0.79, ...}
```

## Requirements

`compare_masks` requires the `mri` extra: `pip install "qortex[mri]"`.

Both masks must be in the same voxel space as the background. Qortex does not resample.

## Related

- [Overlays](overlays.md) — general overlay functions
- [Visual audit](visual-audit.md) — coverage-level inspection across all subjects
