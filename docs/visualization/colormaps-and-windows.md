# Colormaps and Windows

Qortex includes a set of display window presets for CT and MRI, and four pure-numpy colormap LUTs.

## CT window presets

CT images use Hounsfield units (HU). Different anatomical structures are visible at different HU ranges. Qortex provides ten presets:

| Preset | Center (HU) | Width (HU) | Use |
|--------|-------------|------------|-----|
| `brain` | 40 | 80 | Standard brain CT |
| `subdural` | 75 | 215 | Subdural hemorrhage |
| `stroke` | 32 | 8 | Early ischemic stroke |
| `bone` | 400 | 1800 | Skull and vertebrae |
| `soft_tissue` | 50 | 400 | General soft tissue |
| `lung` | -600 | 1500 | Lung parenchyma |
| `liver` | 60 | 160 | Liver and abdominal |
| `angio` | 300 | 600 | CT angiography |
| `abdomen` | 40 | 350 | Abdomen general |
| `pelvis` | 55 | 400 | Pelvis |

Each preset is a `WindowPreset` dataclass with `center` and `width` fields. The display window maps `center - width/2` to the bottom of the colormap and `center + width/2` to the top.

## Applying a preset

```python
from qortex.visualize._colors import CT_PRESETS, apply_window

preset = CT_PRESETS["brain"]
normalized = apply_window(voxel_data, center=preset.center, width=preset.width)
# normalized is float32 in [0.0, 1.0]
```

## Auto-windowing for MRI

For MRI, the appropriate window depends on the image content. `auto_window()` computes window center and width from the 2nd and 98th percentiles of non-zero voxels:

```python
from qortex.visualize._colors import auto_window

center, width = auto_window(voxel_data)
normalized = apply_window(voxel_data, center=center, width=width)
```

## Colormap LUTs

Qortex includes four colormap lookup tables as pure numpy arrays. These work without Pillow or matplotlib.

| Colormap | Shape | Use |
|----------|-------|-----|
| `gray` | (256, 3) float32 | Standard grayscale |
| `hot` | (256, 3) float32 | PET and temperature maps |
| `plasma` | (256, 3) float32 | Sequential with high contrast |
| `RdBu_r` | (256, 3) float32 | Diverging, for stat maps |

```python
from qortex.visualize._colors import get_lut

lut = get_lut("hot")             # shape (256, 3), values in [0, 1]
rgb = lut[(normalized * 255).astype(int)]  # apply to normalized voxel data
```

## Using presets in viewers

Viewers accept window presets by name:

```python
from qortex.visualize.volume import VolumeViewer

# CT with brain window
viewer = VolumeViewer("sub-01_ct.nii.gz", modality="ct", window="brain")
viewer.ortho().show()

# CT with custom window
viewer = VolumeViewer("sub-01_ct.nii.gz", modality="ct")
viewer.ortho(window_center=40, window_width=80).show()
```

For statistical maps, the colormap and threshold are set on the overlay call:

```python
from qortex.visualize import overlay_stat

result = overlay_stat(
    bg,
    stat_path="zstat.nii.gz",
    colormap="RdBu_r",
    threshold=2.3,
    vmax=8.0,
)
```

## MRI presets

In addition to CT, Qortex has presets for common MRI modalities:

```python
from qortex.visualize._colors import MR_PRESETS, FMRI_PRESETS, PET_PRESETS

mr_preset   = MR_PRESETS["T1w"]
fmri_preset = FMRI_PRESETS["bold"]
pet_preset  = PET_PRESETS["fdg"]
```

These are applied automatically when the modality is detected from the BIDS suffix.
