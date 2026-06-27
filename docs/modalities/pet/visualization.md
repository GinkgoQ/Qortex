# PET Visualization

PET volumes can be viewed as standalone quantitative images or overlaid on a structural MRI when co-registered anatomy is available.

## Install

```bash
pip install "qortex[mri,visual]"
```

## View a PET volume

```python
from qortex.visualize import VolumeViewer

viewer = VolumeViewer("data/ds001421/sub-01/pet/sub-01_trc-18FFDG_pet.nii.gz")
viewer.ortho()      # orthogonal view with PET colormap
viewer.lightbox()   # axial slice grid
```

PET volumes are rendered with the `hot` colormap by default (black → red → yellow → white). Change colormap:

```python
viewer.ortho(colormap="plasma")
```

## Overlay PET on structural MRI

When a T1w and a PET are co-registered (same space), overlay them:

```python
from qortex.visualize import overlay_pet

fig = overlay_pet(
    background="data/ds001421/sub-01/anat/sub-01_T1w.nii.gz",
    pet="data/ds001421/sub-01/pet/sub-01_trc-18FFDG_pet.nii.gz",
    alpha=0.6,
    colormap="hot",
    threshold=0.3,   # fraction of max value — hide low-uptake voxels
)
fig.show()
```

The overlay clips the PET at the threshold and applies the colormap on top of the grayscale anatomical. Qortex does not perform spatial registration — the two files must already share the same affine.

## CLI

```bash
qortex visualize-overlay sub-01/anat/sub-01_T1w.nii.gz \
    --overlay sub-01/pet/sub-01_trc-18FFDG_pet.nii.gz \
    --type pet \
    --alpha 0.6 \
    --output sub01_pet_overlay.html
```

## Frame-by-frame viewing

PET files are 4D when dynamic. To view a specific frame:

```python
viewer = VolumeViewer("sub-01/pet/sub-01_trc-18FFDG_pet.nii.gz")
viewer.ortho(volume_index=5)   # frame 5
```

## Related

- [PET metadata](metadata.md) — tracer, frames, SUV information
- [Overlays](../../visualization/overlays.md) — full overlay API reference
- [Colormaps and windows](../../visualization/colormaps-and-windows.md) — PET presets
